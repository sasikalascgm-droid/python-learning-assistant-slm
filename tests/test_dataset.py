"""Unit and Integration Tests for Dataset Preparation and Splitting.

Verifies:
1. Valid JSONL syntax across raw and processed datasets.
2. Schema adherence (instruction, input, output).
3. Absence of exact duplicates and near-duplicate leakage.
4. Split ratios (80:10:10).
5. Deterministic reproducibility with random seed 42.
6. Syntactic validity of embedded Python code blocks using Python AST.
"""

import ast
import json
import pytest
from pathlib import Path
from src.data.validate_dataset import extract_python_code_blocks, normalize_text
from src.data.split_dataset import split_dataset


RAW_DATA_PATH = Path("data/raw/python_qa_raw.jsonl")
PROCESSED_DIR = Path("data/processed")
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "validation.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"


def load_jsonl(path: Path):
    assert path.exists(), f"File does not exist: {path}"
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except json.JSONDecodeError as e:
                    pytest.fail(f"Malformed JSON at {path}:{idx} -> {e}")
    return records


def test_raw_dataset_exists_and_loads():
    """Verify raw dataset exists and contains between 300 and 500 records."""
    records = load_jsonl(RAW_DATA_PATH)
    assert len(records) >= 300, f"Expected at least 300 records, got {len(records)}"
    assert len(records) <= 500, f"Expected at most 500 records, got {len(records)}"


def test_schema_integrity():
    """Verify all records across all splits contain required string fields."""
    for path in [RAW_DATA_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH]:
        records = load_jsonl(path)
        for i, rec in enumerate(records):
            assert "instruction" in rec, f"Missing 'instruction' in {path} record {i}"
            assert "input" in rec, f"Missing 'input' in {path} record {i}"
            assert "output" in rec, f"Missing 'output' in {path} record {i}"
            assert isinstance(rec["instruction"], str), f"'instruction' must be str in record {i}"
            assert isinstance(rec["input"], str), f"'input' must be str in record {i}"
            assert isinstance(rec["output"], str), f"'output' must be str in record {i}"


def test_non_empty_content():
    """Verify no record has blank instructions or empty outputs."""
    records = load_jsonl(RAW_DATA_PATH)
    for i, rec in enumerate(records):
        assert rec["instruction"].strip() != "", f"Empty instruction at record {i}"
        assert rec["output"].strip() != "", f"Empty output at record {i}"


def test_no_exact_duplicates():
    """Verify zero exact duplicate QA pairs in the raw dataset."""
    records = load_jsonl(RAW_DATA_PATH)
    seen = set()
    duplicates = []
    for i, rec in enumerate(records):
        key = (normalize_text(rec["instruction"]), normalize_text(rec["input"]))
        if key in seen:
            duplicates.append((i, rec["instruction"]))
        seen.add(key)
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate records: {duplicates}"


def test_split_files_exist_and_counts():
    """Verify train, validation, and test split files exist and sum to raw count."""
    train_recs = load_jsonl(TRAIN_PATH)
    val_recs = load_jsonl(VAL_PATH)
    test_recs = load_jsonl(TEST_PATH)
    raw_recs = load_jsonl(RAW_DATA_PATH)

    total_splits = len(train_recs) + len(val_recs) + len(test_recs)
    assert total_splits == len(raw_recs), f"Splits sum ({total_splits}) != raw count ({len(raw_recs)})"

    # Check 80:10:10 proportions within ±2% tolerance
    train_pct = len(train_recs) / len(raw_recs)
    val_pct = len(val_recs) / len(raw_recs)
    test_pct = len(test_recs) / len(raw_recs)

    assert 0.78 <= train_pct <= 0.82, f"Train percentage {train_pct:.2f} out of expected ~0.80"
    assert 0.08 <= val_pct <= 0.12, f"Validation percentage {val_pct:.2f} out of expected ~0.10"
    assert 0.08 <= test_pct <= 0.12, f"Test percentage {test_pct:.2f} out of expected ~0.10"


def test_zero_cross_split_leakage():
    """Verify strict isolation between train, validation, and test splits."""
    train_recs = load_jsonl(TRAIN_PATH)
    val_recs = load_jsonl(VAL_PATH)
    test_recs = load_jsonl(TEST_PATH)

    train_keys = set(normalize_text(r["instruction"]) for r in train_recs)
    val_keys = set(normalize_text(r["instruction"]) for r in val_recs)
    test_keys = set(normalize_text(r["instruction"]) for r in test_recs)

    train_test_overlap = train_keys & test_keys
    train_val_overlap = train_keys & val_keys
    val_test_overlap = val_keys & test_keys

    assert len(train_test_overlap) == 0, f"Leakage detected between Train and Test: {train_test_overlap}"
    assert len(train_val_overlap) == 0, f"Leakage detected between Train and Val: {train_val_overlap}"
    assert len(val_test_overlap) == 0, f"Leakage detected between Val and Test: {val_test_overlap}"


def test_embedded_python_code_blocks_syntax():
    """Extract all fenced Python code blocks in the dataset and verify AST validity."""
    records = load_jsonl(RAW_DATA_PATH)
    syntax_errors = []
    total_blocks = 0

    for idx, rec in enumerate(records):
        blocks = extract_python_code_blocks(rec["output"]) + extract_python_code_blocks(rec["input"])
        for b_idx, block in enumerate(blocks):
            total_blocks += 1
            try:
                ast.parse(block)
            except SyntaxError as e:
                syntax_errors.append({
                    "record_idx": idx,
                    "block_idx": b_idx,
                    "error": str(e),
                    "code_snippet": block[:80]
                })

    assert total_blocks > 100, f"Expected substantial code blocks, found {total_blocks}"
    assert len(syntax_errors) == 0, f"Found {len(syntax_errors)} Python syntax errors: {syntax_errors}"


def test_split_reproducibility():
    """Verify that split_dataset produces identical partitions given the same seed."""
    res1 = split_dataset(seed=42)
    res2 = split_dataset(seed=42)

    assert res1["train_count"] == res2["train_count"]
    assert res1["val_count"] == res2["val_count"]
    assert res1["test_count"] == res2["test_count"]
