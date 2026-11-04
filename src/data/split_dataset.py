"""Dataset Splitting Module for Python Learning Assistant.

Partitions validated raw QA records into Train (80%), Validation (10%), and Test (10%) splits.
Key safeguards:
1. Fixed random seed (seed=42) for exact reproducibility.
2. Leakage prevention: groups near-duplicate questions so paraphrases/variants stay in the same split.
3. Strict separation of held-out test split.
4. Outputs validated JSONL files to data/processed/ and root data/.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate_dataset import compute_jaccard_similarity, normalize_text


def cluster_near_duplicates(records: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[List[Dict[str, Any]]]:
    """Groups records with high instruction similarity into clusters to prevent cross-split leakage."""
    n = len(records)
    visited = [False] * n
    clusters: List[List[Dict[str, Any]]] = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [records[i]]
        visited[i] = True
        inst_i = records[i]["instruction"]

        for j in range(i + 1, n):
            if not visited[j]:
                inst_j = records[j]["instruction"]
                sim = compute_jaccard_similarity(inst_i, inst_j)
                if sim >= similarity_threshold:
                    cluster.append(records[j])
                    visited[j] = True
        clusters.append(cluster)

    return clusters


def split_dataset(
    raw_path: str = "data/raw/python_qa_raw.jsonl",
    output_dir: str = "data/processed",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, Any]:
    """Splits dataset into 80/10/10 train/val/test splits without leakage."""
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    records = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    total_records = len(records)
    print(f"Loaded {total_records} raw records for splitting.")

    # 1. Cluster to prevent leakage of near-duplicates
    clusters = cluster_near_duplicates(records, similarity_threshold=0.85)
    print(f"Identified {len(clusters)} distinct clusters from {total_records} records.")

    # 2. Deterministic shuffle of clusters
    rng = random.Random(seed)
    rng.shuffle(clusters)

    # 3. Target counts
    target_val = int(total_records * val_ratio)
    target_test = int(total_records * test_ratio)

    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    train_records: List[Dict[str, Any]] = []

    # Assign clusters greedily to val and test until targets met, remainder to train
    for cluster in clusters:
        if len(val_records) < target_val:
            val_records.extend(cluster)
        elif len(test_records) < target_test:
            test_records.extend(cluster)
        else:
            train_records.extend(cluster)

    # 4. Save to processed directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = out_dir / "train.jsonl"
    val_file = out_dir / "validation.jsonl"
    test_file = out_dir / "test.jsonl"

    def write_jsonl(file_path: Path, data: List[Dict[str, Any]]):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(train_file, train_records)
    write_jsonl(val_file, val_records)
    write_jsonl(test_file, test_records)

    # Also mirror to data/ for backward compatibility with root configs
    data_root = Path("data")
    write_jsonl(data_root / "train.jsonl", train_records)
    write_jsonl(data_root / "validation.jsonl", val_records)
    write_jsonl(data_root / "test.jsonl", test_records)

    # 5. Verify zero cross-split exact instruction leakage
    train_insts = set(normalize_text(r["instruction"]) for r in train_records)
    val_insts = set(normalize_text(r["instruction"]) for r in val_records)
    test_insts = set(normalize_text(r["instruction"]) for r in test_records)

    train_val_overlap = train_insts & val_insts
    train_test_overlap = train_insts & test_insts
    val_test_overlap = val_insts & test_insts

    print(f"Split results:")
    print(f"  Train      : {len(train_records)} records ({len(train_records)/total_records*100:.1f}%) -> {train_file}")
    print(f"  Validation : {len(val_records)} records ({len(val_records)/total_records*100:.1f}%) -> {val_file}")
    print(f"  Test       : {len(test_records)} records ({len(test_records)/total_records*100:.1f}%) -> {test_file}")
    print(f"  Overlap Check: Train-Test={len(train_test_overlap)}, Train-Val={len(train_val_overlap)}, Val-Test={len(val_test_overlap)}")

    return {
        "total_records": total_records,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "train_percent": round(len(train_records) / total_records * 100, 2),
        "val_percent": round(len(val_records) / total_records * 100, 2),
        "test_percent": round(len(test_records) / total_records * 100, 2),
        "leakage_detected": bool(train_val_overlap or train_test_overlap or val_test_overlap),
        "paths": {
            "train": str(train_file),
            "validation": str(val_file),
            "test": str(test_file)
        }
    }


if __name__ == "__main__":
    split_dataset()
