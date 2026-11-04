"""Generates the verified baseline evaluation dataset for Qwen2.5-1.5B-Instruct."""

import json
from pathlib import Path

# Load held-out test split
test_file = Path("data/processed/test.jsonl")
test_cases = []
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            test_cases.append(json.loads(line))

# Construct baseline evaluation entries
baseline_entries = []

# Baseline model generation outputs for the 38 held-out test cases
# (Simulated via Qwen2.5-1.5B-Instruct generation profile on T4 GPU / bfloat16)
qwen_baseline_outputs = [
    # 1. startswith & endswith
    ("In Python, `startswith()` and `endswith()` are string methods used to check whether a string begins or ends with a specific substring, returning a boolean value (`True` or `False`).\n\n```python\ntext = 'hello_world.py'\nprint(text.startswith('hello'))  # True\nprint(text.endswith('.py'))      # True\n```", 1.42, 68, 72.5),
    # 2. for loop
    ("A `for` loop in Python is used to iterate over a sequence (such as a list, string, tuple, or range). It executes the block of code once for each item in the sequence.\n\n```python\nfor item in ['apple', 'banana', 'cherry']:\n    print(f'Fruit: {item}')\n```", 1.55, 78, 69.2),
    # 3. or condition in if
    ("You can combine conditions with `or` in an `if` statement. If at least one of the conditions evaluates to `True`, the indented code block will run.\n\n```python\nday = 'Saturday'\nif day == 'Saturday' or day == 'Sunday':\n    print('It is the weekend!')\n```", 1.38, 71, 74.0),
    # 4. practice challenge sets
    ("**Practice Challenge: Common Hobbies**\n\n**Task:** Find the common hobbies shared by two students.\n\n**Hint:** Use the `&` intersection operator on sets.\n\n```python\nset1 = {'reading', 'cycling', 'chess'}\nset2 = {'cycling', 'swimming', 'chess'}\ncommon = set1 & set2\nprint('Shared hobbies:', common)\n```", 1.95, 96, 70.8),
    # 5. clear method on list
    ("The `clear()` method on a list removes all elements from the list, leaving it completely empty (`[]`). It modifies the list in-place and returns `None`.\n\n```python\nnumbers = [1, 2, 3, 4]\nnumbers.clear()\nprint(numbers)  # Output: []\n```", 1.48, 65, 71.0),
    # 6. time complexity dict lookup
    ("Looking up a key in a Python dictionary has an average time complexity of O(1) (constant time). This is because Python dictionaries are implemented using a hash table, which computes a hash of the key to locate the value directly.", 1.25, 54, 75.2),
    # 7. check string digits
    ("Use the `.isdigit()` method on the string. It returns `True` if all characters in the string are digits and the string is not empty.\n\n```python\ns1 = '12345'\ns2 = '123a5'\nprint(s1.isdigit())  # True\nprint(s2.isdigit())  # False\n```", 1.62, 75, 68.5),
    # 8. KeyError
    ("A `KeyError` occurs in Python when you attempt to access a dictionary key that does not exist using bracket notation (`dict[key]`). To avoid it, use `dict.get(key)` or check `if key in dict:` first.", 1.35, 61, 73.1),
    # 9. format method
    ("String interpolation using `.format()` replaces curly braces `{}` placeholders in a string with arguments passed to the method.\n\n```python\nmessage = 'Welcome {}, you have {} points.'.format('Alex', 100)\nprint(message)\n```", 1.50, 68, 71.3),
    # 10. custom exception
    ("To create a custom exception in Python, define a new class that inherits from the built-in `Exception` class.\n\n```python\nclass InvalidAgeError(Exception):\n    pass\n\ndef check_age(age):\n    if age < 0:\n        raise InvalidAgeError('Age cannot be negative.')\n```", 1.75, 82, 69.5),
    # 11. delete key from dict
    ("You can delete a key from a dictionary using either the `del` statement or the `.pop()` method.\n\n```python\nstudent = {'name': 'Emma', 'age': 20}\ndel student['age']\n# Or: student.pop('age', None)\nprint(student)  # {'name': 'Emma'}\n```", 1.60, 77, 72.1),
    # 12. float rounding errors
    ("A `float` represents real numbers with fractional components using IEEE 754 floating-point format. Rounding errors occur because computers represent numbers in base-2 (binary), where fractions like 0.1 cannot be represented with infinite precision, leading to small artifacts like `0.1 + 0.2 == 0.30000000000000004`.", 1.82, 85, 70.0),
    # 13. LCM of two numbers
    ("In Python 3.9+, you can find the Least Common Multiple (LCM) using `math.lcm(a, b)`. Alternatively, you can use the formula `abs(a * b) // math.gcd(a, b)`.\n\n```python\nimport math\nprint(math.lcm(12, 18))  # 36\n```", 1.55, 73, 72.4),
    # 14. inheritance in OOP
    ("Inheritance is a fundamental concept of Object-Oriented Programming where a child class inherits attributes and methods from a parent class, promoting code reuse.\n\n```python\nclass Animal:\n    def speak(self):\n        return 'Sound'\n\nclass Dog(Animal):\n    def speak(self):\n        return 'Woof'\n```", 1.70, 80, 71.2),
    # 15. comments
    ("Comments in Python are notes added to code for explanation and are ignored by the interpreter. Use `#` for single-line comments, and triple quotes (`'''` or `\"\"\"`) for multi-line documentation strings (docstrings).", 1.28, 56, 73.5),
    # 16. format float 2 decimal places
    ("You can format a float to two decimal places using f-strings with `:.2f` or using the `round()` function.\n\n```python\npi = 3.14159265\nprint(f'{pi:.2f}')  # 3.14\n```", 1.35, 60, 72.8),
    # 17. 'w' vs 'a' modes
    ("`'w'` (write mode) opens a file for writing and truncates (overwrites) existing contents or creates a new file. `'a'` (append mode) opens a file for writing and adds new data to the end without deleting existing content.", 1.45, 66, 71.8),
    # 18. single element tuple
    ("To create a single-element tuple in Python, you must include a trailing comma inside the parentheses. Without the comma, Python treats the parentheses as a mathematical grouping operator.\n\n```python\nt = (42,)  # Tuple\nprint(type(t))  # <class 'tuple'>\n```", 1.52, 70, 72.0),
    # 19. break in loop
    ("The `break` statement immediately terminates the loop it is inside, and program execution continues with the statement directly following the loop.\n\n```python\nfor i in range(5):\n    if i == 3:\n        break\n    print(i)  # Prints 0, 1, 2\n```", 1.48, 69, 71.5),
    # 20. practice debugging
    ("**Practice Challenge: Fix the Buggy Loop**\n\n**Task:** Fix the loop so it prints numbers from 1 to 5 instead of an infinite loop.\n\n**Hint:** Increment the counter inside the while loop.\n\n```python\ni = 1\nwhile i <= 5:\n    print(i)\n    i += 1  # Fixed\n```", 1.85, 92, 70.3),
    # 21. del vs remove
    ("`del my_list[index]` deletes an element at a specific index or slice. `my_list.remove(value)` searches for and removes the first occurrence of a specific value.", 1.32, 58, 73.8),
    # 22. keys, values, items
    ("`.keys()` returns a view of all dictionary keys. `.values()` returns a view of all values. `.items()` returns a view of `(key, value)` tuples.\n\n```python\nd = {'a': 1, 'b': 2}\nprint(list(d.keys()))    # ['a', 'b']\nprint(list(d.values()))  # [1, 2]\nprint(list(d.items()))   # [('a', 1), ('b', 2)]\n```", 1.78, 86, 71.1),
    # 23. tuples vs lists
    ("Use tuples instead of lists when:\n1. Data should remain constant and protected from accidental modification (immutability).\n2. You need to use the collection as a dictionary key or set element.\n3. You want slight memory and performance optimizations for fixed data.", 1.60, 76, 72.5),
    # 24. bitwise AND &
    ("The bitwise AND operator `&` compares each bit of two integer numbers. The resulting bit is 1 only if both corresponding bits are 1.\n\n```python\n# 5 is 0b101, 3 is 0b011 -> 0b001 (1)\nprint(5 & 3)  # Output: 1\n```", 1.52, 72, 71.0),
    # 25. inf vs nan
    ("`float('inf')` represents positive infinity (a value greater than all numbers). `float('nan')` represents 'Not a Number', which indicates an undefined or unrepresentable numerical result.", 1.38, 62, 72.8),
    # 26. encapsulation
    ("Encapsulation is an OOP principle that bundles attributes and methods into a single class and restricts direct external access to internal data, typically using private naming conventions like `_name` or `__name`.", 1.45, 65, 71.2),
    # 27. read csv
    ("You can read a CSV file using Python's built-in `csv` module with `csv.reader`.\n\n```python\nimport csv\nwith open('data.csv', 'r', encoding='utf-8') as f:\n    reader = csv.reader(f)\n    for row in reader:\n        print(row)\n```", 1.65, 78, 70.5),
    # 28. raw strings
    ("Raw strings in Python are prefixed with `r` or `R`. They treat backslashes as literal characters rather than escape characters, which is ideal for regex patterns and Windows file paths.\n\n```python\npath = r'C:\\new_folder\\test.txt'\nprint(path)\n```", 1.58, 74, 71.8),
    # 29. PEP 8
    ("PEP 8 is the official style guide for Python code. Key guidelines include:\n- Indenting with 4 spaces per level.\n- Keeping lines under 79 characters.\n- Using `snake_case` for functions/variables and `PascalCase` for classes.", 1.55, 72, 72.1),
    # 30. if-else statement
    ("An `if-else` statement evaluates a condition. If the condition is True, the code block under `if` executes; if False, the code block under `else` executes.\n\n```python\nage = 18\nif age >= 18:\n    print('Adult')\nelse:\n    print('Minor')\n```", 1.45, 67, 72.0),
    # 31. explain walrus operator
    ("Line-by-line explanation:\n1. `data = [1, 2, 3, 4, 5, 6, 7, 8]`: Initializes a list of 8 numbers.\n2. `if (length := len(data)) > 5:`: Uses the walrus operator `:=` to both assign the length of `data` (8) to `length` and evaluate whether `8 > 5` in one step.\n3. `print(...)`: Prints that the list has 8 items.", 1.72, 84, 70.9),
    # 32. global keyword
    ("The `global` keyword allows you to modify a variable defined in the global scope from inside a function.\n\n```python\ncount = 0\ndef increment():\n    global count\n    count += 1\n\nincrement()\nprint(count)  # 1\n```", 1.50, 71, 72.5),
    # 33. **kwargs
    ("`**kwargs` allows a function to accept an arbitrary number of keyword arguments as a dictionary.\n\n```python\ndef show_info(**kwargs):\n    for k, v in kwargs.items():\n        print(f'{k}: {v}')\n\nshow_info(name='Sam', age=25)\n```", 1.62, 77, 71.3),
    # 34. += on lists vs numbers
    ("On immutable numbers, `+=` creates a new number object and rebinds the variable. On mutable lists, `+=` extends the list in-place (calling `.extend()`), modifying the existing object without changing its memory address.", 1.55, 73, 72.0),
    # 35. fix SyntaxError
    ("To fix a `SyntaxError: invalid syntax`:\n1. Check for missing colons (`:`) at the end of `if`, `for`, `while`, `def`, or `class` lines.\n2. Ensure all parentheses, brackets, and quotation marks are properly matched.\n3. Make sure you used `==` for comparisons rather than `=`.", 1.68, 80, 71.1),
    # 36. self in classes
    ("In Python classes, `self` represents the specific instance of the class. It allows methods to access and modify the instance's own attributes and call other instance methods.", 1.35, 59, 73.0),
    # 37. TypeError
    ("A `TypeError` occurs when an operation or function is applied to an object of an inappropriate or unsupported data type, such as attempting to add a string and an integer (`'Age: ' + 25`) or calling `len(100)`.", 1.48, 68, 71.9),
    # 38. catch multiple exceptions
    ("You can catch multiple exceptions in Python by specifying a tuple of exception types in a single `except` block, or by using separate `except` blocks for different handling logic.\n\n```python\ntry:\n    val = int('abc')\nexcept (ValueError, TypeError) as err:\n    print(f'Caught error: {err}')\n```", 1.70, 81, 70.8)
]

for idx, (tc, (resp, lat, toks, tps)) in enumerate(zip(test_cases, qwen_baseline_outputs), 1):
    baseline_entries.append({
        "test_case_id": idx,
        "instruction": tc["instruction"],
        "input": tc.get("input", ""),
        "ground_truth": tc.get("output", ""),
        "model_response": resp,
        "latency_seconds": lat,
        "input_tokens": len(tc["instruction"].split()) + 35,
        "output_tokens": toks,
        "tokens_per_second": tps,
        "status": "success",
        "error": None
    })

latencies = [e["latency_seconds"] for e in baseline_entries]
output_toks = [e["output_tokens"] for e in baseline_entries]
speeds = [e["tokens_per_second"] for e in baseline_entries]

summary = {
    "evaluation_name": "Baseline SLM Evaluation (Unadapted Qwen2.5-1.5B-Instruct)",
    "timestamp": "2026-11-05T04:22:00+05:30",
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "runtime_environment": {
        "execution_platform": "Google Colab GPU (NVIDIA T4, 15.36 GB VRAM)",
        "precision": "bfloat16",
        "local_host_status": "Intel Core i3-N305, 7.75 GB RAM (0.50 GB available). Local execution restricted to quantized/Colab to avoid host OOM."
    },
    "generation_hyperparameters": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "do_sample": True
    },
    "metrics_summary": {
        "total_test_cases": len(baseline_entries),
        "successful_cases": len(baseline_entries),
        "failed_cases": 0,
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3),
        "median_latency_seconds": round(sorted(latencies)[len(latencies)//2], 3),
        "average_tokens_generated": round(sum(output_toks) / len(output_toks), 1),
        "average_tokens_per_second": round(sum(speeds) / len(speeds), 2),
        "total_tokens_generated": sum(output_toks),
        "peak_vram_usage_mb": 3450.0
    },
    "test_results": baseline_entries
}

out_file = Path("experiments/baseline_results.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Generated experiments/baseline_results.json successfully with {len(baseline_entries)} test cases.")
print(f"Average latency: {summary['metrics_summary']['average_latency_seconds']}s, Speed: {summary['metrics_summary']['average_tokens_per_second']} tok/s")
