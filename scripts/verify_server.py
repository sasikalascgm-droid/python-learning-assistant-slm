"""Runtime Verification Script for FastAPI Application."""

import json
import time
import subprocess
import sys
from pathlib import Path
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PORT = 8008
BASE_URL = f"http://127.0.0.1:{PORT}"
PYTHON_EXE = sys.executable


def make_request(method: str, path: str, payload: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    data_bytes = None
    headers = {"User-Agent": "FastAPIVerifier/1.0"}
    
    if payload is not None:
        data_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            status_code = resp.getcode()
            body_bytes = resp.read()
            body = body_bytes.decode("utf-8")
            dt = round(time.time() - t0, 4)
            try:
                parsed_json = json.loads(body)
            except Exception:
                parsed_json = body
            return {
                "method": method,
                "path": path,
                "status_code": status_code,
                "latency_seconds": dt,
                "response": parsed_json,
                "error": None
            }
    except urllib.error.HTTPError as e:
        dt = round(time.time() - t0, 4)
        err_body = e.read().decode("utf-8")
        try:
            parsed_err = json.loads(err_body)
        except Exception:
            parsed_err = err_body
        return {
            "method": method,
            "path": path,
            "status_code": e.code,
            "latency_seconds": dt,
            "response": parsed_err,
            "error": f"HTTPError: {e.code}"
        }
    except Exception as e:
        dt = round(time.time() - t0, 4)
        return {
            "method": method,
            "path": path,
            "status_code": 0,
            "latency_seconds": dt,
            "response": None,
            "error": str(e)
        }


def main():
    print(f"Starting actual FastAPI server on {BASE_URL} via {PYTHON_EXE}...")
    cmd = [
        PYTHON_EXE, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--log-level", "warning"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for server ready
    ready = False
    max_wait = 15
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        time.sleep(0.5)
        # Check if process terminated prematurely
        poll = proc.poll()
        if poll is not None:
            stdout, stderr = proc.communicate()
            print(f"Server exited prematurely with code {poll}!")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return 1
        
        try:
            probe = make_request("GET", "/health")
            if probe["status_code"] == 200:
                ready = True
                print(f"Server is ready! (took {round(time.time() - start_time, 2)}s)")
                break
        except Exception:
            pass

    if not ready:
        print("Server failed to respond within timeout window.")
        proc.terminate()
        stdout, stderr = proc.communicate()
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        return 1

    results = []

    print("\n--- 1. Testing GET / ---")
    r1 = make_request("GET", "/")
    print(f"Status: {r1['status_code']} | Latency: {r1['latency_seconds']}s")
    print(f"Response: {r1['response']}")
    results.append(r1)

    print("\n--- 2. Testing GET /health ---")
    r2 = make_request("GET", "/health")
    print(f"Status: {r2['status_code']} | Latency: {r2['latency_seconds']}s")
    print(f"Response: {r2['response']}")
    results.append(r2)

    print("\n--- 3. Testing POST /ask (Valid Question) ---")
    r3 = make_request("POST", "/ask", {"question": "How do lists work in Python and how do I append items?"})
    print(f"Status: {r3['status_code']} | Latency: {r3['latency_seconds']}s")
    print(f"Response: {r3['response']}")
    results.append(r3)

    print("\n--- 4. Testing POST /ask (Empty Question -> 422) ---")
    r4 = make_request("POST", "/ask", {"question": "   "})
    print(f"Status: {r4['status_code']} | Expected: 422")
    print(f"Response: {r4['response']}")
    results.append(r4)

    print("\n--- 5. Testing POST /ask (Too Short Question -> 422) ---")
    r5 = make_request("POST", "/ask", {"question": "a"})
    print(f"Status: {r5['status_code']} | Expected: 422")
    print(f"Response: {r5['response']}")
    results.append(r5)

    print("\n--- 6. Testing POST /explain-code (Valid Code) ---")
    code_sample = "def add(x, y):\n    return x + y\n\nprint(add(2, 3))"
    r6 = make_request("POST", "/explain-code", {"code": code_sample})
    print(f"Status: {r6['status_code']} | Latency: {r6['latency_seconds']}s")
    print(f"Response: {r6['response']}")
    results.append(r6)

    print("\n--- 7. Testing POST /explain-code (Syntax Error Handled Safely) ---")
    bad_code = "def broken(\n    print('Missing closing parenthesis')"
    r7 = make_request("POST", "/explain-code", {"code": bad_code})
    print(f"Status: {r7['status_code']} | Syntax Valid: {r7['response'].get('syntax_valid') if isinstance(r7['response'], dict) else None}")
    print(f"Syntax Error: {r7['response'].get('syntax_error') if isinstance(r7['response'], dict) else None}")
    results.append(r7)

    print("\n--- 8. Testing POST /explain-code (Empty Code -> 422) ---")
    r8 = make_request("POST", "/explain-code", {"code": "   \n  "})
    print(f"Status: {r8['status_code']} | Expected: 422")
    results.append(r8)

    print("\n--- 9. Testing POST /practice (Valid Topic & Difficulty) ---")
    r9 = make_request("POST", "/practice", {"topic": "lists", "difficulty": "beginner"})
    print(f"Status: {r9['status_code']} | Latency: {r9['latency_seconds']}s")
    print(f"Title: {r9['response'].get('title') if isinstance(r9['response'], dict) else None}")
    print(f"Hints count: {len(r9['response'].get('hints', [])) if isinstance(r9['response'], dict) else 0}")
    results.append(r9)

    print("\n--- 10. Testing POST /practice (Invalid Difficulty -> 422) ---")
    r10 = make_request("POST", "/practice", {"topic": "lists", "difficulty": "expert_impossible"})
    print(f"Status: {r10['status_code']} | Expected: 422")
    results.append(r10)

    print("\n--- 11. Testing POST /practice (Empty Topic -> 422) ---")
    r11 = make_request("POST", "/practice", {"topic": "   ", "difficulty": "beginner"})
    print(f"Status: {r11['status_code']} | Expected: 422")
    results.append(r11)

    print("\nShutting down FastAPI server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("Server stopped cleanly.")

    # Save results
    out_path = PROJECT_ROOT / "reports" / "api_verification_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved actual verification results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
