"""Runtime Verification Script for Streamlit Application and FastAPI Integration."""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import urllib.request
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_PORT = 8009
UI_PORT = 8509
API_URL = f"http://127.0.0.1:{API_PORT}"
UI_URL = f"http://127.0.0.1:{UI_PORT}"
PYTHON_EXE = sys.executable


def probe_url(url: str, timeout: float = 2.0) -> int:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Probe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode()
    except Exception:
        return 0


def main():
    print(f"=== 1. Starting actual FastAPI server on {API_URL} ===")
    api_cmd = [
        PYTHON_EXE, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "127.0.0.1",
        "--port", str(API_PORT),
        "--log-level", "warning"
    ]
    api_proc = subprocess.Popen(api_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for API server ready
    api_ready = False
    for _ in range(20):
        time.sleep(0.5)
        if probe_url(f"{API_URL}/health") == 200:
            api_ready = True
            break
    if not api_ready:
        print("ERROR: FastAPI server failed to start.")
        api_proc.terminate()
        return 1
    print("FastAPI server is ONLINE and verified.")

    print(f"\n=== 2. Starting actual Streamlit server on {UI_URL} ===")
    env = os.environ.copy()
    env["API_URL"] = API_URL
    ui_cmd = [
        PYTHON_EXE, "-m", "streamlit", "run",
        "src/ui/app.py",
        f"--server.port={UI_PORT}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    ui_proc = subprocess.Popen(ui_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for Streamlit server ready
    ui_ready = False
    for _ in range(30):
        time.sleep(0.5)
        # Check Streamlit core health
        if probe_url(f"{UI_URL}/_stcore/health") == 200:
            ui_ready = True
            break
    if not ui_ready:
        print("ERROR: Streamlit server failed to start or open port.")
        ui_proc.terminate()
        api_proc.terminate()
        return 1
    print("Streamlit server is ONLINE and verified on HTTP port.")

    # Probe main page HTML
    main_code = probe_url(UI_URL)
    print(f"Streamlit root endpoint GET {UI_URL} status: {main_code}")

    results = {
        "api_url": API_URL,
        "ui_url": UI_URL,
        "streamlit_health_endpoint": f"{UI_URL}/_stcore/health",
        "streamlit_health_status": 200,
        "streamlit_main_status": main_code,
        "checks": []
    }

    try:
        print("\n=== 3. Running AppTest UI Component & Flow Verification ===")
        os.environ["API_URL"] = API_URL
        script_path = str(PROJECT_ROOT / "src" / "ui" / "app.py")
        at = AppTest.from_file(script_path, default_timeout=30)
        at.run()

        # Check for exceptions
        has_exceptions = len(at.exception) > 0
        if has_exceptions:
            print(f"ERROR: Streamlit threw exceptions during execution: {at.exception}")
            results["checks"].append({"name": "no_exceptions", "passed": False, "details": str(at.exception)})
        else:
            print("SUCCESS: Zero runtime exceptions in Streamlit script execution.")
            results["checks"].append({"name": "no_exceptions", "passed": True})

        # Verify Sidebar text
        sidebar_text = " ".join([m.value for m in at.sidebar.markdown])
        sidebar_success = " ".join([s.value for s in at.sidebar.success])
        online_verified = "Backend: Online" in sidebar_success or "Backend: Online" in sidebar_text
        model_verified = "Qwen/Qwen2.5-1.5B-Instruct" in sidebar_text
        standby_verified = "Unloaded (Standby) 💤" in sidebar_text
        print(f"Sidebar Status Verified Online: {online_verified} (Success alerts: {[s.value for s in at.sidebar.success]})")
        print(f"Sidebar Model ID Verified: {model_verified}")
        print(f"Sidebar Standby Indicator Verified: {standby_verified}")
        results["checks"].append({
            "name": "sidebar_backend_communication",
            "passed": online_verified and model_verified and standby_verified,
            "details": {
                "online": online_verified,
                "model_id": model_verified,
                "standby": standby_verified
            }
        })

        # Check tabs count and labels
        print(f"Tabs present: {len(at.tabs)}")
        results["checks"].append({"name": "tab_count", "passed": len(at.tabs) == 4, "count": len(at.tabs)})

        # Test Tab 1: Ask Question
        print("\n--- Testing Tab 1: Ask Python Question ---")
        at.text_area[0].input("What is a tuple in Python?").run()
        # Click Get Answer button
        at.button[0].click().run()
        
        # Check execution badge rendered
        tab1_markdown = [m.value for m in at.markdown]
        fallback_badge_found = any("Development Fallback Mode" in m for m in tab1_markdown)
        print(f"Execution Badge 'Development Fallback Mode' Rendered: {fallback_badge_found}")
        results["checks"].append({
            "name": "tab1_ask_execution_badge",
            "passed": fallback_badge_found
        })

        # Check for runtime errors
        results["checks"].append({
            "name": "post_interaction_no_exceptions",
            "passed": len(at.exception) == 0
        })
    finally:
        print("\nShutting down Streamlit and FastAPI servers...")
        ui_proc.terminate()
        api_proc.terminate()
        try:
            ui_proc.wait(timeout=5)
        except Exception:
            ui_proc.kill()
        try:
            api_proc.wait(timeout=5)
        except Exception:
            api_proc.kill()
        print("Servers shut down cleanly.")

    # Save results
    out_path = PROJECT_ROOT / "reports" / "ui_verification_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved actual UI verification results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
