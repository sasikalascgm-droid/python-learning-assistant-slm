"""Streamlit Web Interface for Python Learning Assistant SLM."""

import os
import requests
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Python Learning Assistant (SLM)",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration & Backend URL
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-fallback {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #FCD34D;
    }
    .badge-slm {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #6EE7B7;
    }
    .badge-base {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #93C5FD;
    }
</style>
""", unsafe_allow_html=True)


def render_execution_badge(mode: str, model_status: str):
    """Renders a prominent visual indicator explaining the response origin."""
    if mode == "slm_inference":
        st.markdown(
            '<span class="badge-slm">🤖 Generated via Live Local SLM (Qwen2.5-1.5B)</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="badge-fallback">💡 Development Fallback Mode (SLM Unloaded / Standby)</span>',
            unsafe_allow_html=True
        )


# Sidebar Configuration & Hardware Status
st.sidebar.title("⚙️ System & Controls")
api_url = st.sidebar.text_input("FastAPI Backend URL", value=DEFAULT_API_URL)

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ Hardware & Model Status")

# Query health endpoint
backend_online = False
health_data = None
try:
    resp = requests.get(f"{api_url}/health", timeout=2.0)
    if resp.status_code == 200:
        backend_online = True
        health_data = resp.json()
except Exception:
    pass

if backend_online and health_data:
    st.sidebar.success("Backend: Online (HTTP 200)")
    st.sidebar.markdown(f"**Model ID:** `{health_data['model_id']}`")
    st.sidebar.markdown(f"**Revision:** `{health_data['revision'][:10]}...`")
    st.sidebar.markdown(f"**Model In Memory:** `{'Loaded ✅' if health_data['model_loaded'] else 'Unloaded (Standby) 💤'}`")
    st.sidebar.markdown(f"**Adapter Available:** `{'Yes ✅' if health_data['adapter_exists'] else 'Pending Colab Run ⏳'}`")
    st.sidebar.markdown(f"**Host RAM Free:** `{health_data['host_ram_avail_gb']} GB / {health_data['host_ram_total_gb']} GB`")
    st.sidebar.markdown(f"**CUDA GPU:** `{'Yes' if health_data['cuda_available'] else 'None (CPU Mode)'}`")
    st.sidebar.markdown(f"**Active Mode:** `{health_data['active_configuration']}`")
else:
    st.sidebar.error("Backend: Offline / Unreachable")
    st.sidebar.caption(f"Cannot reach `{api_url}`. Start backend via `uvicorn src.api.main:app`.")

st.sidebar.markdown("---")
st.sidebar.caption("Educational AI Assistant powered by Qwen2.5-1.5B Small Language Model (Apache-2.0).")


# Main Page Header
st.markdown('<div class="main-header">🐍 Python Learning Assistant Using SLM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Locally runnable AI designed to help beginner students master Python programming concepts with patience, clarity, and safety.</div>',
    unsafe_allow_html=True
)

# Tab Navigation
tab_ask, tab_explain, tab_practice, tab_about = st.tabs([
    "💬 Ask Python Question",
    "🔍 Explain Python Code",
    "🎯 Practice Exercises",
    "ℹ️ Architecture & About"
])


# ------------------------------------------------------------------------------
# TAB 1: Ask Python Question
# ------------------------------------------------------------------------------
with tab_ask:
    st.subheader("Ask a Python Concept or Question")
    st.caption("Ask questions about variables, loops, lists, dictionaries, functions, OOP, or exceptions in plain English.")

    example_questions = [
        "What is the difference between a list and a tuple in Python?",
        "How do dictionary lookups work and why are they fast?",
        "What is a list comprehension and when should I use one?",
        "Explain how try-except blocks handle ZeroDivisionError with an example."
    ]
    selected_ex = st.selectbox("Or choose a starter question:", ["-- Select an example --"] + example_questions)

    default_q = selected_ex if selected_ex != "-- Select an example --" else ""
    user_question = st.text_area("Your Python Question:", value=default_q, height=100, placeholder="e.g. How does a for loop work with range()?")

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        ask_btn = st.button("Get Answer", type="primary", use_container_width=True)

    if ask_btn:
        if not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Analyzing question and generating explanation..."):
                try:
                    payload = {"question": user_question.strip()}
                    resp = requests.post(f"{api_url}/ask", json=payload, timeout=60.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("---")
                        render_execution_badge(data["mode"], data["model_status"])
                        st.caption(f"Latency: {data['latency_seconds']} seconds")

                        st.markdown(data["answer"])
                        if data.get("note"):
                            st.info(f"ℹ️ {data['note']}")
                    else:
                        st.error(f"Backend returned error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with FastAPI backend: {e}")


# ------------------------------------------------------------------------------
# TAB 2: Explain Python Code
# ------------------------------------------------------------------------------
with tab_explain:
    st.subheader("Line-by-Line Python Code Explainer")
    st.caption("Paste any Python snippet below. The assistant verifies syntax with static AST parsing and deconstructs every line.")

    sample_code = """def calculate_average(grades: list[float]) -> float:
    if not grades:
        return 0.0
    total = sum(grades)
    return total / len(grades)

scores = [85.0, 92.5, 78.0]
avg = calculate_average(scores)
print(f"Average: {avg:.2f}")"""

    code_input = st.text_area("Python Code Snippet:", value=sample_code, height=180)

    if st.button("Explain Code", type="primary"):
        if not code_input.strip():
            st.warning("Please paste some Python code to analyze.")
        else:
            with st.spinner("Parsing code syntax and deconstructing lines..."):
                try:
                    payload = {"code": code_input.strip()}
                    resp = requests.post(f"{api_url}/explain-code", json=payload, timeout=60.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown("---")
                        render_execution_badge(data["mode"], data["model_status"])
                        st.caption(f"Latency: {data['latency_seconds']} seconds")

                        # Syntax Alert
                        if data["syntax_valid"]:
                            st.success("✅ Syntax Status: Valid Python 3.10 Code")
                        else:
                            st.error(f"❌ Syntax Status: Invalid Syntax\n\n{data['syntax_error']}")

                        # Walkthrough
                        st.markdown(data["explanation"])

                        # Structured Line Table
                        if data.get("line_by_line"):
                            with st.expander("📋 View Line-by-Line Breakdown Table", expanded=True):
                                for item in data["line_by_line"]:
                                    st.markdown(f"**Line {item['line_number']}** ` {item['code'].strip()} `")
                                    st.write(f"↳ *{item['explanation']}*")
                    else:
                        st.error(f"Backend returned error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")


# ------------------------------------------------------------------------------
# TAB 3: Practice Exercises
# ------------------------------------------------------------------------------
with tab_practice:
    st.subheader("Targeted Practice Exercises with Progressive Hints")
    st.caption("Generate hands-on Python problems with graduated hints and solution code.")

    col_topic, col_diff = st.columns([3, 1])
    with col_topic:
        topic_input = st.selectbox("Select Topic:", ["Lists and Slicing", "Dictionaries and Keys", "For Loops and Iteration", "Functions and Return Values", "Try-Except Exception Handling"])
    with col_diff:
        diff_input = st.selectbox("Difficulty:", ["beginner", "intermediate", "advanced"])

    if st.button("Generate Exercise", type="primary"):
        with st.spinner(f"Generating {diff_input} practice problem for {topic_input}..."):
            try:
                payload = {"topic": topic_input, "difficulty": diff_input}
                resp = requests.post(f"{api_url}/practice", json=payload, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown("---")
                    render_execution_badge(data["mode"], data["model_status"])
                    st.caption(f"Latency: {data['latency_seconds']} seconds")

                    st.markdown(f"### 🎯 {data['title']}")
                    st.markdown(data["problem_statement"])

                    if data.get("starter_code"):
                        st.markdown("**Starter Code:**")
                        st.code(data["starter_code"], language="python")

                    if data.get("hints"):
                        st.markdown("#### 💡 Progressive Hints")
                        for idx, hint in enumerate(data["hints"], 1):
                            with st.expander(f"Reveal Hint #{idx}"):
                                st.write(hint)

                    if data.get("solution"):
                        with st.expander("🔑 Reveal Model Solution (Try on your own first!)"):
                            st.code(data["solution"], language="python")
                else:
                    st.error(f"Backend returned error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")


# ------------------------------------------------------------------------------
# TAB 4: Architecture & About
# ------------------------------------------------------------------------------
with tab_about:
    st.subheader("About the Python Learning Assistant")
    st.markdown("""
    ### 🏗️ Technical Architecture
    - **Base SLM:** `Qwen/Qwen2.5-1.5B-Instruct` (Pinned Revision: `c644d673f4439c36291a1827471fb3be52382e26`, Apache-2.0).
    - **Base Parameters:** **1,543,714,304** parameters (derived with tied word embeddings and QKV biases).
    - **LoRA Adapter:** Rank $r=8$, $\alpha=16$ on all 7 projections (`q, k, v, o, gate, up, down_proj`) = **9,232,384** trainable parameters (**0.5981%**).
    - **4-bit Quantization:** NormalFloat4 (`nf4`) with double quantization via bitsandbytes.
    - **Backend:** FastAPI (Python 3.10) with asynchronous routing and strict Pydantic validation.
    - **Security:** Static AST parsing (`ast.parse`) for syntax checks. Arbitrary code execution on bare host is prohibited.
    
    ### 🛡️ Safety & Hardware Safeguards
    - Operates safely on modest hardware (Intel Core i3-N305, 7.75 GB RAM) by starting in standby mode and preventing unquantized host OOM freezes.
    - Completely free and open-source: Zero reliance on paid commercial APIs (no OpenAI, no Gemini, no Anthropic).
    """)
