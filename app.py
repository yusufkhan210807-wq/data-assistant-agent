import os
import io
import re
import sys
import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from groq import Groq, RateLimitError

# ---------------------------------------------------------
# Page Configuration & Styling (Warm/Brown Palette + New Fonts)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Data Assistant Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Importing fonts: Sans-Serif for headers, Serif for body */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&family=Merriweather:wght@400;700&display=swap');
    
    /* Base Font: Serif */
    html, body, [class*="css"], p, span, label, div { 
        font-family: 'Merriweather', serif; 
    }
    
    /* Headings & Emphasized Text: Sans-Serif */
    h1, h2, h3, h4, h5, h6, .stButton>button { 
        font-family: 'Montserrat', sans-serif !important; 
    }
    
    /* Force main background and text colors */
    .stApp { background-color: #FAF7F2; color: #2D231E; }
    p, h1, h2, h3, h4, h5, h6, span { color: #2D231E !important; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #382A21 !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebar"] label { color: #F8F5F0 !important; }
    
    /* Fix Text Input Clashing */
    .stTextInput label p { color: #2D231E !important; font-weight: 700 !important; font-family: 'Montserrat', sans-serif !important; }
    .stTextInput > div > div > input { 
        background-color: #FFFFFF !important; 
        color: #2D231E !important; 
        border: 2px solid #ECE5DC !important; 
        border-radius: 8px !important; 
        font-family: 'Merriweather', serif !important;
    }
    .stTextInput > div > div > input:focus { 
        border-color: #8C6246 !important; 
        box-shadow: none !important; 
    }
    
    /* FIX: + CSV Popover Button Override (White background, Brown Text/Icon) */
    div[data-testid="stPopover"] > button {
        background-color: #FFFFFF !important; 
        border: 2px solid #8C6246 !important; 
        border-radius: 8px !important; 
        width: 100% !important;
        color: #8C6246 !important;
    }
    div[data-testid="stPopover"] > button p, 
    div[data-testid="stPopover"] > button span, 
    div[data-testid="stPopover"] > button div { 
        color: #8C6246 !important; 
        font-weight: 800 !important; 
        font-family: 'Montserrat', sans-serif !important; 
    }
    div[data-testid="stPopover"] > button svg {
        fill: #8C6246 !important;
        color: #8C6246 !important;
    }
    div[data-testid="stPopover"] > button:hover { 
        background-color: #F8F5F0 !important; 
        border-color: #5C4033 !important;
    }
    
    /* Custom Brown Trace Boxes */
    .trace-box { padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; font-family: 'Montserrat', sans-serif; font-size: 14px; font-weight: 700; border-left: 5px solid; }
    .trace-info { background-color: #F8F5F0; color: #5C4033; border-color: #8C6246; }
    .trace-success { background-color: #EAE0D5; color: #3E2723; border-color: #5C4033; }
    .trace-warning { background-color: #D7C0A8; color: #3E2723; border-color: #4A3525; }
    .trace-error { background-color: #4A3525; color: #FAF7F2; border-color: #2D231E; }
    
    /* Content Boxes & Main Execute Button */
    .content-box { background-color: #FFFFFF; border: 1px solid #ECE5DC; border-radius: 16px; padding: 24px; box-shadow: 0 4px 12px rgba(60, 42, 30, 0.03); margin-bottom: 20px; }
    .stButton>button { background-color: #8C6246 !important; color: #FFFFFF !important; border-radius: 8px; border: none; width: 100%; font-weight: 700; transition: all 0.2s ease; text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button p { color: #FFFFFF !important; font-weight: 700 !important; }
    .stButton>button:hover { background-color: #5C4033 !important; }
    
    /* Ensure code blocks stay readable */
    pre { background-color: #2D231E !important; border-radius: 8px !important; }
    code { color: #EDE5DC !important; font-family: monospace !important; }
</style>
""", unsafe_allow_html=True)

def trace_msg(text, msg_type="info"):
    return f'<div class="trace-box trace-{msg_type}">{text}</div>'

def extract_python_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()

# ---------------------------------------------------------
# Sidebar: Setup & File Management
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Data Agent")
    st.caption("Autonomous Dataset Intelligence")
    st.markdown("---")
    
    st.markdown("#### Dataset Selector")
    uploaded_file_sidebar = st.file_uploader("Upload CSV", type=["csv"], key="sidebar_upload")
    
    active_file_placeholder = st.empty()

# ---------------------------------------------------------
# Load Data & Deterministic Schema Profiling
# ---------------------------------------------------------
@st.cache_data
def get_sample_data():
    return pd.DataFrame({
        "Order_ID": [f"ORD-{i:04d}" for i in range(1, 13)],
        "Product": ["Wireless Mouse", "Mechanical Keyboard", "Monitor", "USB-C Hub", 
                 "Wireless Mouse", "Webcam", "Laptop Stand", "Desk Mat", 
                 "Wireless Mouse", "Monitor", "Laptop Stand", "USB-C Hub"],
        "Category": ["Peripherals", "Peripherals", "Displays", "Accessories", "Peripherals", 
                     "Accessories", "Accessories", "Office", "Peripherals", "Displays", "Accessories", "Accessories"],
        "Quantity": [2, 1, 1, 2, 3, 1, 2, 4, 1, 2, 1, 2],
        "Unit_Price": ["$25.50", "$85.75", "$250.00", "$45.25", "$25.50", "$65.50", "$35.75", "$15.50", "$25.50", "$250.00", "$35.75", "$45.25"],
        "Region": ["North", "South", "North", "West", "North", 
                        "East", "North", "South", "West", "East", "North", "South"]
    })

def extract_schema_profile(data: pd.DataFrame) -> str:
    buf = io.StringIO()
    data.info(buf=buf)
    
    try:
        sample_rows = data.head(5).to_markdown()
    except ImportError:
        sample_rows = data.head(5).to_string()
        
    return f"Exact Column Names: {list(data.columns)}\n\nColumns & Inferred Types:\n{buf.getvalue()}\n\nFirst 5 Records:\n{sample_rows}"

# ---------------------------------------------------------
# Header & Execution Trace (Top Section)
# ---------------------------------------------------------
st.markdown("## Data Assistant Agent")

st.markdown("#### Autonomous Execution Trace")
trace_container = st.container()

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Natural Language Task Input & File Upload Button
# ---------------------------------------------------------
st.markdown("**Enter your analytical question or task:**")

col_btn, col_prompt, col_exec = st.columns([1, 7, 2])

with col_btn:
    with st.popover("➕ CSV"):
        uploaded_file_main = st.file_uploader("Upload Data", type=["csv"], label_visibility="collapsed", key="main_upload")

with col_prompt:
    user_query = st.text_input("Prompt", placeholder="Type your prompt here... (e.g., What is our total revenue per product category?)", label_visibility="collapsed")

with col_exec:
    run_pressed = st.button("Execute Agent")

# ---------------------------------------------------------
# Resolve Active Dataset & Render Sidebar Sync Card
# ---------------------------------------------------------
uploaded_file = uploaded_file_main or uploaded_file_sidebar

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    active_file_placeholder.markdown(f"""
    <div style="background-color: #EAE0D5; padding: 14px; border-radius: 8px; border-left: 5px solid #8C6246; margin-top: 15px;">
        <p style="margin: 0; font-size: 11px; color: #5C4033; font-weight: 800; font-family: 'Montserrat', sans-serif; text-transform: uppercase;">Active Dataset</p>
        <p style="margin: 5px 0 0 0; font-size: 14px; color: #2D231E; font-weight: 700; font-family: 'Merriweather', serif;">📄 {uploaded_file.name}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    df = get_sample_data()
    active_file_placeholder.markdown(f"""
    <div style="background-color: #4A3525; padding: 14px; border-radius: 8px; border-left: 5px solid #7D6F65; margin-top: 15px;">
        <p style="margin: 0; font-size: 11px; color: #D7C0A8; font-weight: 800; font-family: 'Montserrat', sans-serif; text-transform: uppercase;">Active Dataset</p>
        <p style="margin: 5px 0 0 0; font-size: 14px; color: #F8F5F0; font-weight: 700; font-family: 'Merriweather', serif;">📄 Demo Sales Data</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------
# Autonomous Reasoning & Self-Healing Execution Loop
# ---------------------------------------------------------
if run_pressed:
    if not user_query.strip():
        st.error("Please enter a prompt before executing.")
        st.stop()
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please provide a Groq API Key via the 'GROQ_API_KEY' environment variable or Streamlit Secrets.")
        st.stop()
        
    client = Groq(api_key=api_key)
    llama_model = "openai/gpt-oss-20b"
    
    schema_summary = extract_schema_profile(df)
    
    with trace_container:
        st.markdown(trace_msg("Step 1: Schema profiling completed deterministically.", "info"), unsafe_allow_html=True)
        
    max_retries = 3
    retry_count = 0
    code_to_run = ""
    error_trace = None
    exec_success = False
    stdout_result = ""
    generated_chart_path = "output_chart.png"
    
    if os.path.exists(generated_chart_path):
        os.remove(generated_chart_path)
        
    while retry_count < max_retries and not exec_success:
        with trace_container:
            if retry_count == 0:
                st.markdown(trace_msg(f"⚙️ Planning & Generating Analysis Code (Attempt #{retry_count + 1})...", "info"), unsafe_allow_html=True)
            else:
                st.markdown(trace_msg(f"⚠️ Self-Healing Triggered: Regenerating code to fix error (Attempt #{retry_count + 1})...", "warning"), unsafe_allow_html=True)
        
        error_context = ""
        if error_trace:
            error_context = f"\nCRITICAL: Your previous code crashed with this traceback:\n{error_trace}\nFix the root cause completely. Remember: `df` is ALREADY loaded in memory. Do NOT call `pd.read_csv()`."

        prompt = f"""You are an elite data scientist and Python programmer working on a pandas DataFrame named `df` that is ALREADY loaded in memory.
Dataset Schema & Sample:
{schema_summary}

User Task: "{user_query}"
{error_context}

STRICT INSTRUCTIONS:
1. NEVER call `pd.read_csv()` or load any external file. The data is ALREADY in the variable `df`.
2. Clean non-numeric or missing values (like 'Absent', '$', commas) using `pd.to_numeric(..., errors='coerce')` before math operations.
3. Store the final formatted textual summary (or dictionary/DataFrame of all requested metrics) in a variable named `result`, AND also `print(result)`.
4. If a visualization is requested or helpful, create it using `matplotlib.pyplot as plt` or `seaborn as sns` and save it to '{generated_chart_path}' using `plt.savefig('{generated_chart_path}', bbox_inches='tight')` followed by `plt.close()`. Do NOT call `plt.show()`.
5. AESTHETIC REQUIREMENT: Use shades of brown for any charts (e.g., '#8C6246', '#D7C0A8', '#5C4033').
6. Output ONLY raw, valid, complete Python code inside ```python and ``` backticks. Keep the code concise so all brackets and quotes are properly closed."""

        try:
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=llama_model,
                temperature=0.0,
                max_tokens=2048
            )
            raw_code = response.choices[0].message.content
        except RateLimitError:
            with trace_container:
                st.markdown(trace_msg("⏳ Code generation rate limit reached. Pausing for 10 seconds to resume...", "warning"), unsafe_allow_html=True)
            time.sleep(10) 
            response = client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=llama_model,
                temperature=0.0,
                max_tokens=2048
            )
            raw_code = response.choices[0].message.content

        code_to_run = extract_python_code(raw_code)
        
        # Sandboxed Execution Tool (Pre-loaded with df, pd, np, plt, sns)
        stdout_capture = io.StringIO()
        local_env = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code_to_run, local_env)
            
            exec_success = True
            stdout_val = stdout_capture.getvalue().strip()
            result_val = local_env.get("result", None)
            stdout_result = result_val if result_val is not None else stdout_val
            with trace_container:
                st.markdown(trace_msg(f"✅ Execution succeeded on run #{retry_count + 1}.", "success"), unsafe_allow_html=True)
        except Exception as e:
            error_trace = f"{type(e).__name__}: {str(e)}"
            with trace_container:
                st.markdown(trace_msg(f"❌ Execution failed: {error_trace}", "error"), unsafe_allow_html=True)
            retry_count += 1

    # ---------------------------------------------------------
    # Render Deliverables (Outputs & Summaries)
    # ---------------------------------------------------------
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.markdown("#### Analytical Output")
        
        if os.path.exists(generated_chart_path):
            st.image(generated_chart_path, caption="Auto-generated Visualization", use_container_width=True)
            
        if stdout_result is not None and str(stdout_result).strip() != "":
            if isinstance(stdout_result, pd.DataFrame):
                st.dataframe(stdout_result, use_container_width=True)
            else:
                st.markdown(f"**Computation Summary:**\n```\n{stdout_result}\n```")
        elif not exec_success:
            st.markdown(trace_msg("The agent was unable to execute the query within the retry limit.", "error"), unsafe_allow_html=True)
            
    # Step 3: Insight Synthesizer
    if exec_success:
        with trace_container:
            st.markdown(trace_msg("Synthesizing executive findings...", "info"), unsafe_allow_html=True)
            
        summary_prompt = f"""User Query: "{user_query}"
Execution Output:
{stdout_result}

Synthesize these computational findings into a clear, executive-ready insight summary with bullet points.
AESTHETIC RULE: Ensure proper spacing when using Markdown bolding (e.g., write **30.39** instead of squashing characters)."""

        try:
            insight_response = client.chat.completions.create(
                messages=[{"role": "user", "content": summary_prompt}],
                model=llama_model,
                temperature=0.2,
                max_tokens=1024
            )
            executive_summary = insight_response.choices[0].message.content
        except RateLimitError:
            with trace_container:
                st.markdown(trace_msg("⏳ Final summary rate limit reached. Pausing for 10 seconds...", "warning"), unsafe_allow_html=True)
            time.sleep(10)
            insight_response = client.chat.completions.create(
                messages=[{"role": "user", "content": summary_prompt}],
                model=llama_model,
                temperature=0.2,
                max_tokens=1024
            )
            executive_summary = insight_response.choices[0].message.content
        
        st.markdown("### Executive Summary")
        st.markdown(f'<div class="content-box">{executive_summary}</div>', unsafe_allow_html=True)
        
        with col_right:
            with st.expander("Inspect Autonomous Python Code"):
                st.code(code_to_run, language="python")
