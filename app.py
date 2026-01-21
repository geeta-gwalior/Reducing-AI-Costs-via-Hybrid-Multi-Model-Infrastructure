import streamlit as st
import pandas as pd
import time
from agents.agent import KifayatiRouter

# 1. Page Configuration
st.set_page_config(page_title="Kifayati AI Agent", page_icon="💰", layout="wide")

st.title(" Kifayati AI: Cost-Optimized Agent")
st.markdown("### Hybrid Architecture: Gemma 3 (Local/Edge) + Gemini 2.5 (Cloud)")

# 2. Initialise Session States for Caching & Dashboard
if "router" not in st.session_state:
    st.session_state.router = KifayatiRouter()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics_data" not in st.session_state:
    st.session_state.metrics_data = []

# 3. Sidebar for Metrics & Controls
with st.sidebar:
    st.header("FinOps Dashboard")
    st.info(" **Gemma 3:4b** handles simple queries (Free/Cheap).")
    st.warning(" **Gemini 2.5** handles complex logic (Premium).")
    
    # Simple Metrics in Sidebar
    if st.session_state.metrics_data:
        df_sidebar = pd.DataFrame(st.session_state.metrics_data)
        total_saved = (len(df_sidebar) * 0.0001) - df_sidebar['Cost'].sum()
        st.metric("Total Money Saved", f"${total_saved:.5f}")

    if st.button("Clear History & Metrics"):
        st.session_state.messages = []
        st.session_state.metrics_data = []
        st.session_state.router = KifayatiRouter()
        st.rerun()

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input & AI Response Logic
if prompt := st.chat_input("Ask me anything..."):
    # User Message Display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # Timing and Execution
            start_t = time.time()
            response = st.session_state.router.route_and_execute(prompt)
            end_t = time.time()
            latency = round(end_t - start_t, 2)

            # Routing Identity Check
            query_key = prompt.lower().strip()
            simple_words = ['hi', 'hello', 'hey', 'help']
            is_gemma = any(word in query_key for word in simple_words) or len(prompt.split()) < 5
            
            model_used = "Gemma 3:4b" if is_gemma else "Gemini 2.5 Flash"
            cost_used = 0.00001 if is_gemma else 0.0001

            # Update Session Metrics
            st.session_state.metrics_data.append({
                "Time": time.strftime("%H:%M:%S"),
                "Model": model_used,
                "Latency": latency,
                "Cost": cost_used
            })

            # Formatting Response
            if is_gemma:
                final_display = f"🟢 **Gemma 3:4b (Kifayati Mode):**\n\n{response}"
                st.toast("Saved cost by using Gemma!", icon="💰")
            else:
                final_display = f"🔵 **Gemini 2.5 Flash (Expert Mode):**\n\n{response}"
            
            message_placeholder.markdown(final_display)
            st.session_state.messages.append({"role": "assistant", "content": final_display})
            
        except Exception as e:
            st.error(f"Error: {e}")

# 6. LIVE DASHBOARD (Below Chat)
if st.session_state.metrics_data:
    st.divider()
    st.subheader("📊 Real-time FinOps Performance")
    df = pd.DataFrame(st.session_state.metrics_data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Latency Comparison (Seconds)**")
        st.bar_chart(df, x="Time", y="Latency", color="Model")
    with col2:
        st.write("**Cost per Request ($)**")
        st.area_chart(df, x="Time", y="Cost", color="Model")
