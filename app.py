"""
Kifayati AI — Cost-Optimized Hybrid Agent
Streamlit frontend with real-time FinOps dashboard.
"""
import time
import streamlit as st
import pandas as pd

from agents.agent   import KifayatiRouter
from agents.tracker import metrics_tracker

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title = "Kifayati AI",
    page_icon  = "💰",
    layout     = "wide",
)

# ── Session state init ───────────────────────────────────────────
if "router" not in st.session_state:
    st.session_state.router   = KifayatiRouter()
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ───────────────────────────────────────────────────────
st.title("💰 Kifayati AI")
st.caption("Hybrid Multi-Model Router · Gemma 3:4b (Edge) + Gemini 2.5 Flash (Cloud)")

# ── Sidebar — FinOps Dashboard ───────────────────────────────────
with st.sidebar:
    st.header("📊 FinOps Dashboard")

    summary = metrics_tracker.summary()

    if summary["total_requests"] == 0:
        st.info("Send a message to start tracking costs.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total Requests", summary["total_requests"])
        col2.metric("Cache Hits",     summary["cache_hits"])

        col3, col4 = st.columns(2)
        col3.metric("Gemma Calls",  summary["gemma_calls"],  help="Cheap route")
        col4.metric("Gemini Calls", summary["gemini_calls"], help="Premium route")

        st.divider()
        st.metric("💸 Money Saved",
                  f"${summary['total_saved_usd']:.5f}",
                  delta=f"{summary['cost_reduction_pct']}% vs Gemini-only")
        st.metric("⚡ Avg Latency", f"{summary['avg_latency_s']}s")
        st.metric("💰 Total Spent", f"${summary['total_cost_usd']:.5f}")

    st.divider()

    # Routing decision log
    logs = metrics_tracker.all_logs()
    if logs:
        st.subheader("Routing Log")
        df_log = pd.DataFrame(logs)[
            ["timestamp", "model", "latency_s", "cost_usd", "routing_reason"]
        ].rename(columns={
            "timestamp":      "Time",
            "model":          "Model",
            "latency_s":      "Latency",
            "cost_usd":       "Cost ($)",
            "routing_reason": "Reason",
        })
        st.dataframe(df_log, use_container_width=True, height=220)

    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.session_state.router   = KifayatiRouter()
        st.rerun()

# ── Chat history ─────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            m = msg["meta"]
            badge = "🟢 Gemma 3:4b" if "Gemma" in m["model"] else (
                    "⚡ Cache"       if m["model"] == "Cache"    else
                    "🔵 Gemini 2.5 Flash")
            st.caption(
                f"{badge} · {m['latency_s']}s · "
                f"${m['cost_usd']:.5f} · reason: {m['routing_reason']}"
            )

# ── Chat input ───────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call router
    with st.chat_message("assistant"):
        with st.spinner("Routing…"):
            try:
                result = st.session_state.router.route_and_execute(prompt)
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                st.stop()

        # Display
        model   = result["model"]
        if model == "Cache":
            prefix = "⚡ **[Cache Hit]**"
            st.toast("Free response from cache!", icon="⚡")
        elif "Gemma" in model:
            prefix = "🟢 **Gemma 3:4b (Kifayati Mode)**"
            st.toast("Saved cost by using Gemma!", icon="💰")
        else:
            prefix = "🔵 **Gemini 2.5 Flash (Expert Mode)**"

        display_text = f"{prefix}\n\n{result['response']}"
        st.markdown(display_text)
        st.caption(
            f"Latency: {result['latency_s']}s · "
            f"Cost: ${result['cost_usd']:.5f} · "
            f"Routed by: {result['routing_reason']}"
        )

        st.session_state.messages.append({
            "role":    "assistant",
            "content": display_text,
            "meta":    result,
        })

# ── Live charts (below chat) ─────────────────────────────────────
logs = metrics_tracker.all_logs()
if logs:
    st.divider()
    st.subheader("📈 Live Performance")
    df = pd.DataFrame(logs)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Latency per request (s)**")
        st.bar_chart(df, x="timestamp", y="latency_s", color="model")
    with c2:
        st.write("**Cost per request ($)**")
        st.area_chart(df, x="timestamp", y="cost_usd", color="model")
