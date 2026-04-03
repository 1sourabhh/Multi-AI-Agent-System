import streamlit as st
import os
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, .stApp {
        background-color: #080b14 !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    .stApp { color: #c9d1e0; }

    .main-header {
        padding: 1.2rem 0 0.5rem;
        border-bottom: 1px solid #1a2035;
        margin-bottom: 1.2rem;
    }
    .main-title {
        font-size: 1.75rem; font-weight: 700;
        color: #e8edf5; letter-spacing: -0.02em; margin: 0;
    }
    .main-subtitle {
        font-size: 0.82rem; color: #4a5568;
        margin-top: 0.25rem; font-family: 'JetBrains Mono', monospace;
    }

    /* Mode tabs */
    .mode-bar {
        display: flex; gap: 0.5rem; margin-bottom: 1.2rem;
    }
    .mode-btn {
        padding: 0.4rem 1.1rem; border-radius: 8px;
        font-size: 0.82rem; font-weight: 600; cursor: pointer;
        border: 1px solid #1a2035; background: #0d1220; color: #6b7a99;
        transition: all 0.2s;
    }
    .mode-btn.active {
        background: #1e2d47; border-color: #3b5bdb; color: #93b4f8;
    }

    /* Chat bubbles */
    .chat-container {
        display: flex; flex-direction: column;
        gap: 0.75rem; margin-bottom: 1rem;
        max-height: 480px; overflow-y: auto;
        padding-right: 0.25rem;
    }
    .chat-msg {
        display: flex; gap: 0.6rem; align-items: flex-start;
    }
    .chat-msg.user   { flex-direction: row-reverse; }
    .chat-avatar {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
    }
    .avatar-user { background: #1e2d47; color: #93b4f8; }
    .avatar-ai   { background: #1a3020; color: #34d399; }
    .chat-bubble {
        max-width: 82%; padding: 0.65rem 0.9rem;
        border-radius: 10px; font-size: 0.85rem; line-height: 1.65;
    }
    .bubble-user {
        background: #1e2d47; color: #c9d1e0;
        border: 1px solid #2a3d5e;
    }
    .bubble-ai {
        background: #0d1220; color: #c9d1e0;
        border: 1px solid #1a2035;
    }
    .bubble-meta {
        font-size: 0.65rem; color: #3d4f6b;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.3rem;
    }

    /* Memory badge */
    .memory-badge {
        display: inline-block; padding: 2px 8px;
        border-radius: 6px; font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace; font-weight: 600;
        background: #1c0f30; color: #a78bfa;
        border: 1px solid #3b2060; margin-bottom: 0.5rem;
    }

    /* Agent cards */
    .agent-card {
        background: #0d1220; border: 1px solid #1a2035;
        border-radius: 10px; padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem; transition: border-color 0.3s;
    }
    .agent-card.active { border-color: #3b5bdb; box-shadow: 0 0 0 1px #3b5bdb22; }
    .agent-card.done   { border-color: #1a6b3c; }
    .agent-row {
        display: flex; align-items: center;
        justify-content: space-between; margin-bottom: 0.35rem;
    }
    .agent-name {
        font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    .name-researcher { color: #f97316; }
    .name-writer     { color: #60a5fa; }
    .name-critic     { color: #a78bfa; }
    .name-save       { color: #34d399; }
    .agent-text {
        font-size: 0.82rem; color: #6b7a99;
        line-height: 1.55; font-family: 'JetBrains Mono', monospace;
    }
    .agent-text.active { color: #c9d1e0; }

    /* Pills */
    .pill {
        font-size: 0.67rem; font-weight: 600;
        padding: 2px 9px; border-radius: 999px; letter-spacing: 0.04em;
    }
    .pill-waiting  { background: #12182b; color: #3d4f6b; border: 1px solid #1e2d47; }
    .pill-running  { background: #0f1e3d; color: #60a5fa; border: 1px solid #1e3a6e; }
    .pill-done     { background: #0a2018; color: #34d399; border: 1px solid #1a4a30; }
    .pill-revision { background: #1c0f30; color: #a78bfa; border: 1px solid #3b2060; }

    /* Metrics */
    .metrics-row {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 0.6rem; margin-bottom: 1rem;
    }
    .metric-card {
        background: #0d1220; border: 1px solid #1a2035;
        border-radius: 8px; padding: 0.75rem 1rem;
    }
    .metric-label {
        font-size: 0.65rem; color: #4a5568; text-transform: uppercase;
        letter-spacing: 0.07em; font-weight: 600; margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.35rem; font-weight: 700; color: #e8edf5;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-sub { font-size: 0.68rem; color: #4a5568; font-family: 'JetBrains Mono', monospace; }

    /* Output */
    .output-box {
        background: #0d1220; border: 1px solid #1a2035;
        border-radius: 10px; padding: 1.25rem; color: #c9d1e0;
        font-size: 0.88rem; line-height: 1.8; white-space: pre-wrap;
        max-height: 400px; overflow-y: auto;
    }
    .output-placeholder {
        color: #2a3550; font-style: italic; min-height: 120px;
        display: flex; align-items: center; justify-content: center;
    }

    /* Save */
    .save-panel {
        background: #0a1628; border: 1px solid #1e3a5f;
        border-radius: 10px; padding: 1.1rem; margin-top: 0.75rem;
    }
    .save-title {
        font-size: 0.75rem; font-weight: 600; color: #60a5fa;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.75rem;
    }

    /* History */
    .history-item {
        background: #0d1220; border: 1px solid #1a2035;
        border-radius: 8px; padding: 0.65rem 0.9rem; margin-bottom: 0.4rem;
    }
    .history-query {
        font-size: 0.78rem; color: #c9d1e0;
        white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis; max-width: 100%;
    }
    .history-meta {
        font-size: 0.65rem; color: #3d4f6b;
        font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem;
    }

    /* Env status */
    .env-status {
        padding: 0.5rem 0.75rem; border-radius: 8px;
        font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 0.5rem;
    }
    .env-ok      { background: #0a2018; border: 1px solid #1a4a30; color: #34d399; }
    .env-missing { background: #200a0a; border: 1px solid #4a1a1a; color: #f87171; }

    /* Streamlit overrides */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #0d1220 !important; border: 1px solid #1a2035 !important;
        color: #c9d1e0 !important; border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    }
    .stButton > button[kind="primary"] { background: #3b5bdb !important; border: none !important; }
    .stButton > button[kind="primary"]:hover { background: #4c6ef5 !important; }
    div[data-testid="stSidebar"] {
        background: #080b14 !important; border-right: 1px solid #1a2035 !important;
    }
    .stExpander { border: 1px solid #1a2035 !important; border-radius: 8px !important; }
    hr { border-color: #1a2035 !important; }
    label { color: #6b7a99 !important; font-size: 0.8rem !important; }
    .stSpinner > div { border-top-color: #3b5bdb !important; }
    div[data-testid="stChatInput"] { background: #0d1220 !important; }
</style>
""", unsafe_allow_html=True)

# ── AgentState ────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:         Annotated[list, operator.add]
    query:            str
    research_output:  str
    research_sources: list
    final_output:     str
    critic_score:     int
    critic_feedback:  str
    critic_approved:  bool
    revision_count:   int
    past_context:     str

# ── Session defaults ──────────────────────────────────────────────────────────
_defaults = {
    "mode":             "single",       # "single" | "chat"
    "stage":            "idle",
    "research_output":  "",
    "research_sources": [],
    "final_output":     "",
    "critic_score":     0,
    "critic_feedback":  "",
    "critic_approved":  False,
    "revision_count":   0,
    "last_query":       "",
    "error_msg":        "",
    "start_time":       None,
    "elapsed":          0.0,
    "history":          [],
    "thread_id":        f"session-{int(time.time())}",
    "langgraph_memory": MemorySaver(),
    # Chat mode
    "chat_messages":    [],             # [{role, content, meta}]
    "chat_processing":  False,
    # Vector store
    "vectorstore":      None,
    "vectorstore_count": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(model="qwen/qwen3-32b", api_key=GROQ_API_KEY, temperature=0.6)

def score_color(score):
    if score >= 8: return "#34d399"
    if score >= 6: return "#fbbf24"
    return "#f87171"

def pill(kind, label=None):
    labels = {"waiting": "Waiting", "running": "● Running", "done": "✓ Done", "revision": "↻ Revising"}
    text = label or labels.get(kind, kind)
    return f'<span class="pill pill-{kind}">{text}</span>'

def agent_card(name, css_class, status, text):
    active_cls = "active" if status == "running" else ("done" if status == "done" else "")
    text_cls   = "active" if status in ("running", "done") else ""
    return f"""<div class="agent-card {active_cls}">
      <div class="agent-row"><span class="agent-name {css_class}">{name}</span>{pill(status)}</div>
      <div class="agent-text {text_cls}">{text}</div>
    </div>"""

def preview(text, fallback="Waiting…"):
    if not text: return fallback
    return (text[:220] + "…" if len(text) > 220 else text).replace("\n", " ")

def agent_status(agent):
    stage = st.session_state.stage
    if stage == "researching" and agent == "researcher": return "running"
    if stage == "writing"     and agent == "writer":     return "running"
    if stage == "critiquing"  and agent == "critic":     return "running"
    if stage == "done":                                   return "done"
    if stage in ("writing", "critiquing", "done") and agent == "researcher": return "done"
    if stage in ("critiquing", "done")            and agent == "writer":     return "done"
    return "waiting"

# ── Vector store helpers ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

def store_in_vectorstore(query, research, output, score):
    """Embed and store a completed pipeline run."""
    emb = get_embeddings()
    text = f"Query: {query}\n\nResearch Summary:\n{research[:1000]}\n\nAnswer:\n{output[:1500]}"
    meta = {
        "query":    query,
        "score":    str(score),
        "saved_at": datetime.now().isoformat(),
    }
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_texts([text], emb, metadatas=[meta])
    else:
        st.session_state.vectorstore.add_texts([text], metadatas=[meta])
    st.session_state.vectorstore_count += 1

def retrieve_past_context(query, k=3):
    """Retrieve relevant past research for current query."""
    if st.session_state.vectorstore is None:
        return ""
    emb     = get_embeddings()
    results = st.session_state.vectorstore.similarity_search(query, k=k)
    if not results:
        return ""
    parts = []
    for i, doc in enumerate(results):
        parts.append(f"[Past context {i+1}] {doc.page_content[:600]}")
    return "\n\n".join(parts)

# ── Run pipeline (shared by both modes) ──────────────────────────────────────
def run_full_pipeline(query):
    """Execute Researcher → Writer → Critic and return final_output."""
    # 1. Retrieve past context from vector store
    past_ctx = retrieve_past_context(query)

    # 2. Researcher
    tool    = TavilySearchResults(max_results=6, tavily_api_key=TAVILY_API_KEY)
    results = tool.invoke(query)
    if not results:
        raise ValueError("Tavily returned no results.")
    snippets = "\n\n".join(f"[{i+1}] {r.get('content','')}" for i, r in enumerate(results))
    sources  = [{"url": r.get("url",""), "title": r.get("title") or r.get("url","")} for r in results]

    st.session_state.research_output  = snippets
    st.session_state.research_sources = sources
    st.session_state.stage            = "writing"

    # 3. Writer (with past context injected)
    llm = get_llm()
    past_section = f"\n\nRelevant past research from memory:\n{past_ctx}" if past_ctx else ""
    writer_prompt = f"""You are an expert writer and analyst.

User Query: {query}

Research Findings:
{snippets}{past_section}

Instructions:
- Use clear headings and sections
- Cite sources using [1], [2], etc.
- Be factual, thorough, and professional
- End with a Key Takeaways section

Your response:"""

    # If this is a follow-up in chat, include conversation history
    messages_to_send = []
    if st.session_state.mode == "chat" and len(st.session_state.chat_messages) > 0:
        for m in st.session_state.chat_messages[-6:]:  # last 3 turns
            if m["role"] == "user":
                messages_to_send.append(HumanMessage(content=m["content"]))
            else:
                messages_to_send.append(AIMessage(content=m["content"]))
    messages_to_send.append(HumanMessage(content=writer_prompt))

    response = llm.invoke(messages_to_send)
    if not response or not response.content:
        raise ValueError("Writer returned empty response.")
    output = response.content
    st.session_state.final_output = output
    st.session_state.stage        = "critiquing"

    # 4. Critic
    critic_prompt = f"""You are a strict quality critic.
Review the response and return ONLY valid JSON — no markdown, no explanation.

Query: {query}
Response: {output}

Return exactly:
{{"score": <1-10>, "feedback": "<one sentence>", "approved": <true if score>=7>}}"""

    cr = llm.invoke([HumanMessage(content=critic_prompt)])
    raw = re.sub(r"```json|```", "", cr.content.strip()).strip()
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        data     = json.loads(match.group())
        score    = int(data.get("score", 7))
        feedback = str(data.get("feedback", ""))
        approved = bool(data.get("approved", score >= 7))
    else:
        score, feedback, approved = 7, "Could not parse critic response.", True

    st.session_state.critic_score    = score
    st.session_state.critic_feedback = feedback
    st.session_state.critic_approved = approved

    # 5. Revision if needed
    if not approved and st.session_state.revision_count < 2:
        st.session_state.revision_count += 1
        revision_prompt = f"""{writer_prompt}

REVISION REQUIRED. Critic feedback: {feedback}
Please address this feedback specifically."""
        rev = llm.invoke([HumanMessage(content=revision_prompt)])
        output = rev.content
        st.session_state.final_output = output

    # 6. Store in vector store
    store_in_vectorstore(query, snippets, output, score)

    # 7. Save to history
    if st.session_state.start_time:
        st.session_state.elapsed = round(time.time() - st.session_state.start_time, 1)
    st.session_state.history.append({
        "query":  query,
        "output": output,
        "score":  score,
        "time":   datetime.now().strftime("%H:%M"),
    })
    st.session_state.stage = "done"
    return output, sources, score

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ API Status")
    groq_ok   = bool(GROQ_API_KEY)
    tavily_ok = bool(TAVILY_API_KEY)
    st.markdown(
        f'<div class="env-status {"env-ok" if groq_ok else "env-missing"}">'
        f'{"✓" if groq_ok else "✗"} GROQ_API_KEY {"loaded" if groq_ok else "missing"}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="env-status {"env-ok" if tavily_ok else "env-missing"}">'
        f'{"✓" if tavily_ok else "✗"} TAVILY_API_KEY {"loaded" if tavily_ok else "missing"}</div>',
        unsafe_allow_html=True
    )
    if not groq_ok or not tavily_ok:
        st.warning("Add missing keys to `.env` and restart.")

    st.divider()
    st.markdown("### 🧠 Vector Memory")
    vc = st.session_state.vectorstore_count
    st.markdown(
        f'<div class="env-status env-ok">✓ {vc} session{"s" if vc!=1 else ""} stored in FAISS</div>'
        if vc > 0 else
        '<div class="env-status pill-waiting" style="background:#12182b;border:1px solid #1e2d47;color:#3d4f6b;padding:0.5rem 0.75rem;border-radius:8px;font-size:0.78rem;font-family:monospace;">No sessions stored yet</div>',
        unsafe_allow_html=True
    )
    if vc > 0 and st.button("🗑 Clear memory", use_container_width=True):
        st.session_state.vectorstore       = None
        st.session_state.vectorstore_count = 0
        st.rerun()

    st.divider()
    st.markdown("### 🤖 Agent Pipeline")
    st.markdown("1. **Researcher** → Tavily Search\n2. **Writer** → Qwen3-32B\n3. **Critic** → Reviews & scores\n4. **Save** → File / Download")

    st.divider()
    st.markdown("### 🕓 History")
    if not st.session_state.history:
        st.markdown('<span style="color:#3d4f6b;font-size:0.8rem;">No queries yet</span>', unsafe_allow_html=True)
    else:
        for h in reversed(st.session_state.history[-8:]):
            score_txt = f"Score: {h['score']}/10 · " if h.get("score") else ""
            st.markdown(f"""<div class="history-item">
              <div class="history-query">{h['query'][:55]}{'…' if len(h['query'])>55 else ''}</div>
              <div class="history-meta">{score_txt}{h['time']}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    with st.expander("🐛 Debug"):
        st.write("Mode:", st.session_state.mode)
        st.write("Stage:", st.session_state.stage)
        st.write("Vector docs:", st.session_state.vectorstore_count)
        st.write("Revisions:", st.session_state.revision_count)
        if st.session_state.error_msg:
            st.error(st.session_state.error_msg)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="main-title">🤖 Multi-Agent AI System</div>
  <div class="main-subtitle">researcher → writer → critic → save &nbsp;·&nbsp; langgraph · langchain · groq · faiss · qwen3-32b</div>
</div>
""", unsafe_allow_html=True)

if not groq_ok or not tavily_ok:
    st.error("⚠️ API keys not loaded. Check your `.env` file and restart.")
    st.code("GROQ_API_KEY=gsk_...\nTAVILY_API_KEY=tvly_...", language="bash")
    st.stop()

# ── Mode selector ─────────────────────────────────────────────────────────────
m1, m2, _ = st.columns([1, 1, 4])
with m1:
    if st.button(
        "📄 Single Query" if st.session_state.mode != "single" else "📄 Single Query ●",
        use_container_width=True,
        type="primary" if st.session_state.mode == "single" else "secondary"
    ):
        st.session_state.mode  = "single"
        st.session_state.stage = "idle"
        st.rerun()
with m2:
    if st.button(
        "💬 Chat Mode" if st.session_state.mode != "chat" else "💬 Chat Mode ●",
        use_container_width=True,
        type="primary" if st.session_state.mode == "chat" else "secondary"
    ):
        st.session_state.mode  = "chat"
        st.session_state.stage = "idle"
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  SINGLE QUERY MODE
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.mode == "single":

    # Metrics
    stage           = st.session_state.stage
    elapsed_display = f"{st.session_state.elapsed:.1f}s" if st.session_state.elapsed else "—"
    sources_count   = len(st.session_state.research_sources)
    words_count     = len(st.session_state.final_output.split()) if st.session_state.final_output else 0
    critic_display  = f"{st.session_state.critic_score}/10" if st.session_state.critic_score else "—"
    sc              = score_color(st.session_state.critic_score) if st.session_state.critic_score else "#3d4f6b"
    mem_count       = st.session_state.vectorstore_count

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-label">Critic Score</div>
        <div class="metric-value" style="color:{sc}">{critic_display}</div>
        <div class="metric-sub">quality rating</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Sources Used</div>
        <div class="metric-value">{sources_count if sources_count else '—'}</div>
        <div class="metric-sub">web results</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Output Words</div>
        <div class="metric-value">{words_count if words_count else '—'}</div>
        <div class="metric-sub">in final answer</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Memory</div>
        <div class="metric-value" style="color:#a78bfa">{mem_count}</div>
        <div class="metric-sub">sessions in FAISS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.05, 1], gap="large")

    with col_left:
        st.markdown("#### Query")

        # Show if past context is available
        if st.session_state.vectorstore_count > 0:
            st.markdown(
                f'<div class="memory-badge">🧠 {st.session_state.vectorstore_count} past session(s) in memory — agents will use relevant context</div>',
                unsafe_allow_html=True
            )

        query = st.text_area(
            label="q", label_visibility="collapsed",
            placeholder="e.g. Compare LangGraph vs AutoGen vs CrewAI for multi-agent systems",
            height=100, key="single_query_input"
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            run_clicked = st.button(
                "▶ Run agents", type="primary", use_container_width=True,
                disabled=stage in ("researching", "writing", "critiquing")
            )
        with c2:
            if st.button("✕ Reset", use_container_width=True):
                for k in ["stage","research_output","research_sources","final_output",
                          "critic_score","critic_feedback","critic_approved",
                          "revision_count","error_msg","start_time","elapsed"]:
                    st.session_state[k] = _defaults[k]
                st.rerun()

        if run_clicked and query.strip():
            for k in ["research_output","research_sources","final_output",
                      "critic_score","critic_feedback","critic_approved","error_msg"]:
                st.session_state[k] = _defaults[k]
            st.session_state.revision_count = 0
            st.session_state.last_query     = query.strip()
            st.session_state.start_time     = time.time()
            st.session_state.elapsed        = 0.0
            st.session_state.stage          = "researching"
            st.rerun()

        st.markdown("#### Agent activity")
        st.markdown(agent_card("Researcher Agent","name-researcher", agent_status("researcher"),
            preview(st.session_state.research_output, "Will search the web using Tavily…")), unsafe_allow_html=True)
        st.markdown(agent_card("Writer Agent","name-writer", agent_status("writer"),
            preview(st.session_state.final_output, "Will structure findings into a response…")), unsafe_allow_html=True)

        critic_st = agent_status("critic")
        if not st.session_state.critic_approved and st.session_state.revision_count > 0 and stage != "done":
            critic_st = "revision"
        st.markdown(agent_card("Critic Agent","name-critic", critic_st,
            preview(st.session_state.critic_feedback, "Will review and score the writer's output…")), unsafe_allow_html=True)

        if st.session_state.critic_score:
            with st.expander("🔍 View critic details"):
                st.markdown(f"**Score:** {st.session_state.critic_score}/10 · **{'✓ Approved' if st.session_state.critic_approved else '↻ Revised'}**")
                st.progress(st.session_state.critic_score / 10)
                st.markdown(f"**Feedback:** {st.session_state.critic_feedback}")
                st.markdown(f"**Revision rounds:** {st.session_state.revision_count}")

        st.markdown(agent_card("Save Output","name-save", agent_status("save"),
            "Ready — use the save panel →" if stage == "done" else "Waiting for pipeline…"), unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Final output")
        if st.session_state.final_output and stage == "done":
            st.markdown(f'<div class="output-box">{st.session_state.final_output}</div>', unsafe_allow_html=True)

            if st.session_state.research_sources:
                with st.expander("📚 Sources used"):
                    for i, src in enumerate(st.session_state.research_sources):
                        url   = src.get("url","")
                        title = src.get("title") or url or f"Source {i+1}"
                        st.markdown(f"**[{i+1}]** [{title}]({url})" if url else f"**[{i+1}]** {title}")

            st.markdown('<div class="save-panel"><div class="save-title">💾 Save output</div>', unsafe_allow_html=True)
            default_name = f"agent_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sc1, sc2 = st.columns([1.6, 1])
            with sc1:
                filename = st.text_input("fn", value=default_name, label_visibility="collapsed")
            with sc2:
                fmt = st.selectbox("fmt", [".txt",".md",".json"], label_visibility="collapsed")
            folder = st.text_input("folder", value=os.path.expanduser("~/Downloads"), label_visibility="collapsed")
            sb1, sb2 = st.columns(2)
            with sb1:
                if st.button("💾 Save to disk", use_container_width=True, type="primary"):
                    try:
                        os.makedirs(folder, exist_ok=True)
                        full_path = os.path.join(folder, filename + fmt)
                        content = json.dumps({"query":st.session_state.last_query,"output":st.session_state.final_output,"score":st.session_state.critic_score,"sources":st.session_state.research_sources,"saved_at":datetime.now().isoformat()},indent=2) if fmt==".json" else st.session_state.final_output
                        open(full_path,"w",encoding="utf-8").write(content)
                        st.success(f"Saved → `{os.path.abspath(full_path)}`")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with sb2:
                dl = json.dumps({"query":st.session_state.last_query,"output":st.session_state.final_output},indent=2) if fmt==".json" else st.session_state.final_output
                st.download_button("⬇ Download", data=dl, file_name=filename+fmt, mime="application/json" if fmt==".json" else "text/plain", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif stage in ("researching","writing","critiquing"):
            st.markdown('<div class="output-box output-placeholder">Agents working…</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="output-box output-placeholder">Run a query to see output here.</div>', unsafe_allow_html=True)

    # Pipeline execution for single mode
    MAX_REVISIONS = 2
    if st.session_state.stage == "researching":
        with st.spinner("🔍 Researcher Agent searching…"):
            try:
                past_ctx = retrieve_past_context(st.session_state.last_query)
                tool     = TavilySearchResults(max_results=6, tavily_api_key=TAVILY_API_KEY)
                results  = tool.invoke(st.session_state.last_query)
                if not results: raise ValueError("Tavily returned no results.")
                snippets = "\n\n".join(f"[{i+1}] {r.get('content','')}" for i,r in enumerate(results))
                sources  = [{"url":r.get("url",""),"title":r.get("title") or r.get("url","")} for r in results]
                st.session_state.research_output  = snippets
                st.session_state.research_sources = sources
                st.session_state.stage            = "writing"
                st.session_state["_past_ctx"]     = past_ctx
            except Exception as e:
                st.session_state.error_msg = f"Researcher failed: {e}"
                st.session_state.stage     = "idle"
                st.error(st.session_state.error_msg)
        st.rerun()

    elif st.session_state.stage == "writing":
        rev_note   = f"\n\nREVISION #{st.session_state.revision_count}. Address: {st.session_state.critic_feedback}" if st.session_state.revision_count > 0 else ""
        past_ctx   = st.session_state.get("_past_ctx", "")
        past_sec   = f"\n\nRelevant past research from memory:\n{past_ctx}" if past_ctx else ""
        with st.spinner(f"✍️ Writer Agent composing{'  (revision #'+str(st.session_state.revision_count)+')' if st.session_state.revision_count else ''}…"):
            try:
                llm      = get_llm()
                prompt   = f"""You are an expert writer and analyst.

User Query: {st.session_state.last_query}

Research Findings:
{st.session_state.research_output}{past_sec}

Instructions:
- Use clear headings and sections
- Cite sources using [1], [2], etc.
- Be factual, thorough, and professional
- End with a Key Takeaways section{rev_note}

Your response:"""
                response = llm.invoke([HumanMessage(content=prompt)])
                if not response or not response.content: raise ValueError("Writer returned empty response.")
                st.session_state.final_output = response.content
                st.session_state.stage        = "critiquing"
            except Exception as e:
                st.session_state.error_msg = f"Writer failed: {e}"
                st.session_state.stage     = "idle"
                st.error(st.session_state.error_msg)
        st.rerun()

    elif st.session_state.stage == "critiquing":
        with st.spinner("🔍 Critic Agent reviewing…"):
            try:
                llm   = get_llm()
                cr    = llm.invoke([HumanMessage(content=f"""Review and return ONLY valid JSON.
Query: {st.session_state.last_query}
Response: {st.session_state.final_output}
Return: {{"score":<1-10>,"feedback":"<one sentence>","approved":<true if score>=7>}}""")])
                raw   = re.sub(r"```json|```","",cr.content.strip()).strip()
                m     = re.search(r'\{.*?\}', raw, re.DOTALL)
                data  = json.loads(m.group()) if m else {"score":7,"feedback":"OK","approved":True}
                score    = int(data.get("score",7))
                feedback = str(data.get("feedback",""))
                approved = bool(data.get("approved", score>=7))
                st.session_state.critic_score    = score
                st.session_state.critic_feedback = feedback
                st.session_state.critic_approved = approved
                if approved or st.session_state.revision_count >= MAX_REVISIONS:
                    if st.session_state.start_time:
                        st.session_state.elapsed = round(time.time()-st.session_state.start_time,1)
                    store_in_vectorstore(st.session_state.last_query, st.session_state.research_output, st.session_state.final_output, score)
                    st.session_state.history.append({"query":st.session_state.last_query,"output":st.session_state.final_output,"score":score,"time":datetime.now().strftime("%H:%M")})
                    st.session_state.stage = "done"
                else:
                    st.session_state.revision_count += 1
                    st.session_state.stage           = "writing"
            except Exception as e:
                st.session_state.critic_score    = 0
                st.session_state.critic_feedback = f"Critic error: {e}"
                st.session_state.critic_approved = True
                if st.session_state.start_time:
                    st.session_state.elapsed = round(time.time()-st.session_state.start_time,1)
                store_in_vectorstore(st.session_state.last_query, st.session_state.research_output, st.session_state.final_output, 0)
                st.session_state.history.append({"query":st.session_state.last_query,"output":st.session_state.final_output,"score":0,"time":datetime.now().strftime("%H:%M")})
                st.session_state.stage = "done"
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  CHAT MODE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.mode == "chat":

    col_chat, col_agents = st.columns([1.2, 1], gap="large")

    with col_chat:
        st.markdown("#### 💬 Conversation")

        if st.session_state.vectorstore_count > 0:
            st.markdown(
                f'<div class="memory-badge">🧠 {st.session_state.vectorstore_count} past session(s) in memory · agents use relevant context automatically</div>',
                unsafe_allow_html=True
            )

        # Render chat history
        if st.session_state.chat_messages:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    chat_html += f"""<div class="chat-msg user">
                      <div class="chat-avatar avatar-user">U</div>
                      <div><div class="chat-bubble bubble-user">{msg["content"]}</div>
                      <div class="bubble-meta" style="text-align:right">{msg.get("time","")}</div></div>
                    </div>"""
                else:
                    score_badge = f' &nbsp; <span style="color:{score_color(msg["score"])};font-size:0.68rem;">Score {msg["score"]}/10</span>' if msg.get("score") else ""
                    ctx_badge   = ' &nbsp; <span style="color:#a78bfa;font-size:0.68rem;">🧠 used memory</span>' if msg.get("used_memory") else ""
                    chat_html += f"""<div class="chat-msg">
                      <div class="chat-avatar avatar-ai">AI</div>
                      <div><div class="chat-bubble bubble-ai">{msg["content"][:800]}{"…" if len(msg["content"])>800 else ""}</div>
                      <div class="bubble-meta">{msg.get("time","")}{score_badge}{ctx_badge}</div></div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#2a3550;font-style:italic;padding:1rem 0;">Start a conversation — ask anything and the agents will research and respond.</div>', unsafe_allow_html=True)

        # Chat input
        chat_input = st.chat_input("Ask anything…", disabled=st.session_state.chat_processing)

        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.chat_messages  = []
                st.session_state.chat_processing = False
                st.session_state.stage           = "idle"
                st.rerun()

        if chat_input and not st.session_state.chat_processing:
            # Add user message
            st.session_state.chat_messages.append({
                "role":    "user",
                "content": chat_input,
                "time":    datetime.now().strftime("%H:%M"),
            })
            st.session_state.chat_processing = True
            st.session_state.last_query      = chat_input
            st.session_state.start_time      = time.time()
            st.session_state.stage           = "researching"
            for k in ["research_output","research_sources","final_output","critic_score","critic_feedback","critic_approved","error_msg"]:
                st.session_state[k] = _defaults[k]
            st.session_state.revision_count = 0
            st.rerun()

    with col_agents:
        st.markdown("#### Agent activity")
        stage = st.session_state.stage

        st.markdown(agent_card("Researcher Agent","name-researcher", agent_status("researcher"),
            preview(st.session_state.research_output, "Will search the web using Tavily…")), unsafe_allow_html=True)
        st.markdown(agent_card("Writer Agent","name-writer", agent_status("writer"),
            preview(st.session_state.final_output, "Will structure findings into a response…")), unsafe_allow_html=True)

        critic_st = agent_status("critic")
        if not st.session_state.critic_approved and st.session_state.revision_count > 0 and stage != "done":
            critic_st = "revision"
        st.markdown(agent_card("Critic Agent","name-critic", critic_st,
            preview(st.session_state.critic_feedback, "Will review and score the writer's output…")), unsafe_allow_html=True)

        if st.session_state.critic_score:
            with st.expander("🔍 Critic details"):
                st.markdown(f"**Score:** {st.session_state.critic_score}/10")
                st.progress(st.session_state.critic_score / 10)
                st.markdown(f"**Feedback:** {st.session_state.critic_feedback}")

        # Memory panel
        st.markdown("#### 🧠 Vector memory")
        vc = st.session_state.vectorstore_count
        if vc > 0:
            st.markdown(f'<div class="memory-badge">FAISS · {vc} document(s) stored · sentence-transformers/all-MiniLM-L6-v2</div>', unsafe_allow_html=True)
            with st.expander("📂 Stored sessions"):
                for i, h in enumerate(reversed(st.session_state.history[-5:])):
                    st.markdown(f"**[{i+1}]** {h['query'][:60]} — Score {h.get('score','?')}/10")
        else:
            st.markdown('<span style="color:#3d4f6b;font-size:0.82rem;">Memory is empty — run a query to start building it.</span>', unsafe_allow_html=True)

    # Chat pipeline execution
    MAX_REVISIONS = 2
    if st.session_state.stage == "researching" and st.session_state.chat_processing:
        with st.spinner("🔍 Researcher searching…"):
            try:
                past_ctx = retrieve_past_context(st.session_state.last_query)
                tool     = TavilySearchResults(max_results=6, tavily_api_key=TAVILY_API_KEY)
                results  = tool.invoke(st.session_state.last_query)
                if not results: raise ValueError("Tavily returned no results.")
                snippets = "\n\n".join(f"[{i+1}] {r.get('content','')}" for i,r in enumerate(results))
                sources  = [{"url":r.get("url",""),"title":r.get("title") or r.get("url","")} for r in results]
                st.session_state.research_output  = snippets
                st.session_state.research_sources = sources
                st.session_state["_past_ctx"]     = past_ctx
                st.session_state["_used_memory"]  = bool(past_ctx)
                st.session_state.stage            = "writing"
            except Exception as e:
                st.session_state.error_msg       = f"Researcher failed: {e}"
                st.session_state.stage           = "idle"
                st.session_state.chat_processing = False
                st.error(st.session_state.error_msg)
        st.rerun()

    elif st.session_state.stage == "writing" and st.session_state.chat_processing:
        past_ctx  = st.session_state.get("_past_ctx","")
        past_sec  = f"\n\nRelevant past research from memory:\n{past_ctx}" if past_ctx else ""
        rev_note  = f"\n\nREVISION #{st.session_state.revision_count}. Address: {st.session_state.critic_feedback}" if st.session_state.revision_count > 0 else ""

        # Build conversation history for context
        history_msgs = []
        for m in st.session_state.chat_messages[:-1][-6:]:  # last 3 turns excluding latest
            if m["role"] == "user":
                history_msgs.append(HumanMessage(content=m["content"]))
            else:
                history_msgs.append(AIMessage(content=m["content"][:600]))

        with st.spinner(f"✍️ Writer composing{'  (revision #'+str(st.session_state.revision_count)+')' if st.session_state.revision_count else ''}…"):
            try:
                llm    = get_llm()
                prompt = f"""You are an expert writer and analyst in a multi-turn conversation.

Current Query: {st.session_state.last_query}

Research Findings:
{st.session_state.research_output}{past_sec}

Instructions:
- Use clear headings and sections
- Cite sources using [1], [2], etc.
- If this is a follow-up question, refer to the previous conversation naturally
- Be factual, thorough, and professional
- End with a Key Takeaways section{rev_note}

Your response:"""
                response = llm.invoke(history_msgs + [HumanMessage(content=prompt)])
                if not response or not response.content: raise ValueError("Writer returned empty.")
                st.session_state.final_output = response.content
                st.session_state.stage        = "critiquing"
            except Exception as e:
                st.session_state.error_msg       = f"Writer failed: {e}"
                st.session_state.stage           = "idle"
                st.session_state.chat_processing = False
                st.error(st.session_state.error_msg)
        st.rerun()

    elif st.session_state.stage == "critiquing" and st.session_state.chat_processing:
        with st.spinner("🔍 Critic reviewing…"):
            try:
                llm  = get_llm()
                cr   = llm.invoke([HumanMessage(content=f"""Review and return ONLY valid JSON.
Query: {st.session_state.last_query}
Response: {st.session_state.final_output}
Return: {{"score":<1-10>,"feedback":"<one sentence>","approved":<true if score>=7>}}""")])
                raw  = re.sub(r"```json|```","",cr.content.strip()).strip()
                m    = re.search(r'\{.*?\}', raw, re.DOTALL)
                data = json.loads(m.group()) if m else {"score":7,"feedback":"OK","approved":True}
                score    = int(data.get("score",7))
                feedback = str(data.get("feedback",""))
                approved = bool(data.get("approved", score>=7))
                st.session_state.critic_score    = score
                st.session_state.critic_feedback = feedback
                st.session_state.critic_approved = approved

                if approved or st.session_state.revision_count >= MAX_REVISIONS:
                    # Store in vector memory
                    store_in_vectorstore(st.session_state.last_query, st.session_state.research_output, st.session_state.final_output, score)
                    # Add AI reply to chat
                    st.session_state.chat_messages.append({
                        "role":        "assistant",
                        "content":     st.session_state.final_output,
                        "score":       score,
                        "used_memory": st.session_state.get("_used_memory", False),
                        "sources":     st.session_state.research_sources,
                        "time":        datetime.now().strftime("%H:%M"),
                    })
                    st.session_state.history.append({"query":st.session_state.last_query,"output":st.session_state.final_output,"score":score,"time":datetime.now().strftime("%H:%M")})
                    if st.session_state.start_time:
                        st.session_state.elapsed = round(time.time()-st.session_state.start_time,1)
                    st.session_state.stage           = "done"
                    st.session_state.chat_processing = False
                else:
                    st.session_state.revision_count += 1
                    st.session_state.stage           = "writing"
            except Exception as e:
                st.session_state.chat_messages.append({
                    "role":"assistant","content":st.session_state.final_output,
                    "score":0,"used_memory":False,"sources":[],"time":datetime.now().strftime("%H:%M"),
                })
                st.session_state.stage           = "done"
                st.session_state.chat_processing = False
        st.rerun()