"""Spotify Discovery Review Engine — Phase 6 Streamlit app.

Tab 1: Insights Dashboard (reads from data/insights/*.json — no API calls on load)
Tab 2: Ask the Reviews (live RAG: ChromaDB retrieval + Groq 70B streaming)
"""

import json
import os
from pathlib import Path

import chromadb
import groq as groq_sdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

st.set_page_config(
    page_title="Spotify Discovery Review Engine",
    page_icon="🎵",
    layout="wide",
)

load_dotenv(".env.local")
load_dotenv()

# ── Spotify dark theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── CSS variables ── */
:root {
    --bg:        #121212;
    --card:      #181818;
    --card-hov:  #282828;
    --accent:    #1DB954;
    --accent-lt: #1ed760;
    --text:      #FFFFFF;
    --text-sec:  #B3B3B3;
    --border:    #282828;
    --negative:  #E01E5A;
    --font:      -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-size: 17px !important;
}
.stApp, .main, section.main > div {
    background-color: var(--bg) !important;
}
header[data-testid="stHeader"] {
    background-color: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}
/* Push everything below the native Streamlit header bar */
.block-container {
    padding-top: 4rem !important;
    max-width: 1200px !important;
}
/* Toolbar stays readable but doesn't crowd the tab strip */
[data-testid="stToolbar"] {
    background-color: var(--bg) !important;
}

/* ── App header bar ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--border);
}
.app-header-dot {
    width: 44px;
    height: 44px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.app-header-title {
    font-size: 34px;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0;
}
.app-header-sub {
    font-size: 14px;
    color: var(--text-sec);
    margin: 3px 0 0 0;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 24px 18px;
    min-height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: border-color 0.15s ease;
}
.metric-card:hover { border-color: var(--accent); }
.metric-number {
    color: var(--accent);
    font-size: 40px;
    font-weight: 700;
    line-height: 1.05;
    margin-bottom: 6px;
}
.metric-label {
    color: var(--text-sec);
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.metric-sub {
    color: var(--text-sec);
    font-size: 13px;
    margin-top: 4px;
    opacity: 0.8;
}

/* ── Section headers ── */
h2, h3, [data-testid="stHeading"] {
    color: var(--text) !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-top: 28px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-sec) !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg) !important;
    padding-top: 20px !important;
}

/* ── Expanders ── */
details {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
}
details[open] { border-color: var(--accent) !important; }
details summary {
    padding: 14px 18px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    cursor: pointer !important;
    list-style: none !important;
}
details summary:hover { background: var(--card-hov) !important; }
details > div {
    padding: 16px 20px 20px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--card) !important;
    color: var(--text) !important;
}

/* ── Quote blocks ── */
.quote-block {
    border-left: 3px solid var(--accent);
    padding: 10px 16px;
    margin: 10px 0 14px 0;
    background: rgba(29, 185, 84, 0.07);
    border-radius: 0 8px 8px 0;
}
.quote-text {
    font-style: italic;
    color: var(--text-sec);
    font-size: 15px;
    line-height: 1.55;
    margin-bottom: 7px;
}
.quote-meta { font-size: 13px; color: #666; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 26px 0 !important; }

/* ── Captions ── */
small, .stCaption, [data-testid="stCaptionContainer"] p {
    color: var(--text-sec) !important;
    font-size: 14px !important;
}

/* ── Regular buttons ── */
.stButton > button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--card-hov) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Primary / submit button ── */
.stButton > button[kind="primary"],
button[data-testid="stFormSubmitButton"] {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 32px !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stFormSubmitButton"]:hover {
    background: var(--accent-lt) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    min-height: 100px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
.stTextArea label { color: var(--text-sec) !important; font-size: 14px !important; }

/* ── Form ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
}

/* ── Info/error/markdown ── */
.stAlert { background: var(--card) !important; border-color: var(--border) !important; }
.stMarkdown p { color: var(--text) !important; font-size: 17px !important; line-height: 1.6 !important; }
.stMarkdown li { color: var(--text) !important; font-size: 16px !important; }
.stMarkdown strong { color: var(--text) !important; }
blockquote {
    border-left: 3px solid var(--accent) !important;
    background: rgba(29,185,84,0.06) !important;
    color: var(--text-sec) !important;
    padding: 10px 16px !important;
    border-radius: 0 6px 6px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── paths & constants ─────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
SUMMARY_PATH = BASE / "data" / "insights" / "summary.json"
ANSWERS_PATH = BASE / "data" / "insights" / "answers.json"
CHROMA_DIR   = str(BASE / "chroma_db")

COLLECTION_NAME = "spotify_reviews"
EMBED_MODEL     = "all-MiniLM-L6-v2"
MODEL_RAG       = "llama-3.3-70b-versatile"
RAG_TOP_K       = 10

DISCOVERY_THEMES = [
    "discovery_friction", "recommendation_repetition", "generic_recommendations",
    "discover_weekly_dailymix", "autoplay_radio_loop", "no_control_or_intent",
    "filter_bubble", "wants_new_but_safe", "positive_discovery",
]

EXAMPLE_QUESTIONS = [
    "Why do users struggle to discover new music?",
    "What are the most common frustrations with recommendations?",
    "What listening behaviors are users trying to achieve?",
    "What causes users to repeatedly listen to the same content?",
    "Which user segments experience different discovery challenges?",
    "What unmet needs emerge consistently across reviews?",
]

RAG_SYSTEM = (
    "You are a product analyst synthesizing real Spotify user reviews for a Growth PM. "
    "Answer using ONLY the retrieved evidence provided — no external knowledge. "
    "Name specific Spotify features (Discover Weekly, Daily Mix, Song Radio, autoplay) when mentioned in evidence. "
    "Do not quote verbatim in the answer body — cited reviews are shown separately."
)

# ── data loaders (cached — unchanged) ────────────────────────────────────────

@st.cache_data
def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_answers() -> list[dict]:
    return json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))["questions"]


@st.cache_resource
def get_collection():
    ef     = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def _groq_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_groq_client() -> groq_sdk.Groq:
    return groq_sdk.Groq(api_key=_groq_key())


# ── display helpers ───────────────────────────────────────────────────────────

def _stars(rating) -> str:
    if rating is None:
        return ""
    try:
        n = round(float(rating))
        return "⭐" * max(0, min(n, 5)) if n > 0 else ""
    except (TypeError, ValueError):
        return ""


def _metric_card(label: str, number: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-number">{number}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{sub_html}</div>'
    )


def _render_quote(q: dict) -> None:
    text   = (q.get("text") or "")[:280]
    src    = q.get("source", "")
    rating = q.get("rating")
    score  = q.get("score")
    stars  = _stars(rating)
    parts  = [f"<strong>{src.title()}</strong>"]
    if stars:
        parts.append(stars)
    if score and float(score) > 0:
        parts.append(f"👍 {int(float(score))}")
    meta = "  ·  ".join(parts)
    st.markdown(
        f'<div class="quote-block">'
        f'<div class="quote-text">"{text}"</div>'
        f'<div class="quote-meta">{meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── dark Plotly base layout ───────────────────────────────────────────────────

_DARK = dict(
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    font=dict(color="#FFFFFF", size=14,
              family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"),
    title_font=dict(size=18, color="#FFFFFF"),
    xaxis=dict(gridcolor="#282828", color="#B3B3B3", zerolinecolor="#282828", linecolor="#282828"),
    yaxis=dict(gridcolor="#282828", color="#B3B3B3", zerolinecolor="#282828", linecolor="#282828"),
)


# ── charts ────────────────────────────────────────────────────────────────────

def _theme_bar(summary: dict) -> go.Figure:
    tf = summary.get("theme_frequency", {})
    rows = sorted(
        [(t.replace("_", " ").title(), v["count"])
         for t, v in tf.items() if t in set(DISCOVERY_THEMES)],
        key=lambda x: x[1],
    )[-8:]
    labels, counts = zip(*rows) if rows else ([], [])
    fig = go.Figure(go.Bar(
        x=list(counts), y=list(labels), orientation="h",
        marker_color="#1DB954",
        text=[str(c) for c in counts], textposition="outside",
        textfont=dict(color="#B3B3B3", size=13),
    ))
    fig.update_layout(
        title="Top discovery themes",
        height=340, showlegend=False,
        margin=dict(t=50, b=20, l=0, r=70),
        xaxis_title="Reviews", yaxis_title="",
        **_DARK,
    )
    return fig


def _sentiment_bar(summary: dict) -> go.Figure:
    sent   = summary.get("overview", {}).get("sentiment", {})
    labels = ["positive", "neutral", "negative"]
    counts = [sent.get(l, {}).get("count", 0) for l in labels]
    pcts   = [sent.get(l, {}).get("pct", 0) for l in labels]
    colors = {"positive": "#1DB954", "neutral": "#535353", "negative": "#E01E5A"}
    fig = go.Figure(go.Bar(
        x=[l.title() for l in labels], y=counts,
        marker_color=[colors[l] for l in labels],
        text=[f"{p:.0f}%" for p in pcts], textposition="auto",
        textfont=dict(color="#FFFFFF", size=14, family="sans-serif"),
    ))
    fig.update_layout(
        title="Sentiment split",
        height=340, showlegend=False,
        margin=dict(t=50, b=20, l=0, r=20),
        yaxis_title="Reviews", xaxis_title="",
        **_DARK,
    )
    return fig


def _segment_heatmap(summary: dict) -> go.Figure:
    seg_data = summary.get("segment_x_theme", {})
    segments = [s for s in ["casual", "power_user", "genre_explorer",
                             "mood_context_listener", "podcast_listener"]
                if s in seg_data]
    tf = summary.get("theme_frequency", {})
    top6 = [t for t, _ in sorted(
        [(t, v["count"]) for t, v in tf.items() if t in set(DISCOVERY_THEMES)],
        key=lambda x: x[1], reverse=True,
    )[:6]]

    z, text_z = [], []
    for seg in segments:
        rates = seg_data[seg].get("theme_rates", {})
        row   = [rates.get(theme, {}).get("pct", 0.0) for theme in top6]
        z.append(row)
        text_z.append([f"{v:.0f}%" for v in row])

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[t.replace("_", "<br>") for t in top6],
        y=[s.replace("_", " ").title() for s in segments],
        colorscale=[[0, "#1a1a1a"], [0.4, "#0d3320"], [1.0, "#1DB954"]],
        text=text_z, texttemplate="%{text}",
        textfont=dict(color="#FFFFFF", size=13),
        hovertemplate="Segment: %{y}<br>Theme: %{x}<br>Rate: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Theme rate by user segment (%)",
        height=290,
        margin=dict(t=50, b=0, l=0, r=0),
        xaxis_title="", yaxis_title="",
        **_DARK,
    )
    return fig


# ── main app ──────────────────────────────────────────────────────────────────

summary = load_summary()
answers = load_answers()
ov      = summary.get("overview", {})

tab1, tab2, tab3 = st.tabs(["📊  Insights Dashboard", "💬  Ask the Reviews (RAG)", "⚙️  Run Workflow"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INSIGHTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Header bar
    st.markdown("""
    <div class="app-header">
      <div class="app-header-dot"><svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg"><circle cx="22" cy="22" r="22" fill="#1DB954"/><path d="M33 17.5C26.8 14.2 15.5 14 10 16.5" stroke="white" stroke-width="2.8" stroke-linecap="round" fill="none"/><path d="M31.5 22.8C26 20 16.5 19.5 11.5 21.8" stroke="white" stroke-width="2.4" stroke-linecap="round" fill="none"/><path d="M30 27.8C25 25.5 17 25.2 13 27" stroke="white" stroke-width="2" stroke-linecap="round" fill="none"/></svg></div>
      <div>
        <div class="app-header-title">Spotify Discovery Review Engine</div>
        <div class="app-header-sub">5,708 reviews &nbsp;·&nbsp; 4 sources</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1 — metric cards
    seg_analysis = summary.get("segment_x_theme", {}).get("_analysis", {}).get(
        "highest_discovery_friction", {}
    )
    top_seg     = seg_analysis.get("segment", "power_user").replace("_", " ").title()
    top_seg_pct = seg_analysis.get("pct", 0)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            _metric_card("Total reviews analyzed",
                         f"{ov.get('total_reviews', 0):,}"),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            _metric_card("Discovery-related",
                         f"{ov.get('discovery_related', {}).get('pct', 0):.1f}%",
                         "tagged with at least one discovery theme"),
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            _metric_card("Most frustrated segment",
                         top_seg,
                         f"{top_seg_pct:.1f}% Discovery Friction Rate"),
            unsafe_allow_html=True,
        )

    st.divider()

    # Row 2 — two bar charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_theme_bar(summary), use_container_width=True)
    with c2:
        st.plotly_chart(_sentiment_bar(summary), use_container_width=True)

    st.divider()

    # Row 3 — segment × theme heatmap (Q5)
    st.plotly_chart(_segment_heatmap(summary), use_container_width=True)

    st.divider()

    # Row 4 — six Q&A sections
    st.subheader("The Six Discovery Questions")
    for qa in answers:
        qid = qa.get("id", "")
        with st.expander(f"{qid} — {qa['question']}"):
            st.markdown(qa.get("answer", ""))

            findings = qa.get("key_findings", [])
            if findings:
                st.markdown("**Key findings:**")
                for f in findings:
                    st.markdown(f"- {f}")

            quotes = qa.get("supporting_quotes", [])
            if quotes:
                st.markdown("**Evidence:**")
                for q in quotes:
                    _render_quote(q)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASK THE REVIEWS (LIVE RAG)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="app-header">
      <div class="app-header-dot"><svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg"><circle cx="22" cy="22" r="22" fill="#1DB954"/><path d="M33 17.5C26.8 14.2 15.5 14 10 16.5" stroke="white" stroke-width="2.8" stroke-linecap="round" fill="none"/><path d="M31.5 22.8C26 20 16.5 19.5 11.5 21.8" stroke="white" stroke-width="2.4" stroke-linecap="round" fill="none"/><path d="M30 27.8C25 25.5 17 25.2 13 27" stroke="white" stroke-width="2" stroke-linecap="round" fill="none"/></svg></div>
      <div>
        <div class="app-header-title">Ask the Reviews</div>
        <div class="app-header-sub">Answers grounded in real reviews &nbsp;·&nbsp; Groq llama-3.3-70b-versatile</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if "rag_query" not in st.session_state:
        st.session_state.rag_query = ""

    st.markdown("<div style='color:#B3B3B3;font-size:14px;margin-bottom:10px;'>Example questions</div>",
                unsafe_allow_html=True)
    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    all_cols  = row1_cols + row2_cols
    for i, (col, eq) in enumerate(zip(all_cols, EXAMPLE_QUESTIONS)):
        with col:
            if st.button(eq, key=f"eq_{i}", use_container_width=True):
                st.session_state.rag_query = eq
                st.rerun()

    st.write("")

    with st.form(key="rag_form", clear_on_submit=False):
        query = st.text_area(
            "Your question:",
            value=st.session_state.rag_query,
            height=100,
            placeholder="e.g. Why does Discover Weekly feel repetitive?",
        )
        submitted = st.form_submit_button("Ask the Reviews", type="primary")

    if submitted and query.strip():
        st.session_state.rag_query = query

        try:
            coll    = get_collection()
            results = coll.query(
                query_texts=[query],
                n_results=RAG_TOP_K,
                include=["documents", "metadatas", "distances"],
            )
            docs   = results["documents"][0]
            metas  = results["metadatas"][0]
            dists  = results["distances"][0]
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            st.stop()

        evidence_parts = []
        for doc, meta in zip(docs, metas):
            src    = meta.get("source", "?")
            rating = f"rating={meta['rating']:.0f}" if meta.get("rating", -1) > 0 else ""
            score  = f"upvotes={meta['score']:.0f}" if meta.get("score", 0) > 0 else ""
            ctx    = ", ".join(filter(None, [src, rating, score]))
            evidence_parts.append(f"[{ctx}]\n{doc[:400]}")
        evidence = "\n\n---\n\n".join(evidence_parts)

        st.subheader("Answer")
        groq_client = get_groq_client()

        def _stream():
            stream = groq_client.chat.completions.create(
                model=MODEL_RAG,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM},
                    {"role": "user", "content": (
                        f"Question: {query}\n\n"
                        f"Retrieved evidence ({RAG_TOP_K} reviews):\n{evidence}"
                    )},
                ],
                stream=True,
                timeout=90,
            )
            for chunk in stream:
                yield chunk.choices[0].delta.content or ""

        try:
            st.write_stream(_stream())
        except Exception as exc:
            st.error(f"Generation error: {exc}")

        st.subheader(f"Retrieved reviews (top {RAG_TOP_K})")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            src     = meta.get("source", "?")
            themes  = meta.get("themes", "")
            rating  = f"⭐ {meta['rating']:.0f}" if meta.get("rating", -1) > 0 else ""
            score   = f"👍 {int(meta['score'])}" if meta.get("score", 0) > 0 else ""
            snippet = doc[:60].replace("\n", " ")
            label   = f"[{i}] {src.title()} — {snippet}…"
            with st.expander(label):
                st.write(doc[:500])
                meta_parts = filter(None, [src, rating, score, f"🏷️ {themes}" if themes else ""])
                st.caption(f"dist={dist:.3f}  ·  " + "  ·  ".join(meta_parts))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RUN WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="app-header">
      <div class="app-header-dot"><svg width="44" height="44" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg"><circle cx="22" cy="22" r="22" fill="#1DB954"/><path d="M33 17.5C26.8 14.2 15.5 14 10 16.5" stroke="white" stroke-width="2.8" stroke-linecap="round" fill="none"/><path d="M31.5 22.8C26 20 16.5 19.5 11.5 21.8" stroke="white" stroke-width="2.4" stroke-linecap="round" fill="none"/><path d="M30 27.8C25 25.5 17 25.2 13 27" stroke="white" stroke-width="2" stroke-linecap="round" fill="none"/></svg></div>
      <div>
        <div class="app-header-title">Run Workflow</div>
        <div class="app-header-sub">Re-run the full pipeline against a new data source</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Point the engine at a new app or data file and regenerate all insights end-to-end.
    Each run fully replaces the previous pipeline outputs (raw → clean → tag → embed → aggregate).

    > **Note:** Real-time streaming is intentionally out of scope. The tagging step
    > (Stage 3) calls Groq's API and takes ~2 min per 100 rows on the free tier.
    > Keep reviews under **1,500** to stay within the daily token budget.
    """)

    source_choice = st.radio(
        "Data source", ["Play Store app", "Upload CSV"],
        horizontal=True,
    )

    st.divider()

    if source_choice == "Play Store app":
        col_a, col_b = st.columns([2, 1])
        with col_a:
            wf_app_id = st.text_input(
                "App ID", value="com.spotify.music",
                help="Google Play package name, e.g. com.netflix.mediaclient",
            )
        with col_b:
            wf_count = st.slider("Reviews to scrape", 100, 2000, 500, 100)

        if wf_count > 1500:
            st.warning(
                f"⚠️ {wf_count} reviews ≈ {wf_count * 150:,} tokens — "
                "may exceed Groq free-tier daily limit. Consider 500–1,000."
            )

        run_btn = st.button("▶  Run Pipeline", type="primary", key="wf_run_ps")
        uploaded_csv = None

    else:
        uploaded_csv = st.file_uploader(
            "Upload CSV — must contain a **text** column. "
            "Optional: rating (1–5), date, score, source, url.",
            type=["csv"],
        )
        wf_max_rows = st.slider("Max rows to process", 100, 2000, 500, 100)
        wf_app_id   = "com.spotify.music"
        wf_count    = 500

        if wf_max_rows > 1500:
            st.warning(
                f"⚠️ {wf_max_rows} rows ≈ {wf_max_rows * 150:,} tokens — "
                "may exceed Groq free-tier daily limit."
            )

        run_btn = st.button(
            "▶  Run Pipeline", type="primary", key="wf_run_csv",
            disabled=(uploaded_csv is None),
        )

    # ── pipeline execution ──────────────────────────────────────────────────
    if run_btn:
        import sys as _sys
        _sys.path.insert(0, str(BASE / "src"))
        from run_workflow import run_pipeline as _run_pipeline

        # Save uploaded CSV to a temp path
        csv_tmp = None
        if source_choice == "Upload CSV" and uploaded_csv is not None:
            import tempfile, shutil as _shutil
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded_csv.read())
            tmp.close()
            csv_tmp = tmp.name

        # Progress state
        progress_bar   = st.progress(0.0)
        stage_text     = st.empty()
        stage_log      = st.container()
        completed: list[str] = []

        def _on_stage(name: str, current: int, total: int) -> None:
            progress_bar.progress(current / total)
            stage_text.info(f"⏳ Stage {current}/{total}: **{name}**")
            completed.append(f"✅ Stage {current}: {name}")
            with stage_log:
                for msg in completed[:-1]:
                    st.caption(msg)

        try:
            wf_result = _run_pipeline(
                source="playstore" if source_choice == "Play Store app" else "csv",
                app_id=wf_app_id,
                count=wf_count,
                csv_path=csv_tmp,
                max_csv_rows=wf_max_rows if source_choice == "Upload CSV" else 2000,
                progress_callback=_on_stage,
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            if csv_tmp:
                import os as _os
                _os.unlink(csv_tmp)
            st.stop()

        if csv_tmp:
            import os as _os
            _os.unlink(csv_tmp)

        progress_bar.progress(1.0)
        stage_text.success(
            f"✅ Pipeline complete in {wf_result['elapsed_s']:.0f}s "
            f"({wf_result['elapsed_s']/60:.1f} min)"
        )

        # Invalidate caches so the dashboard reflects fresh data
        load_summary.clear()
        load_answers.clear()
        st.cache_resource.clear()

        # ── results ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Results")

        r1, r2, r3 = st.columns(3)
        r1.markdown(
            _metric_card("Raw items", f"{wf_result['raw_count']:,}"),
            unsafe_allow_html=True,
        )
        r2.markdown(
            _metric_card("Clean rows", f"{wf_result['clean_count']:,}"),
            unsafe_allow_html=True,
        )
        r3.markdown(
            _metric_card("Discovery-related", f"{wf_result['disc_pct']:.1f}%"),
            unsafe_allow_html=True,
        )

        st.divider()

        # Fresh summary and answers from disk
        fresh_summary = load_summary()
        fresh_answers = load_answers()

        st.subheader("Theme Frequencies")
        st.plotly_chart(_theme_bar(fresh_summary), use_container_width=True)

        st.subheader("The Six Discovery Questions")
        for qa in fresh_answers:
            qid = qa.get("id", "")
            with st.expander(f"{qid} — {qa['question']}"):
                st.markdown(qa.get("answer", ""))
                findings = qa.get("key_findings", [])
                if findings:
                    st.markdown("**Key findings:**")
                    for f in findings:
                        st.markdown(f"- {f}")
                for q in qa.get("supporting_quotes", []):
                    _render_quote(q)

        st.info(
            "Dashboard tab now reflects these fresh insights. "
            "Reload the page to see the updated charts and answers."
        )
