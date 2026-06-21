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

load_dotenv(".env.local") or load_dotenv()

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

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Discovery Review Engine",
    page_icon="🎵",
    layout="wide",
)

# ── data loaders (cached) ──────────────────────────────────────────────────────

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


def _render_quote(q: dict) -> None:
    text   = (q.get("text") or "")[:280]
    src    = q.get("source", "")
    rating = q.get("rating")
    score  = q.get("score")
    parts  = [f"**{src}**"]
    stars  = _stars(rating)
    if stars:
        parts.append(stars)
    if score and float(score) > 0:
        parts.append(f"👍 {int(float(score))}")
    st.markdown(f"> *\"{text}\"*")
    st.caption("  ·  ".join(parts))


# ── charts ────────────────────────────────────────────────────────────────────

def _theme_bar(summary: dict) -> go.Figure:
    tf = summary.get("theme_frequency", {})
    rows = sorted(
        [(t.replace("_", " "), v["count"])
         for t, v in tf.items() if t in set(DISCOVERY_THEMES)],
        key=lambda x: x[1],
    )[-8:]  # bottom-8 so highest appears at top in horizontal bar
    labels, counts = zip(*rows) if rows else ([], [])
    fig = go.Figure(go.Bar(
        x=list(counts), y=list(labels), orientation="h",
        marker_color="#1DB954",
        text=[str(c) for c in counts], textposition="outside",
    ))
    fig.update_layout(
        title="Top discovery themes (review count)",
        height=340, showlegend=False,
        margin=dict(t=40, b=20, l=0, r=60),
        xaxis_title="Reviews", yaxis_title="",
    )
    return fig


def _sentiment_bar(summary: dict) -> go.Figure:
    sent   = summary.get("overview", {}).get("sentiment", {})
    labels = ["positive", "neutral", "negative"]
    counts = [sent.get(l, {}).get("count", 0) for l in labels]
    pcts   = [sent.get(l, {}).get("pct", 0) for l in labels]
    colors = {"positive": "#1DB954", "neutral": "#BBBBBB", "negative": "#E01E5A"}
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=[colors[l] for l in labels],
        text=[f"{p:.0f}%" for p in pcts], textposition="auto",
    ))
    fig.update_layout(
        title="Sentiment split",
        height=340, showlegend=False,
        margin=dict(t=40, b=20, l=0, r=20),
        yaxis_title="Reviews", xaxis_title="",
    )
    return fig


def _segment_heatmap(summary: dict) -> go.Figure:
    seg_data = summary.get("segment_x_theme", {})
    segments = [s for s in ["casual", "power_user", "genre_explorer",
                             "mood_context_listener", "podcast_listener"]
                if s in seg_data]
    # Top 6 discovery themes by count
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
        y=[s.replace("_", " ") for s in segments],
        colorscale="Blues",
        text=text_z, texttemplate="%{text}",
        hovertemplate="Segment: %{y}<br>Theme: %{x}<br>Rate: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Theme rate by user segment (%)",
        height=280,
        margin=dict(t=40, b=0, l=0, r=0),
        xaxis_title="", yaxis_title="",
    )
    return fig


# ── main app ──────────────────────────────────────────────────────────────────

summary = load_summary()
answers = load_answers()
ov      = summary.get("overview", {})

tab1, tab2 = st.tabs(["📊 Insights Dashboard", "💬 Ask the Reviews (RAG)"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INSIGHTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Spotify Discovery Review Engine")
    st.caption("5,708 reviews across 4 sources · themes tagged by Groq llama-3.1-8b · answers by llama-3.3-70b")

    # Row 1 — metric cards
    seg_analysis = summary.get("segment_x_theme", {}).get("_analysis", {}).get(
        "highest_discovery_friction", {}
    )
    top_seg     = seg_analysis.get("segment", "power_user").replace("_", " ")
    top_seg_pct = seg_analysis.get("pct", 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total reviews analyzed", f"{ov.get('total_reviews', 0):,}")
    m2.metric(
        "Discovery-related",
        f"{ov.get('discovery_related', {}).get('pct', 0):.1f}%",
        help="Reviews tagged with at least one discovery theme",
    )
    m3.metric(
        "Most frustrated segment",
        top_seg,
        delta=f"{top_seg_pct:.1f}% discovery_friction",
        delta_color="inverse",
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
            # Answer text
            st.markdown(qa.get("answer", ""))

            # Key findings
            findings = qa.get("key_findings", [])
            if findings:
                st.markdown("**Key findings:**")
                for f in findings:
                    st.markdown(f"- {f}")

            # Supporting quotes
            quotes = qa.get("supporting_quotes", [])
            if quotes:
                st.markdown("**Evidence:**")
                for q in quotes:
                    _render_quote(q)
                    st.write("")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASK THE REVIEWS (LIVE RAG)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Ask the Reviews")
    st.caption(
        "Ask anything about Spotify user feedback — answers grounded in real reviews "
        "retrieved from ChromaDB and synthesized by Groq llama-3.3-70b-versatile."
    )

    # Session state for pre-loading example questions
    if "rag_query" not in st.session_state:
        st.session_state.rag_query = ""

    # Example question buttons (2 rows × 3 columns)
    st.markdown("**Example questions:**")
    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    all_cols  = row1_cols + row2_cols
    for i, (col, eq) in enumerate(zip(all_cols, EXAMPLE_QUESTIONS)):
        with col:
            if st.button(eq, key=f"eq_{i}", use_container_width=True):
                st.session_state.rag_query = eq
                st.rerun()

    st.write("")

    # Input form (form prevents rerun on every keystroke)
    with st.form(key="rag_form", clear_on_submit=False):
        query = st.text_area(
            "Your question:",
            value=st.session_state.rag_query,
            height=80,
            placeholder="e.g. Why does Discover Weekly feel repetitive?",
        )
        submitted = st.form_submit_button("Ask the Reviews", type="primary")

    if submitted and query.strip():
        st.session_state.rag_query = query

        # Retrieve from ChromaDB
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

        # Build evidence string for the LLM
        evidence_parts = []
        for doc, meta in zip(docs, metas):
            src    = meta.get("source", "?")
            rating = f"rating={meta['rating']:.0f}" if meta.get("rating", -1) > 0 else ""
            score  = f"upvotes={meta['score']:.0f}" if meta.get("score", 0) > 0 else ""
            ctx    = ", ".join(filter(None, [src, rating, score]))
            evidence_parts.append(f"[{ctx}]\n{doc[:400]}")
        evidence = "\n\n---\n\n".join(evidence_parts)

        # Stream answer
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

        # Show retrieved source reviews
        st.subheader(f"Retrieved reviews (top {RAG_TOP_K})")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            src     = meta.get("source", "?")
            themes  = meta.get("themes", "")
            rating  = f"⭐ {meta['rating']:.0f}" if meta.get("rating", -1) > 0 else ""
            score   = f"👍 {int(meta['score'])}" if meta.get("score", 0) > 0 else ""
            snippet = doc[:60].replace("\n", " ")
            label   = f"[{i}] {src} — {snippet}…"
            with st.expander(label):
                st.write(doc[:500])
                meta_parts = filter(None, [src, rating, score, f"🏷️ {themes}" if themes else ""])
                st.caption(f"dist={dist:.3f}  ·  " + "  ·  ".join(meta_parts))
