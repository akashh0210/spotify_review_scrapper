"""Phase 5B — answer Q1-Q6 using ChromaDB + Groq llama-3.3-70b-versatile.

Retrieves top-15 chunks per question, synthesizes a grounded answer.
All quotes are real retrieved text — never model-generated.
Writes data/insights/answers.json.

Run:  python src/rag.py   (run aggregate.py first)
"""

import json
import os
from pathlib import Path

import chromadb
import groq as groq_sdk
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv(".env.local") or load_dotenv()

BASE            = Path(__file__).parent.parent
CHROMA_DIR      = str(BASE / "chroma_db")
SUMMARY_PATH    = BASE / "data" / "insights" / "summary.json"
OUT             = BASE / "data" / "insights" / "answers.json"
COLLECTION_NAME = "spotify_reviews"
EMBED_MODEL     = "all-MiniLM-L6-v2"
MODEL_70B       = os.getenv("GROQ_RAG_MODEL", "llama-3.3-70b-versatile")
TOP_K           = 15

# ── question definitions ──────────────────────────────────────────────────────

QUESTIONS = [
    {
        "id": "Q1",
        "question": "Why do users struggle to discover new music on Spotify?",
        "query": "struggle to discover new music algorithm filter bubble can't find",
        "where": {"discovery_related": {"$eq": 1}},
        "extra_context_key": None,
    },
    {
        "id": "Q2",
        "question": "What are the most common frustrations with Spotify's music recommendations?",
        "query": "frustrating recommendations generic boring repetitive not personalized algorithm",
        "where": {"discovery_related": {"$eq": 1}},
        "extra_context_key": None,
    },
    {
        "id": "Q3",
        "question": "What listening behaviors and goals are users trying to achieve on Spotify? (What jobs are they hiring Spotify to do?)",
        "query": "I want to I wish I need looking for music mood discover explore find",
        "where": None,
        "extra_context_key": None,
    },
    {
        "id": "Q4",
        "question": "What causes users to repeatedly listen to the same content on Spotify?",
        "query": "same songs repeat loop stuck repetitive algorithm keeps playing autoplay",
        "where": {"discovery_related": {"$eq": 1}},
        "extra_context_key": None,
    },
    {
        "id": "Q5",
        "question": "Which user segments experience different discovery challenges on Spotify?",
        "query": "power user genre listener casual discover challenge algorithm playlist",
        "where": {"discovery_related": {"$eq": 1}},
        "extra_context_key": "segment_x_theme",
    },
    {
        "id": "Q6",
        "question": "What unmet needs emerge consistently across Spotify user reviews?",
        "query": "Spotify should feature request wish could would improve need missing",
        "where": None,
        "extra_context_key": "unmet_needs",
    },
]

# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM = """\
You are a product analyst synthesizing real Spotify user reviews for a Growth PM capstone.
Answer the question using ONLY the retrieved evidence provided. Rules:
- Ground every claim in the retrieved reviews. No external knowledge or assumptions.
- Write 2-3 coherent paragraphs. Be specific: name actual Spotify features mentioned
  (Discover Weekly, Daily Mix, Song Radio, autoplay, etc.) when they appear in evidence.
- Do not reproduce verbatim quotes in the answer text — supporting quotes are handled separately.
- Do not pad with generic observations. If the evidence is thin on a point, say so briefly."""


def _build_user_prompt(question: str, evidence_text: str, extra_context: str = "") -> str:
    parts = []
    if extra_context:
        parts.append(extra_context)
    parts.append(f"QUESTION: {question}\n")
    parts.append(f"RETRIEVED EVIDENCE ({TOP_K} reviews):\n{evidence_text}\n")
    parts.append("Synthesize a 2-3 paragraph answer grounded in the evidence above.")
    return "\n".join(parts)


def _format_evidence(results: dict) -> str:
    lines = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        src    = meta.get("source", "?")
        rating = f"rating={meta['rating']:.0f}" if meta.get("rating", -1) > 0 else ""
        score  = f"upvotes={meta['score']:.0f}" if meta.get("score", 0) > 0 else ""
        ctx    = ", ".join(filter(None, [src, rating, score]))
        lines.append(f"[{ctx}]\n{doc[:400]}")
    return "\n\n---\n\n".join(lines)


def _pick_quotes(results: dict, n: int = 3) -> list[dict]:
    """Pick n supporting quotes from retrieved results — real text only."""
    docs  = results["documents"][0]
    metas = results["metadatas"][0]

    # Score (upvotes/kudos) signals real-world salience — prefer those first
    scored   = [(d, m) for d, m in zip(docs, metas) if m.get("score", 0) > 0]
    unscored = [(d, m) for d, m in zip(docs, metas) if m.get("score", 0) <= 0]
    pool     = scored[:2] + unscored[:4]

    quotes = []
    seen   = set()
    for doc, meta in pool:
        text = doc[:300]
        if len(text) < 30 or text in seen:
            continue
        seen.add(text)
        quotes.append({
            "text":   text,
            "source": meta.get("source", ""),
            "rating": float(meta["rating"]) if meta.get("rating", -1) > 0 else None,
            "score":  float(meta["score"])  if meta.get("score",  0) > 0 else None,
        })
        if len(quotes) >= n:
            break
    return quotes


def _extract_key_findings(answer_text: str) -> list[str]:
    """Heuristic: take up to 3 substantial sentences from the answer as key findings."""
    sentences = [s.strip() for s in answer_text.replace("\n", " ").split(".") if len(s.strip()) > 45]
    return sentences[:3]


def _extra_context_text(summary: dict, key: str) -> str:
    if key == "segment_x_theme":
        seg_data = summary.get("segment_x_theme", {})
        lines = ["SEGMENT ANALYSIS (from full corpus):"]
        for seg, data in seg_data.items():
            if seg.startswith("_"):
                continue
            top = [t["theme"] for t in data.get("top_themes", [])[:3]]
            lines.append(
                f"  {seg}: {data['total']} reviews, disc_rate={data['discovery_rate']}%,"
                f" top themes: {', '.join(top)}"
            )
        a = seg_data.get("_analysis", {}).get("highest_discovery_friction", {})
        if a:
            lines.append(f"  Highest discovery_friction: {a['segment']} ({a['pct']}%)")
        return "\n".join(lines) + "\n"

    if key == "unmet_needs":
        unmet = summary.get("unmet_needs", [])
        lines = ["UNMET NEEDS (score-weighted — reddit upvotes + forum kudos):"]
        for u in unmet:
            lines.append(
                f"  {u['theme']}: weighted_score={u['weighted_score']:.0f},"
                f" {u['item_count']} scored items"
            )
        return "\n".join(lines) + "\n"

    return ""


# ── Groq call (retried on rate limit) ─────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(groq_sdk.RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=90),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _call_groq(client: groq_sdk.Groq, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_70B,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_prompt},
        ],
        timeout=90,
    )
    return resp.choices[0].message.content.strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GROQ_API_KEY not set in .env.local")

    groq_client = groq_sdk.Groq(api_key=api_key)

    ef      = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    chroma  = chromadb.PersistentClient(path=CHROMA_DIR)
    coll    = chroma.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    print(f"Collection: {coll.count()} vectors | Model: {MODEL_70B}")
    print(f"Answering {len(QUESTIONS)} questions...\n")

    answers      = []
    failed_flags = []

    for q in QUESTIONS:
        print(f"[{q['id']}] {q['question'][:70]}...")

        # Retrieve
        try:
            kwargs: dict = {
                "query_texts": [q["query"]],
                "n_results":   TOP_K,
                "include":     ["documents", "metadatas", "distances"],
            }
            if q.get("where"):
                kwargs["where"] = q["where"]
            results = coll.query(**kwargs)
        except Exception as exc:
            print(f"  Retrieval error: {exc}")
            answers.append({
                "id": q["id"], "question": q["question"],
                "answer": "GENERATION_FAILED",
                "key_findings": [], "supporting_quotes": [],
            })
            failed_flags.append(q["id"])
            continue

        evidence_text = _format_evidence(results)
        extra_ctx     = _extra_context_text(summary, q.get("extra_context_key") or "")
        user_prompt   = _build_user_prompt(q["question"], evidence_text, extra_ctx)

        # Generate
        try:
            answer_text = _call_groq(groq_client, user_prompt)
        except Exception as exc:
            print(f"  Generation error: {exc}")
            answer_text = "GENERATION_FAILED"
            failed_flags.append(q["id"])

        quotes       = _pick_quotes(results)
        key_findings = _extract_key_findings(answer_text) if answer_text != "GENERATION_FAILED" else []

        answers.append({
            "id":               q["id"],
            "question":         q["question"],
            "answer":           answer_text,
            "key_findings":     key_findings,
            "supporting_quotes": quotes,
        })
        print(f"  OK — {len(answer_text)} chars, {len(quotes)} quotes")

    # Write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"questions": answers}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten -> {OUT}  ({OUT.stat().st_size // 1024} KB)")

    # ── terminal print ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TERMINAL SUMMARY")
    print("=" * 70)

    # Top 5 themes
    tf = summary.get("theme_frequency", {})
    by_count = sorted(
        [(t, v["count"]) for t, v in tf.items() if t != "non_discovery"],
        key=lambda x: x[1], reverse=True,
    )[:5]
    print("\nTop 5 themes by count:")
    for t, c in by_count:
        print(f"  {t:<30} {c:>5}")

    # Top segment by discovery_friction
    a = summary.get("segment_x_theme", {}).get("_analysis", {}).get("highest_discovery_friction", {})
    print(f"\nHighest discovery_friction segment: {a.get('segment')} ({a.get('pct')}%)")

    # Top 3 unmet needs
    print("\nTop 3 unmet needs (score-weighted):")
    for u in summary.get("unmet_needs", [])[:3]:
        print(f"  {u['theme']:<30} weighted={u['weighted_score']:.0f}  items={u['item_count']}")

    # Q1 answer + quotes
    q1 = next((a for a in answers if a["id"] == "Q1"), None)
    if q1:
        print("\n" + "=" * 70)
        print("Q1 — Why do users struggle to discover new music?")
        print("-" * 70)
        print(q1["answer"])
        print("\nSupporting quotes:")
        for i, qt in enumerate(q1["supporting_quotes"], 1):
            src = qt.get("source", "")
            rating = f"rating={qt['rating']:.0f}" if qt.get("rating") else ""
            score  = f"upvotes={qt['score']:.0f}" if qt.get("score") else ""
            ctx = ", ".join(filter(None, [src, rating, score]))
            print(f"  [{i}] [{ctx}] {qt['text'][:200]!r}")

    # Q4 answer + quotes
    q4 = next((a for a in answers if a["id"] == "Q4"), None)
    if q4:
        print("\n" + "=" * 70)
        print("Q4 — What causes users to repeatedly listen to the same content?")
        print("-" * 70)
        print(q4["answer"])
        print("\nSupporting quotes:")
        for i, qt in enumerate(q4["supporting_quotes"], 1):
            src = qt.get("source", "")
            rating = f"rating={qt['rating']:.0f}" if qt.get("rating") else ""
            score  = f"upvotes={qt['score']:.0f}" if qt.get("score") else ""
            ctx = ", ".join(filter(None, [src, rating, score]))
            print(f"  [{i}] [{ctx}] {qt['text'][:200]!r}")

    # Failures
    if failed_flags:
        print(f"\nGENERATION_FAILED: {failed_flags}")
    else:
        print("\nAll 6 questions answered successfully.")


if __name__ == "__main__":
    main()
