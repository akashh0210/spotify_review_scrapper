"""Phase 4 — embed clean reviews into ChromaDB using all-MiniLM-L6-v2 (local).

Filters tag_error rows, embeds ~5,708 reviews in batches of 64, stores in a
persistent ChromaDB collection 'spotify_reviews' at ./chroma_db/.
Runs 3 test similarity queries at the end to confirm retrieval is sane.

No API calls — sentence-transformers runs fully local on CPU.

Run:  python src/embed.py
"""

import math
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv(".env.local") or load_dotenv()

BASE             = Path(__file__).parent.parent
TAGGED           = BASE / "data" / "tagged" / "reviews_tagged.parquet"
CHROMA_DIR       = str(BASE / "chroma_db")
COLLECTION_NAME  = "spotify_reviews"
EMBED_MODEL      = "all-MiniLM-L6-v2"
BATCH_SIZE       = 64

TEST_QUERIES = [
    "can't discover new music",
    "keeps playing the same songs on repeat",
    "recommendations feel generic and boring",
]


# ── metadata helpers ──────────────────────────────────────────────────────────

def _safe_num(val, default: float) -> float:
    """Convert NaN / None to a numeric sentinel. Chroma rejects null metadata."""
    try:
        v = float(val)
        return default if (v != v) else v   # v != v is True only for NaN
    except (TypeError, ValueError):
        return default


def _make_metadata(row: dict) -> dict:
    return {
        "source":            str(row.get("source")    or ""),
        "rating":            _safe_num(row.get("rating"), -1.0),
        "score":             _safe_num(row.get("score"),   0.0),
        "sentiment":         str(row.get("sentiment") or ""),
        "segment":           str(row.get("segment")   or ""),
        "themes":            ", ".join(list(row["themes"]) if row.get("themes") is not None else []),
        "discovery_related": int(bool(row.get("discovery_related", False))),
        "tagged_by":         str(row.get("tagged_by") or ""),
        "language":          str(row.get("language")  or "en"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load + filter
    df       = pd.read_parquet(TAGGED)
    df_clean = df[~df["tag_error"]].reset_index(drop=True)
    n_drop   = len(df) - len(df_clean)
    n        = len(df_clean)
    print(f"Loaded {len(df)} rows; dropped {n_drop} tag_error rows.")
    print(f"Embedding {n} rows -> collection '{COLLECTION_NAME}'\n")

    # Init ChromaDB with persistent storage + sentence-transformers EF
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} vectors — upserting (idempotent).")

    # Embed in batches
    n_batches = math.ceil(n / BATCH_SIZE)
    print(f"Model: {EMBED_MODEL}  |  {n_batches} batches of {BATCH_SIZE}")
    print("(First run downloads the model ~22 MB — subsequent runs use cache)\n")

    for i in range(n_batches):
        batch     = df_clean.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        ids       = batch["id"].astype(str).tolist()
        documents = batch["text"].fillna("").tolist()
        metadatas = [_make_metadata(row) for row in batch.to_dict("records")]

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        done = min((i + 1) * BATCH_SIZE, n)
        if (i + 1) % 10 == 0 or i == n_batches - 1:
            print(f"  batch {i+1}/{n_batches}  ({done}/{n} rows)", flush=True)

    total = collection.count()
    print(f"\nStored {total} vectors in '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # ── 3 test similarity queries ─────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("TEST QUERIES (top 3 results each, cosine distance — lower = closer)")
    print("=" * 62)

    for query in TEST_QUERIES:
        print(f"\nQuery: {query!r}")
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            snippet = doc[:120].replace("\n", " ")
            print(f"  [{rank}] source={meta['source']}  dist={dist:.3f}  "
                  f"sentiment={meta['sentiment']}")
            print(f"       themes: {meta['themes']}")
            print(f"       text  : {snippet!r}")


if __name__ == "__main__":
    main()
