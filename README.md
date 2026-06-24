# Spotify Discovery Review Engine

> **AI-powered review analysis engine** that ingests Spotify user feedback at scale and surfaces *why users struggle to discover new music and fall back on repeat-listening* — built as Part 1 of a Growth PM capstone.

**🚀 Live app: [https://spotifyreviewscrapper-02.streamlit.app/]**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange)](https://console.groq.com)
[![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-blue)](https://trychroma.com)

---

## What it does

The engine ingests **7,431 raw reviews** from four channels, tags every review with discovery-related themes using an LLM, stores embeddings in a local vector database, and answers six product-strategy questions with cited evidence — all on free infrastructure.

### Six PS questions it answers

| # | Question | How answered |
|---|---|---|
| Q1 | Why do users struggle to discover new music? | Theme aggregation (`discovery_friction`, `filter_bubble`, `no_control_or_intent`) + RAG |
| Q2 | What are the most common recommendation frustrations? | Theme aggregation (`generic_recommendations`, `discover_weekly_dailymix`, `autoplay_radio_loop`) + sentiment |
| Q3 | What listening behaviors are users trying to achieve? | Free-text RAG retrieval — no tag field; inferred from review language |
| Q4 | What causes repeat-listening loops? | Theme aggregation (`recommendation_repetition`, `wants_new_but_safe`) + RAG |
| Q5 | Which user segments face different discovery challenges? | Segment × theme cross-tab (pre-computed) |
| Q6 | What unmet needs recur consistently? | Score-weighted RAG (Reddit upvotes + forum kudos surface highest-signal items) |

### Key findings (from 5,708 tagged reviews)

- **Discovery Friction** is the #1 theme — 1,104 reviews (19.4%) flag a hard time finding unfamiliar music
- **Power Users** experience the highest discovery friction rate at 28.2%
- **Autoplay Radio Loop** ranks as the #2 weighted unmet need by community upvote signal (score = 1,504)
- **69.4%** of reviews are discovery-related; **56%** of the corpus is negative sentiment — users want more, not less, from Spotify's algorithm

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES (Phase 1)                      │
│   Google Play (3,169) │ App Store RSS (2,500) │ Reddit (1,735)     │
│                        │ Community Forum (27)                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │  7,431 raw items
                                  ▼
                         ┌─────────────────┐
                         │   clean.py       │  dedupe · normalize · schema
                         └────────┬────────┘
                                  │  6,778 rows (reviews.parquet)
                                  ▼
                         ┌─────────────────┐
                         │   tag.py         │  Groq llama-3.1-8b · batch=10
                         └────────┬────────┘  10 themes · sentiment · segment
                                  │  5,808 tagged rows
                                  ▼
                         ┌─────────────────┐
                         │   embed.py       │  all-MiniLM-L6-v2 · local CPU
                         └────────┬────────┘
                                  │  5,708 vectors
                                  ▼
                    ┌─────────────────────────┐
                    │   ChromaDB (chroma_db/)  │  cosine similarity index
                    └──────┬──────────────────┘
                           │
          ┌────────────────┼─────────────────┐
          ▼                ▼                  ▼
   ┌────────────┐  ┌─────────────┐   ┌──────────────┐
   │aggregate.py│  │   rag.py    │   │   app.py      │
   │summary.json│  │answers.json │   │ Streamlit UI  │
   └────────────┘  └─────────────┘   └──────────────┘
   theme counts     Q1–Q6 answered    Tab 1: Dashboard
   segment cross-   Groq 70B, T=0.3  Tab 2: Live RAG
   tab, Q6 weights  cited quotes
```

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| **Tagging LLM** | Groq `llama-3.1-8b-instant` | High-throughput, free tier, JSON-structured output |
| **Synthesis LLM** | Groq `llama-3.3-70b-versatile` | Richer reasoning for Q1–Q6 answers + RAG |
| **Embeddings** | `sentence-transformers all-MiniLM-L6-v2` | Local CPU — Groq has no embedding endpoint |
| **Vector store** | ChromaDB (persistent, local) | Free, file-based, cosine search |
| **Scraping** | `google-play-scraper`, iTunes RSS, Pullpush.io | No-auth APIs; Reddit's public JSON deprecated |
| **Data pipeline** | pandas + pyarrow (Parquet) | Resumable stages — each step writes to disk |
| **UI** | Streamlit + Plotly | Fast to build, one-click deploy on Community Cloud |
| **Rate limiting** | `tenacity` exponential backoff | On all Groq calls + scrapers |

---

## Re-runnable workflow

The engine is designed as a **re-runnable workflow** — not a one-time script on a frozen dataset. Point it at any Play Store app or CSV and regenerate all insights end-to-end with one command.

```bash
# Analyse a different app (e.g. YouTube Music)
python src/run_workflow.py --source playstore --app-id com.google.android.apps.youtube.music --count 500

# Analyse a CSV of your own reviews
python src/run_workflow.py --source csv --path my_reviews.csv --max-rows 1000
```

**CSV format:** must contain a `text` column. Optional: `rating` (1–5), `date`, `source`, `score`, `url`.

**Rate-limit note:** tagging uses Groq's free tier (~40 rows/min). Keep `--count` / `--max-rows` ≤ 1,500 to stay within the daily token budget. Larger datasets will take 30–60 min.

**Tab 3 in the app** ("Run Workflow") provides the same capability with a UI — app ID input or CSV upload, per-stage progress bar, and immediate results rendering. Real-time streaming is intentionally out of scope per the assignment brief.

---

## Pipeline — run in order

Each stage writes to disk; the pipeline is fully resumable.

```bash
# 1 — Scrape
python src/scrape_playstore.py     # → data/raw/playstore.json  (3,169 reviews)
python src/scrape_appstore.py      # → data/raw/appstore.json   (2,500 reviews)
python src/scrape_reddit.py        # → data/raw/reddit.json     (1,735 threads, via Pullpush.io)
python src/scrape_forum.py         # → data/raw/forum.json      (27 community posts)

# 2 — Clean
python src/clean.py                # → data/clean/reviews.parquet  (6,778 rows)

# 3 — Tag
python src/tag.py                  # → data/tagged/reviews_tagged.parquet
                                   #   primary: Groq 8B; fallback: Gemini 2.5 on daily-quota hit

# 4 — Embed
python src/embed.py                # → chroma_db/  (5,708 vectors)

# 5 — Aggregate + Answer
python src/aggregate.py            # → data/insights/summary.json
python src/rag.py                  # → data/insights/answers.json  (Q1–Q6 with cited quotes)

# 6 — Launch
streamlit run app.py
```

> **Note:** Steps 4–6 use pre-committed outputs (`chroma_db/`, `data/insights/`) — the deployed app does **not** re-scrape or re-embed on load.

---

## Local setup

```bash
git clone https://github.com/akashh0210/spotify_review_scrapper.git
cd spotify_review_scrapper

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env.local      # fill in your keys
```

### Keys required (all free)

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — optional, Groq daily-quota fallback |

---

## Repository structure

```
spotify_review_scrapper/
├── app.py                       # Streamlit app (two tabs)
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml              # dark theme, headless=true
│
├── src/
│   ├── scrape_playstore.py      # Google Play → data/raw/
│   ├── scrape_appstore.py       # iTunes RSS → data/raw/
│   ├── scrape_reddit.py         # Pullpush.io → data/raw/
│   ├── scrape_forum.py          # community.spotify.com → data/raw/
│   ├── clean.py                 # dedupe + normalize → reviews.parquet
│   ├── tag.py                   # Groq 8B tagging (+ Gemini fallback)
│   ├── retag_errors.py          # Groq re-tag of tag_error rows
│   ├── retag_gemini.py          # Groq re-tag of mislabeled Gemini rows
│   ├── embed.py                 # sentence-transformers → ChromaDB
│   ├── aggregate.py             # theme counts, segment cross-tab, Q6 weights
│   └── rag.py                   # Q1–Q6 RAG answers with cited quotes
│
├── data/
│   ├── raw/          (gitignored — regenerable)
│   ├── clean/        (gitignored — regenerable)
│   ├── tagged/       (gitignored — regenerable)
│   └── insights/     ✅ committed — summary.json + answers.json
│
└── chroma_db/        ✅ committed — 5,708 pre-built vectors
```

---

## Tagging schema

Every review is tagged with this strict JSON schema (Groq 8B, temperature 0):

```json
{
  "themes":            ["discovery_friction", "no_control_or_intent"],
  "sentiment":         "negative",
  "segment":           "power_user",
  "discovery_related": true,
  "one_line":          "User can't steer the algorithm toward new artists.",
  "language":          "en",
  "tagged_by":         "groq-llama-3.1-8b"
}
```

**Allowed themes (closed list):** `recommendation_repetition` · `discovery_friction` · `generic_recommendations` · `discover_weekly_dailymix` · `autoplay_radio_loop` · `no_control_or_intent` · `filter_bubble` · `wants_new_but_safe` · `positive_discovery` · `non_discovery`

**Segments:** `casual` · `power_user` · `genre_explorer` · `mood_context_listener` · `podcast_listener` · `unknown`

---

## Deploy (Streamlit Community Cloud)

1. Fork / push to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo → main file: `app.py`
3. Under **Advanced → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. Deploy — first boot takes ~5 min (installs sentence-transformers + downloads model)
5. Subsequent loads are fast — `chroma_db/` and `data/insights/` are pre-committed

---

## Guardrails

- All cited quotes in the UI are **real retrieved text** — never model-generated
- Every Groq call is wrapped in `tenacity` exponential backoff
- Secrets loaded from `st.secrets` on Cloud, `.env.local` locally — never committed
- `data/raw/`, `data/clean/`, `data/tagged/` are gitignored (regenerable from source)
