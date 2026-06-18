# Spotify Discovery Review Engine

An AI-powered engine that analyzes Spotify user reviews at scale to surface **why users struggle to discover new music and fall back on repeat-listening**. Built as the Part 1 review-analysis workflow for a Growth PM capstone.

It ingests reviews from the Google Play Store, the Apple App Store, and Reddit (r/spotify, r/truespotify), tags each one with discovery-related themes and segments using an LLM, stores them in a vector database, and exposes the findings through a dashboard and a retrieval-augmented "ask the reviews" interface.

## What it answers

- Why do users struggle to discover new music?
- What are the most common frustrations with recommendations?
- What listening behaviors are users trying to achieve?
- What causes users to repeatedly listen to the same content?
- Which user segments experience different discovery challenges?
- What unmet needs emerge consistently across reviews?

## Stack

- **LLM:** Groq — `llama-3.1-8b-instant` (bulk tagging) + `llama-3.3-70b-versatile` (RAG synthesis)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (local, free)
- **Vector store:** ChromaDB
- **UI:** Streamlit + Plotly
- **Sources:** google-play-scraper, app-store-scraper, PRAW (Reddit)

> Note: Groq has no embedding endpoint, so embeddings run locally via sentence-transformers.

## Setup

```bash
git clone https://github.com/akashh0210/spotify-discovery-engine.git
cd spotify-discovery-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then fill in your keys
```

### Keys you need (all free)

- **GROQ_API_KEY** — console.groq.com
- **Reddit app creds** — reddit.com/prefs/apps → create a "script" app → copy client id + secret

Put them in `.env`.

## Run order

```bash
python src/scrape_playstore.py     # → data/raw/playstore.json
python src/scrape_appstore.py      # → data/raw/appstore.json   (best-effort)
python src/scrape_reddit.py        # → data/raw/reddit.json
python src/clean.py                # → data/clean/reviews.parquet
python src/tag.py                  # → data/tagged/reviews_tagged.parquet
python src/embed.py                # → ChromaDB collection
python src/aggregate.py            # → data/insights/summary.json + answers.json
streamlit run app.py               # launch the app
```

## Deploy

**Streamlit Community Cloud:** push to GitHub → share.streamlit.io → point at `app.py` → add `GROQ_API_KEY` (and Reddit creds if scraping live) under app **Secrets**.

**Hugging Face Spaces (fallback):** create a Streamlit Space, upload the repo, add keys under Space **Settings → Secrets**.

The deployed app can read pre-computed `data/insights/` and the committed ChromaDB so it doesn't need to re-scrape on each load.

## Notes

- App Store scraping can be unreliable; the pipeline runs fine on Play Store + Reddit alone if it fails.
- Cited quotes shown in the app are real retrieved review text, not model-generated.
