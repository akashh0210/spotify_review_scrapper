# ARCHITECTURE.md — Spotify Discovery Review Engine

> Living document. Updated at the end of each build phase. Do not duplicate PROJECT.md (the spec) — this is the as-built map.

---

## Data-flow overview

```
Sources (Phase 1)
  Google Play Store  ──────────────────────────────┐
  Apple App Store RSS (5 storefronts)  ────────────┤
  Reddit via Pullpush.io (r/spotify, r/truespotify)┤──> data/raw/*.json
  Spotify Community forum (Khoros/BS4)  ───────────┘
                                                        |
                                               Phase 2: clean.py
                                                        |
                                               data/clean/reviews.parquet
                                               (unified schema, deduped)
                                                        |
                                               Phase 3: tag.py  (TODO)
                                                        |
                                               data/tagged/reviews_tagged.parquet
                                                        |
                                       Phase 4: embed.py  (TODO)
                                                        |
                                             ChromaDB (local, persistent)
                                          collection: spotify_reviews
                                                        |
                                       Phase 5: aggregate.py  (TODO)
                                                        |
                                        data/insights/summary.json
                                                        |
                                           Phase 5: rag.py  (TODO)
                                                        |
                                        data/insights/answers.json
                                                        |
                                       Phase 6: app.py  (TODO)
                                                        |
                                           Streamlit UI (two tabs)
```

---

## src/ file inventory

| File | Responsibility | Input | Output |
|---|---|---|---|
| `scrape_playstore.py` | Pulls 3,000 reviews from `com.spotify.music` via `google-play-scraper`. NEWEST + MOST_RELEVANT batches, deduped by `reviewId`. | Google Play API (no auth) | `data/raw/playstore.json` |
| `scrape_appstore.py` | Loops iTunes RSS customer-review JSON feed, pages 1-10 x storefronts (us, gb, ca, au, in). Parses `feed.entry`, dedupes by review id. Tenacity backoff on 429. | `itunes.apple.com/rss` (no auth) | `data/raw/appstore.json` |
| `scrape_reddit.py` | Searches r/spotify + r/truespotify for 10 discovery terms via Pullpush.io (no OAuth). Submission title + selftext only; 2 s delay between searches + tenacity retry. | `api.pullpush.io` (no auth) | `data/raw/reddit.json` |
| `scrape_forum.py` | Scrapes community.spotify.com (Khoros platform) with requests + BeautifulSoup. Targets Discovery & Promo, Closed Ideas, Content Questions boards. Filters to discovery-relevant posts. Captures title, body, kudos count. | `community.spotify.com` (no auth) | `data/raw/forum.json` |
| `clean.py` | Loads all four raw sources, applies text cleaning (HTML entities, zero-width chars, Reddit markdown/quote artifacts), drops empties / <15-char / no-alpha rows, dedupes by normalized-text fingerprint. Outputs unified 7-column schema. | `data/raw/*.json` | `data/clean/reviews.parquet` |
| `retag_errors.py` | Re-tags all `tag_error=True` rows using Groq llama-3.1-8b-instant (same model as tag.py). Identical prompt + schema + parse/validate path. Overwrites only error rows; good rows untouched. Checkpoint-aware: skips rows already merged into good_df from prior Gemini retag pass. Checkpoints every 50 rows. | `reviews_tagged.parquet` | `reviews_tagged.parquet` (updated in-place) |
| `test_gemini.py` | One-call smoke test to confirm GEMINI_API_KEY + google-genai SDK work before a full run. | — | stdout |
| `tag.py` | Groq 8B batched tagging (primary) with Gemini 2.0 Flash daily-quota fallback. Strict JSON schema: themes, sentiment, segment, discovery_related, one_line, language, tag_error, **tagged_by** (model provenance). Depth-counting JSON parser avoids trailing-bracket corruption. Retry once on parse failure; row-by-row fallback on second failure; FALLBACK tag on individual row failure. Sources tagged: playstore (all), appstore (partial 1370/2340), reddit (all), forum (all). | `data/clean/reviews.parquet` | `data/tagged/reviews_tagged.parquet` |
| `embed.py` | sentence-transformers `all-MiniLM-L6-v2` (local) → ChromaDB persistent collection `spotify_reviews`. | `data/tagged/reviews_tagged.parquet` | ChromaDB at `./chroma_db/` |
| `aggregate.py` | Count themes, sentiment, segment splits. Pull top quoted examples per theme. | `data/tagged/reviews_tagged.parquet` | `data/insights/summary.json` |
| `rag.py` | Groq 70B answering the six discovery questions over ChromaDB-retrieved evidence. Cited verbatim quotes only. | ChromaDB + `data/insights/summary.json` | `data/insights/answers.json` |

---

## Stack — as actually wired

| Concern | Tool / Model | Notes |
|---|---|---|
| Play Store scraping | `google-play-scraper` | App ID `com.spotify.music` |
| App Store scraping | iTunes RSS JSON feed (requests) | App ID `324684580` |
| Reddit scraping | Pullpush.io REST API (requests) | No OAuth; replaces deprecated Reddit public JSON |
| Forum scraping | requests + BeautifulSoup4 | community.spotify.com (Khoros) |
| Rate limiting | `tenacity` exponential backoff | All network scrapers; 2–4 s base delay |
| Tagging LLM (primary) | Groq `llama-3.1-8b-instant` | Per-review JSON tagging; temp=0; per-minute 429 → tenacity backoff |
| Tagging LLM (fallback) | Google Gemini `gemini-2.5-flash-lite` (+ `gemini-2.5-flash` if lite errors) | Activated only on Groq TPD daily-quota 429; identical prompt + schema. Uses **google-genai SDK** with AQ.\* auth keys. `response_mime_type: application/json` enforces pure JSON. Note: `gemini-2.0-*` models have zero free-tier quota on this project's key. |
| Provenance field | `tagged_by` column | Records model used per row (`groq-llama-3.1-8b` or `gemini-2.0-flash`) for cross-provider audit |
| Synthesis LLM | Groq `llama-3.3-70b-versatile` | Six-question answers + RAG |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Local CPU inference; Groq has no embedding endpoint |
| Vector store | ChromaDB (persistent, local) | Collection: `spotify_reviews` |
| Data format | Parquet via `pandas` + `pyarrow` | Intermediate pipeline stages |
| UI | Streamlit | Two tabs: Insights Dashboard + Ask the Reviews |
| Charts | Plotly | Theme frequency, segment breakdown |
| Secrets | `python-dotenv` (.env) | Keys never committed |
| Deploy target | Streamlit Community Cloud | Fallback: Hugging Face Spaces |

---

## Status by phase

| Phase | Description | Status | Notes |
|---|---|---|---|
| 1 | Scrape | **Done** | Play: 3,169 \| App Store RSS: 2,500 \| Reddit (Pullpush): 1,735 \| Forum: 27 \| **Total: 7,431** |
| 2 | Clean / dedupe / normalize | **Done** | 7,431 raw → 6,778 clean (653 dropped: 584 too-short, 57 duplicates, 12 no-alpha). Non-English: 17 rows (0.3%) — kept, tagged in Phase 3. Schema: `id\|source\|text\|rating\|date\|score\|url`. |
| 3 | Tag (Groq 8B) | **Done** | Groq llama-3.1-8b-instant, temp=0, batch=10, checkpoint every 50 rows. Coverage: playstore (all 2,730) + appstore (1,370/2,340 — partial by design, sufficient signal) + reddit (all 1,681) + forum (all 27). Appstore remainder intentionally skipped. Output: `reviews_tagged.parquet`. |
| 4 | Embed (sentence-transformers → Chroma) | Pending | — |
| 5 | Aggregate + six-question answers | Pending | — |
| 6 | Streamlit app | Pending | — |
| 7 | Deploy (public URL) | Pending | — |
