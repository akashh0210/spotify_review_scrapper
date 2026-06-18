# PROJECT.md — Spotify Discovery Review Engine

> This is the build spec. Read this fully before writing code. Build **phase by phase** (see "Build Phases"), run each phase against real data, and confirm output before moving on. Do not scaffold all phases at once.

## 1. What this is

An AI-powered review-analysis engine that ingests Spotify user reviews at scale and surfaces *why users struggle to discover new music and keep repeat-listening*. It is Part 1 of a Growth PM capstone. Its only job is to produce **one defensible insight** about music discovery that downstream user interviews will validate. It is not a general-purpose review tool — keep it aimed at discovery + repetition.

Deliverable: a **deployed, public** web app + a single slide explaining how it works.

## 2. Focus (do not let this sprawl)

Analyze Spotify reviews to answer these six questions:

1. Why do users struggle to discover new music?
2. What are the most common frustrations with recommendations?
3. What listening behaviors are users trying to achieve?
4. What causes users to repeatedly listen to the same content?
5. Which user segments experience different discovery challenges?
6. What unmet needs emerge consistently across reviews?

Everything the engine does maps back to these six. Bugs / billing / UI / ads are tagged and **excluded** from the discovery analysis (kept only as a "non-discovery" bucket for context).

## 3. Stack (fixed — free infra only)

- **Language:** Python 3.11+
- **LLM (tagging):** Groq `llama-3.1-8b-instant` — fast, cheap, used for high-volume per-review tagging
- **LLM (RAG answers):** Groq `llama-3.3-70b-versatile` — used for synthesis / answering the six questions
- **Embeddings:** `sentence-transformers` `all-MiniLM-L6-v2`, run **locally** (Groq has NO embedding endpoint — do not attempt to embed via Groq)
- **Vector store:** ChromaDB (persistent, local)
- **UI:** Streamlit
- **Charts:** Plotly
- **Deploy:** Streamlit Community Cloud (primary) or Hugging Face Spaces (fallback)
- **Rate-limit handling:** `tenacity` retry with exponential backoff on all Groq calls; batch tagging in chunks

## 4. Data sources

| Source | Tool | Landed count | Notes |
|---|---|---|---|
| Google Play reviews | `google-play-scraper` | **3,169** | App ID `com.spotify.music`. No auth. NEWEST + MOST_RELEVANT batches, deduped. Primary source. |
| App Store reviews | iTunes RSS JSON feed (`requests`) | **2,500** | App ID `324684580`. Pages 1-10 x storefronts us/gb/ca/au/in. No auth. Replaced flaky `app-store-scraper` package with direct RSS API call. |
| Reddit | Pullpush.io REST API (`requests`) | **1,735** | Subreddits `spotify`, `truespotify`. No OAuth required. Reddit's public `.json` endpoints return 403 as of 2023; Pullpush.io is the community-run Pushshift replacement that indexes all public Reddit content without auth. Submission title + selftext only (no per-post comment API calls). |
| Spotify Community forum | `requests` + BeautifulSoup4 | **27** | community.spotify.com (Khoros/Lithium platform). Discovery & Promo board, Closed Ideas, Content Questions boards. Filtered to discovery-relevant posts by keyword. Captures title, body, kudos count. |

> **Note on paid social (X/Twitter):** Out of scope for a free build. X's API requires paid access ($100/month minimum). Reddit + Community Forum serve as the social-conversation proxy covering the same qualitative signal at zero cost.

**Reddit/forum search terms:** `discover weekly`, `recommendations`, `same songs`, `repetitive`, `algorithm`, `new music`, `daily mix`, `autoplay`, `stuck`, `radio`.

Target volume: a few thousand items total is plenty. Do not over-scrape.

## 5. Pipeline (each stage writes to disk so it's resumable)

1. **Scrape** → `data/raw/{playstore,appstore,reddit,forum}.json`
2. **Clean / dedupe / normalize** → `data/clean/reviews.parquet` (unified schema: `id, source, text, rating, date, raw_meta`)
3. **Tag** (Groq 8B, batched, JSON output) → `data/tagged/reviews_tagged.parquet`
4. **Embed** (sentence-transformers) → ChromaDB persistent collection `spotify_reviews`
5. **Aggregate** → `data/insights/summary.json` (theme counts, sentiment, segment splits, top quotes per theme)
6. **Answer the six questions** (Groq 70B over retrieved evidence) → `data/insights/answers.json`

## 6. Tagging schema (Groq 8B must return strict JSON per review)

```json
{
  "themes": ["recommendation_repetition", "discovery_friction"],
  "sentiment": "negative",
  "segment": "power_user",
  "discovery_related": true,
  "one_line": "User says Discover Weekly keeps recycling songs they already know."
}
```

**Allowed `themes`** (closed list — do not invent new ones):
`recommendation_repetition`, `discovery_friction`, `generic_recommendations`, `discover_weekly_dailymix`, `autoplay_radio_loop`, `no_control_or_intent`, `filter_bubble`, `wants_new_but_safe`, `positive_discovery`, `non_discovery`

**`sentiment`:** `positive` | `neutral` | `negative`
**`segment`:** `casual` | `power_user` | `genre_explorer` | `mood_context_listener` | `podcast_listener` | `unknown`
Temperature 0 for tagging. Validate/parse JSON; on failure, retry once then skip the row (log it).

## 7. The app (`app.py`) — two tabs

**Tab 1 — Insights Dashboard**
- Headline metrics: total reviews analyzed, % discovery-related, sentiment split
- Plotly bar: theme frequency (discovery themes only)
- Plotly: segment × top-frustration breakdown
- The **six questions**, each rendered with its Groq-generated answer + 2–3 cited verbatim quotes (source + rating shown)

**Tab 2 — Ask the Reviews (RAG)**
- Free-text box → retrieve top-k from Chroma → Groq 70B answers **only from retrieved reviews**, with quoted evidence and source tags
- Pre-loaded example questions = the six above (one click each)

The RAG box answering the rubric's own questions *is* the demo. Make it clean and fast.

## 8. Guardrails

- Free infrastructure only. No paid APIs beyond the Groq free tier.
- Never commit secrets. All keys via `.env` locally / platform secrets on deploy.
- Keep the scope on discovery + repetition. Resist building a generic analytics tool.
- Every Groq call wrapped in retry/backoff. Batch tagging; don't fire one call per review without throttling.
- Cited quotes in the UI must be real retrieved text, never model-fabricated.

## 9. File structure

```
spotify-discovery-engine/
├── PROJECT.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── data/
│   ├── raw/        (gitignored)
│   ├── clean/      (gitignored)
│   ├── tagged/     (gitignored)
│   └── insights/   (summary.json + answers.json — small, can be committed)
└── src/
    ├── scrape_playstore.py
    ├── scrape_appstore.py
    ├── scrape_reddit.py
    ├── clean.py
    ├── tag.py
    ├── embed.py
    ├── aggregate.py
    └── rag.py
```

## 10. Build Phases (do these in order, confirm each before next)

- **Phase 1 — Scrape.** Build the three scrapers. Run them. Confirm real review counts on disk before anything else.
- **Phase 2 — Clean.** Dedupe + unify schema → `reviews.parquet`. Print row count + sample.
- **Phase 3 — Tag.** Groq 8B batched tagging with the schema in §6. Validate JSON. Show theme distribution.
- **Phase 4 — Embed.** sentence-transformers → ChromaDB. Confirm a test similarity query returns sane results.
- **Phase 5 — Aggregate + Answer.** Build `summary.json` and the six answers with cited quotes.
- **Phase 6 — App.** Streamlit dashboard + RAG tab.
- **Phase 7 — Deploy.** Public link on Streamlit Cloud / HF Spaces.

## 11. Definition of done

- Public URL loads the app with no key exposed.
- Dashboard shows real theme/segment data from real reviews.
- The six questions are answered with real cited quotes.
- RAG box answers a free-typed discovery question from retrieved evidence.
- One clear insight stands out strongly enough to point the user interviews at it.
