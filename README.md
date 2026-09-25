# Movie Discovery Assistant

A conversational assistant that helps a MovieLens user find films by investigating the supplied dataset on their behalf. It personalises from the user's rating history, combines several data signals (taste neighbours, co-liked films, plot text, genres, tags) and explains every answer with evidence from the data. It answers in English or Vietnamese, following the language of the question.

The design choices, the evaluation and the failure analysis are in **[REPORT.md](REPORT.md)**.

## Quick start

Requires Python 3.10+. There are no third-party dependencies, no API keys and no model downloads. Everything runs offline on the CSVs in `data/`.

```bash
python -m movie_assistant --user 15          # interactive CLI (:help, :user 30, :quit)
python -m movie_assistant.web                # web UI + API at http://127.0.0.1:8765
```

With Docker:

```bash
docker compose up --build                    # http://localhost:8765, API docs at /docs
# or
docker build -t movie-assistant:1.0.0 . && docker run --rm -p 8765:8765 movie-assistant:1.0.0
```

The image contains only the `movie_assistant` package and the dataset. It runs as a non-root user and has a `HEALTHCHECK` on `/health`.

## What you can ask

| Request | What the assistant does |
|---|---|
| `What should I watch tonight?` | Ranks unseen films from the user's history (item-item CF + user-neighbour CF + content profile) |
| `I want a dark psychological thriller with a twist` | BM25 over plots/tags with concept expansion, filtered by genre, blended with personal ranking |
| `What do people with similar taste to mine think about Pulp Fiction?` | Finds the 20 closest taste neighbours, aggregates their ratings, compares with all viewers |
| `Why do you think I'd like that?` | Explains the last recommendation for this user from its stored evidence |
| `I liked Toy Story but I'm tired of animated movies` | Uses Toy Story as a seed for this request and excludes Animation |
| `Movies like Aliens but no horror` | Seed + genre exclusion |
| `What's my blind spot?` | Rarely rated genres, each with one personalised film to try |
| `phim hành động không kinh dị`, `tối nay xem gì?` | The same features in Vietnamese |

Example (user 1, from `artifacts/demo.json`):

```text
You: What do people with similar taste to mine think about Pulp Fiction?
Assistant: Among your 20 closest taste neighbours, 9 rated Pulp Fiction: weighted average 4.22/5
(7 gave 4+, 0 gave 2.5 or less). Across all viewers: 307 ratings, average 4.20/5, so your taste
group rates it about the same. Closest raters: user 178 (similarity 0.39, 24 films in common)
gave 4.5/5, ... You rated it 3.0/5 yourself.
```

## HTTP API

The web server is also a JSON API: chat, recommendations with filters, taste-neighbour opinion, explanations, blind spots, movie search and lookup. There is a health check, errors are structured, and the service is configured through environment variables.

- Reference: **[docs/API.md](docs/API.md)**
- Interactive: `http://127.0.0.1:8765/docs`
- OpenAPI 3.1: `http://127.0.0.1:8765/openapi.json`

```bash
curl -s 'localhost:8765/api/v1/users/15/recommendations?limit=3&exclude_genres=Horror'
curl -s localhost:8765/api/v1/chat -H 'Content-Type: application/json' \
     -d '{"user_id": 1, "message": "What is my blind spot?"}'
```

## Evaluation and reproducible outputs

| Command | Output | Runtime |
|---|---|---|
| `python -m unittest discover -s tests -v` | 59 unit, dialogue and API tests | < 1 s |
| `python scripts/verify_dataset.py` | CSV schema check | < 5 s |
| `python scripts/capture_demo.py` | `artifacts/demo.json`: 28 real exchanges for users 1, 15, 30 | ~ 10 s |
| `python scripts/api_smoke.py` | `artifacts/api_smoke.json`: 10 real HTTP request/response pairs | ~ 5 s |
| `python scripts/evaluate_queries.py` | `artifacts/query_probes.json`: 19 intent probes | < 1 s |
| `python scripts/evaluate_v2.py` | `artifacts/evaluation_v2.json`: temporal hold-out, cohorts, bootstrap | ~ 3 min |
| `python scripts/evaluate_recommender.py` | `artifacts/evaluation.json`: V1 baselines (popularity, hybrid) | ~ 3 min |

Headline result on a per-user temporal hold-out (303 users, full-catalog ranking, K = 10): **Recall@10 = 0.116 and NDCG@10 = 0.059**. For comparison, the previous ranking scores 0.102 / 0.052 and a Bayesian popularity baseline 0.063 / 0.035. The improvement over the previous ranking is **not** statistically significant (the paired bootstrap interval includes 0), and films with five or fewer training ratings are never recovered. See [REPORT.md](REPORT.md#evaluation).

## Project layout

```text
movie_assistant/
  storage.py          CSV loading into typed records
  engine.py           ranking (user CF, item CF, content, rank fusion), neighbour opinion, explanations, blind spots
  item_similarity.py  activity-weighted item-item cosine with support shrinkage
  retrieval.py        field-boosted BM25 with concept groups
  query.py            bilingual query planner: genres, negations, concepts
  dialogue.py         intent routing shared by CLI and API
  messages.py         English/Vietnamese wording; evidence -> sentences
  web.py, api_spec.py HTTP API, OpenAPI document, static UI
  cli.py              interactive terminal client
scripts/              evaluation, demo capture, API smoke test, dataset check
tests/                unit, dialogue and HTTP API tests
artifacts/            saved outputs referenced by the report
docs/API.md           API reference
Dockerfile, compose.yaml
```

## Known limits

Intent detection and query understanding are transparent rules plus lexical retrieval, not a language model. Paraphrases outside the rules, mood and plot twists are only approximated. Films with very few ratings rarely reach the default top 10. The catalog ends in 2014. Details and concrete failure cases are in [REPORT.md](REPORT.md#failure-analysis).

---

## The Problem

You have a dataset of movies with plot summaries, user ratings, and tags. Your task:

**Build an AI assistant that helps users discover movies by investigating the dataset on their behalf.**

This is not a search engine — the assistant should reason about what to look up, combine multiple data signals, and explain its thinking. When a user asks "why would I like that?", the assistant should be able to dig into their rating history, find patterns, and give a grounded answer.

### Requirements

1. The user identifies themselves (e.g., by user ID), and the assistant uses their rating history to personalize recommendations
2. The assistant can answer questions that require combining multiple pieces of information — e.g., "what do people with similar taste to mine think of Inception?" requires finding similar users, checking their Inception ratings, and synthesizing
3. The assistant explains its reasoning using historical data — not just LLM knowledge
4. Evaluate your system's recommendation quality with evidence — show where it works and where it fails (you might use metrics, qualitative examples, or both — explain why you chose what you chose)

How you get there is up to you.

## Dataset

Located in `data/ml-latest-small-filtered/`:

| File | Size | Description |
|------|------|-------------|
| `movies_with_plots.csv` | 16MB | 5,135 movies — `movieId`, `title`, `year`, `genres`, `plot` (100–5,000+ chars, avg ~3,200) |
| `ratings.csv` | 1.7MB | 74,064 ratings from 610 users — `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` |
| `tags.csv` | 74KB | 2,440 user-generated tags — sparse, most movies have none |
| `links.csv` | 98KB | External links (IMDb, TMDB) |
| `movies.csv` | 228KB | Basic movie info without plots |

**Data notes:**
- Movies span 1903–2014. This is a filtered subset of MovieLens — some well-known movies (e.g., The Matrix, Ocean's Eleven) may be absent due to missing plot data in the source.
- ~51% of movies have fewer than 5 ratings.
- Tags are very sparse (2,440 tags across 5,135 movies). Don't build your approach around tags alone.
- Users have ~121 ratings on average — relatively dense on the user side. The sparsity is on the movie side.

### What's in the data

The dataset gives you several signals to work with:

- **Rating patterns:** 610 users × 5,135 movies. Users who rate similar movies similarly have similar taste — this is the basis of collaborative filtering. You can find "users like me" and see what they enjoyed.
- **Plot summaries:** Full text descriptions (avg ~3,200 chars). Useful for content-based search — finding movies that match a description like "dark thriller with a twist."
- **Genres:** 19 genres per movie (pipe-separated). Useful for filtering, profiling user preferences, and finding blind spots.
- **Tags:** User-generated labels like "twist ending", "atmospheric", "dark comedy". Sparse but high-signal where they exist.

### Suggested users for testing

These users have different profiles — useful for testing personalization:

| User ID | Ratings | Avg | Profile |
|---------|---------|-----|---------|
| 1 | 190 | 4.33 | Action/comedy fan — likes Terminator, Blues Brothers, Full Metal Jacket |
| 15 | 85 | 3.55 | Sci-fi oriented — likes Aliens, Star Wars, Back to the Future |
| 30 | 18 | 4.61 | Sparse history — likes Braveheart, Inception, Shawshank Redemption |

### Sample Queries

Use these to sanity-check your system during development:

- "What should I watch tonight?" *(requires knowing the user's taste)*
- "I want a dark psychological thriller with a twist" *(content search + quality filter)*
- "What do people with similar taste to mine think about Pulp Fiction?" *(find similar users + aggregate their ratings)*
- "Why do you think I'd like that?" *(explain using user's history + movie data)*
- "I liked Toy Story but I'm tired of animated movies — what else?" *(use history + apply constraints)*
- "What's my blind spot? What genres am I missing?" *(analyze user's rating patterns)*

## Deliverables

### 1. Code
- Working implementation with setup instructions
- Include reproducible output: sample conversations, evaluation results, or screenshots that demonstrate the system working
- If your solution uses external APIs (e.g., OpenAI), document this and include example outputs so we can evaluate without running it

### 2. Report
Follow the template in `REPORT_TEMPLATE.md`. **This is as important as the code.**

We weight the report equally with the code. A mediocre system with excellent analysis beats a good system with a shallow report.

### 3. Interview
You will:
- Demo your system live
- Walk us through your report
- Discuss your decisions and tradeoffs

## Time

3-4 hours. Rough guide: ~2 hours building, ~1 hour on the report and evaluation, ~30 min cleanup.

Tip: keep notes on your decisions as you go — it makes the report much easier to write.

AI tools are welcome — be ready to discuss your work in depth.

## What We Care About

- How you break down the problem
- Why you made your choices
- Honest assessment of your solution — especially where it fails
- Code someone else can read

## What We Don't Care About

- State-of-the-art performance
- Complex infrastructure
- Perfect solutions
- Exhaustive hyperparameter tuning

We're more interested in your thinking than your metrics.
