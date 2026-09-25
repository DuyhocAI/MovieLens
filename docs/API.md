# HTTP API reference

The backend is a single-process HTTP service built on the Python standard library. It serves the web UI at `/`, an interactive reference at `/docs` (Swagger UI, loaded from a CDN) and the machine-readable OpenAPI 3.1 document at `/openapi.json`. The OpenAPI document is generated from [`movie_assistant/api_spec.py`](../movie_assistant/api_spec.py) and is the source of truth; this page summarises it.

## Running

| Setting | Flag | Environment variable | Default |
|---|---|---|---|
| Bind address | `--host` | `MOVIE_ASSISTANT_HOST` | `127.0.0.1` (`0.0.0.0` in the Docker image) |
| Port | `--port` | `PORT` | `8765` |
| Dataset directory | `--data-dir` | `MOVIE_ASSISTANT_DATA_DIR` | `data/ml-latest-small-filtered` |
| Browser origins allowed by CORS | `--cors-origins` | `MOVIE_ASSISTANT_CORS_ORIGINS` | none (same-origin only) |

```bash
python -m movie_assistant.web                      # local
docker compose up --build                          # container, http://localhost:8765
```

Startup loads the CSVs and builds the indexes in about 3 seconds. `SIGTERM` stops the server cleanly.

## Conventions

- All API routes are under `/api/v1` and return `application/json; charset=utf-8`.
- IDs are MovieLens `userId` / `movieId`.
- `lang=en|vi` selects the language of human-readable fields (`text`, `explanation`). Structured fields are language-neutral. `POST /api/v1/chat` detects the language from the message instead.
- `rank_score` only orders results. It is **not** a predicted star rating or a probability.
- Every recommendation carries an `evidence` object with the facts its explanation is built from: shared genres, taste-neighbour count and weighted mean, item-link source and support, matched plot/tag terms, rating count, and a low-confidence flag.
- Errors always look like this:

```json
{"error": {"code": "not_found", "message": "User 9999 is not present in the rating data."}}
```

| Status | `code` | When |
|---|---|---|
| 400 | `bad_request` | Invalid parameter, unknown genre, malformed JSON, empty message |
| 404 | `not_found` | Unknown user, movie or route |
| 405 | `method_not_allowed` | Known path, wrong method |
| 413 | `payload_too_large` | Body over 16 KB |
| 500 | `internal_error` | Unexpected error (details are logged, not returned) |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check with dataset counts (used by the Docker `HEALTHCHECK`) |
| GET | `/openapi.json` | OpenAPI 3.1 document |
| GET | `/docs` | Interactive reference |
| POST | `/api/v1/chat` | Natural-language entry point (intent detection + answer) |
| GET | `/api/v1/users` | All user IDs |
| GET | `/api/v1/users/{user_id}` | Rating count, average, favourite genres |
| DELETE | `/api/v1/users/{user_id}/context` | Forget the user's last recommendations |
| GET | `/api/v1/users/{user_id}/recommendations` | Personalised ranking with filters |
| GET | `/api/v1/users/{user_id}/blind-spots` | Rarely rated genres, each with one film to try |
| GET | `/api/v1/users/{user_id}/movies/{movie_id}/neighbor-opinion` | What users with similar taste think of a movie |
| GET | `/api/v1/users/{user_id}/movies/{movie_id}/explanation` | Why a movie would (or would not) suit the user |
| GET | `/api/v1/movies/search` | BM25 search over plot, title, genres and tags |
| GET | `/api/v1/movies/{movie_id}` | Movie details, rating statistics, tags and plot |

### `POST /api/v1/chat`

```bash
curl -s localhost:8765/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"user_id": 15, "message": "I don'\''t want horror, give me sci-fi"}'
```

```json
{
  "intent": "recommend",
  "language": "en",
  "headline": "Picks that match your request:",
  "answer": "Picks that match your request:\n1. Men in Black (a.k.a. MIB) (1997) — Action, Comedy, Sci-Fi. You rated other ...",
  "recommendations": [
    {"rank": 1, "movie_id": 1580, "title": "Men in Black (a.k.a. MIB)", "year": 1997,
     "genres": ["Action", "Comedy", "Sci-Fi"], "rank_score": 4.5556, "rating_count": 165, "average_rating": 3.488,
     "explanation": "...", "evidence": {"neighbors": {"count": 19, "weighted_mean": 3.425}, "...": "..."}}
  ],
  "details": {"seed_movie_id": null, "query": "I don't want horror, give me sci-fi", "excluded_genres": ["Horror"],
              "required_genres": ["Sci-Fi"], "min_year": null, "max_year": null, "discovery": false}
}
```

`intent` is one of `recommend`, `neighbor_opinion`, `explain`, `blind_spots`, `help`, `clarify` or `empty`. For `neighbor_opinion`, `explain` and `blind_spots`, `details` holds the same structured data as the dedicated endpoints below. Follow-ups such as "Why would I like that?" refer to the latest recommendations **for the same user ID**.

### `GET /api/v1/users/{user_id}/recommendations`

| Query parameter | Type | Meaning |
|---|---|---|
| `limit` | int 1–50, default 5 | Number of results |
| `q` | string | Free-text request. Genre words, negations (`no horror`, `don't want horror`) and 12 concept groups are parsed |
| `exclude_genres` | CSV | Genres to exclude, e.g. `Horror,Animation` (case-insensitive; unknown names return 400) |
| `require_genres` | CSV | Genres that must all be present |
| `seed_movie_ids` | CSV of ints | Films the user says they like right now |
| `min_year`, `max_year` | int | Inclusive release-year bounds |
| `max_ratings` | int ≥ 0 | Only films with at most this many ratings (discovery) |
| `lang` | `en`/`vi` | Language of `explanation` |

```bash
curl -s 'localhost:8765/api/v1/users/15/recommendations?limit=3&q=space&exclude_genres=Horror'
```

### `GET /api/v1/users/{user_id}/movies/{movie_id}/neighbor-opinion`

Uses the user's 20 closest taste neighbours: mean-centred Pearson similarity on co-rated films, shrunk by `overlap / (overlap + 10)`, requiring at least 3 co-rated films.

```json
{
  "status": "ok", "movie": {"movie_id": 296, "title": "Pulp Fiction", "year": 1994, "genres": ["Comedy", "Crime", "Drama", "Thriller"]},
  "neighbor_pool": 20,
  "raters": [{"user_id": 178, "similarity": 0.385, "common_movies": 24, "rating": 4.5}, "..."],
  "weighted_mean": 4.221, "liked": 7, "disliked": 0,
  "overall": {"count": 307, "average": 4.197}, "your_rating": 3.0,
  "text": "Among your 20 closest taste neighbours, 9 rated Pulp Fiction: weighted average 4.22/5 ..."
}
```

### `GET /api/v1/users/{user_id}/movies/{movie_id}/explanation`

`status` is `recommended` (the movie is in the user's latest recommendations; `evidence` is included), `already_rated` (`your_rating` is included) or `not_recommended` (shared genres, plot similarity to the user's liked films, and rating statistics).

### `GET /api/v1/users/{user_id}/blind-spots`

Genres with fewer than `max(3, 5% of the user's ratings)` ratings. `IMAX` is ignored because it is a format, not a genre. The first three genres each get one personalised suggestion, and those suggestions become the context for "why would I like that?".

## Operational notes

- **State.** The only per-user state is the latest recommendation list, which lets follow-ups refer back. It is held in memory. If you run several replicas, use sticky sessions or accept that follow-ups may miss.
- **Concurrency.** Requests are served on threads. Calls into the engine are serialised by a lock because its caches are not thread-safe. Typical requests take well under a second on the full dataset, and the first request for a user also builds that user's item-similarity cache.
- **Security.** There is no authentication, because the dataset is public and read-only. Put the service behind a gateway before exposing it. HTML pages send a strict Content-Security-Policy, and `/docs` additionally allows `cdn.jsdelivr.net`.
- **Outbound traffic.** None. The API only reads the bundled CSV files.
