"""OpenAPI 3.1 description of the HTTP API, served at /openapi.json."""

from . import __version__


def _ref(name):
    return {"$ref": f"#/components/schemas/{name}"}


def _json(schema, description="OK"):
    return {"description": description, "content": {"application/json": {"schema": schema}}}


def _param(name, where, schema, description, required=False):
    return {"name": name, "in": where, "required": required or where == "path",
            "schema": schema, "description": description}


USER_ID = _param("user_id", "path", {"type": "integer"}, "MovieLens user ID present in ratings.csv.")
MOVIE_ID = _param("movie_id", "path", {"type": "integer"}, "MovieLens movie ID.")
LANG = _param("lang", "query", {"type": "string", "enum": ["en", "vi"], "default": "en"},
              "Language of the human-readable `text`/`explanation` fields.")
ERRORS = {
    "400": _json(_ref("Error"), "Invalid parameter or body."),
    "404": _json(_ref("Error"), "Unknown user, movie or route."),
}

SPEC = {
    "openapi": "3.1.0",
    "info": {
        "title": "Movie Discovery Assistant API",
        "version": __version__,
        "description": (
            "Personalised movie discovery over the MovieLens subset. Every answer is grounded in the "
            "local rating, plot, genre and tag data; `evidence` objects expose the facts behind each "
            "sentence. `rank_score` orders results and is **not** a predicted star rating. Conversation "
            "context (what \"that\" refers to) is kept in memory per user ID."),
    },
    "servers": [{"url": "/"}],
    "tags": [
        {"name": "service", "description": "Health and API description."},
        {"name": "chat", "description": "Natural-language entry point used by the web UI and CLI."},
        {"name": "users", "description": "Profiles, recommendations and user-centred analysis."},
        {"name": "movies", "description": "Catalog lookup and plot search."},
    ],
    "paths": {
        "/health": {"get": {
            "tags": ["service"], "summary": "Liveness and dataset size",
            "responses": {"200": _json(_ref("Health"))}}},
        "/openapi.json": {"get": {
            "tags": ["service"], "summary": "This OpenAPI document",
            "responses": {"200": {"description": "OpenAPI 3.1 JSON"}}}},
        "/api/v1/chat": {"post": {
            "tags": ["chat"], "summary": "Answer one chat message",
            "description": ("Detects the intent (recommend, neighbor_opinion, explain, blind_spots, help, "
                            "clarify) and answers in the language of the message (English or Vietnamese)."),
            "requestBody": {"required": True, "content": {"application/json": {"schema": _ref("ChatRequest"),
                "example": {"user_id": 15, "message": "Movies like Aliens but no horror"}}}},
            "responses": {"200": _json(_ref("ChatResponse")), **ERRORS}}},
        "/api/v1/users": {"get": {
            "tags": ["users"], "summary": "List user IDs",
            "responses": {"200": _json({"type": "object", "properties": {
                "count": {"type": "integer"}, "user_ids": {"type": "array", "items": {"type": "integer"}}}})}}},
        "/api/v1/users/{user_id}": {"get": {
            "tags": ["users"], "summary": "Rating profile", "parameters": [USER_ID],
            "responses": {"200": _json(_ref("Profile")), **ERRORS}}},
        "/api/v1/users/{user_id}/context": {"delete": {
            "tags": ["users"], "summary": "Forget the user's last recommendations",
            "parameters": [USER_ID], "responses": {"204": {"description": "Context cleared."}, **ERRORS}}},
        "/api/v1/users/{user_id}/recommendations": {"get": {
            "tags": ["users"], "summary": "Personalised recommendations",
            "description": "Unwatched films ranked by item-item CF, user-neighbour CF and content, fused by rank.",
            "parameters": [
                USER_ID,
                _param("limit", "query", {"type": "integer", "minimum": 1, "maximum": 50, "default": 5}, "Number of results."),
                _param("q", "query", {"type": "string"}, "Free-text request, e.g. `dark psychological thriller`; genre words and negations are parsed."),
                _param("exclude_genres", "query", {"type": "string"}, "Comma-separated genres to exclude, e.g. `Horror,Animation`."),
                _param("require_genres", "query", {"type": "string"}, "Comma-separated genres that must all be present."),
                _param("seed_movie_ids", "query", {"type": "string"}, "Comma-separated movie IDs the user says they like right now."),
                _param("min_year", "query", {"type": "integer"}, "Earliest release year (inclusive)."),
                _param("max_year", "query", {"type": "integer"}, "Latest release year (inclusive)."),
                _param("max_ratings", "query", {"type": "integer", "minimum": 0}, "Only films with at most this many ratings (discovery)."),
                LANG],
            "responses": {"200": _json({"type": "object", "properties": {
                "user_id": {"type": "integer"},
                "recommendations": {"type": "array", "items": _ref("Recommendation")}}}), **ERRORS}}},
        "/api/v1/users/{user_id}/blind-spots": {"get": {
            "tags": ["users"], "summary": "Rarely rated genres with one film to try each",
            "parameters": [USER_ID, _param("limit", "query", {"type": "integer", "minimum": 1, "maximum": 19, "default": 5}, "Number of genres."), LANG],
            "responses": {"200": _json(_ref("BlindSpots")), **ERRORS}}},
        "/api/v1/users/{user_id}/movies/{movie_id}/neighbor-opinion": {"get": {
            "tags": ["users"], "summary": "What users with similar taste think of a movie",
            "parameters": [USER_ID, MOVIE_ID, LANG],
            "responses": {"200": _json(_ref("NeighborOpinion")), **ERRORS}}},
        "/api/v1/users/{user_id}/movies/{movie_id}/explanation": {"get": {
            "tags": ["users"], "summary": "Why this movie would (or would not) suit the user",
            "parameters": [USER_ID, MOVIE_ID, LANG],
            "responses": {"200": _json(_ref("Explanation")), **ERRORS}}},
        "/api/v1/movies/search": {"get": {
            "tags": ["movies"], "summary": "BM25 search over plot, title, genres and tags",
            "parameters": [_param("q", "query", {"type": "string"}, "Search text.", required=True),
                           _param("limit", "query", {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}, "Number of results.")],
            "responses": {"200": _json({"type": "object", "properties": {"results": {"type": "array", "items": {
                "allOf": [_ref("Movie"), {"type": "object", "properties": {"relevance": {"type": "number", "description": "Normalised to the best match (1.0)."}}}]}}}}),
                **ERRORS}}},
        "/api/v1/movies/{movie_id}": {"get": {
            "tags": ["movies"], "summary": "Movie details and rating statistics", "parameters": [MOVIE_ID],
            "responses": {"200": _json(_ref("MovieDetail")), **ERRORS}}},
    },
    "components": {"schemas": {
        "Error": {"type": "object", "required": ["error"], "properties": {"error": {
            "type": "object", "required": ["code", "message"],
            "properties": {"code": {"type": "string", "examples": ["bad_request", "not_found"]},
                           "message": {"type": "string"}}}}},
        "Health": {"type": "object", "properties": {
            "status": {"type": "string", "const": "ok"}, "version": {"type": "string"},
            "movies": {"type": "integer"}, "users": {"type": "integer"}, "ratings": {"type": "integer"}}},
        "Movie": {"type": "object", "properties": {
            "movie_id": {"type": "integer"}, "title": {"type": "string", "description": "MovieLens form, e.g. `Godfather, The`."},
            "year": {"type": ["integer", "null"]}, "genres": {"type": "array", "items": {"type": "string"}}}},
        "MovieDetail": {"allOf": [_ref("Movie"), {"type": "object", "properties": {
            "rating_count": {"type": "integer"}, "average_rating": {"type": ["number", "null"]},
            "smoothed_rating": {"type": "number", "description": "Bayesian mean with a prior of 8 ratings."},
            "tags": {"type": "array", "items": {"type": "string"}}, "plot": {"type": "string"}}}]},
        "Profile": {"type": "object", "properties": {
            "user_id": {"type": "integer"}, "rating_count": {"type": "integer"},
            "average_rating": {"type": "number"},
            "favorite_genres": {"type": "array", "items": {"type": "string"},
                                "description": "Genres most frequent among the user's 4+ ratings."}}},
        "Evidence": {"type": "object", "description": "Facts behind a recommendation.", "properties": {
            "shared_genres": {"type": "array", "items": {"type": "string"}},
            "seed_matches": {"type": "array", "items": {"type": "object"}},
            "content_signal": {"type": ["string", "null"], "enum": ["profile", "history", None]},
            "content_similarity": {"type": "number"},
            "query": {"type": ["string", "null"]},
            "query_terms": {"type": "array", "items": {"type": "object", "properties": {
                "concept": {"type": "string"}, "terms": {"type": "array", "items": {"type": "string"}}}}},
            "tone_caveat": {"type": "boolean"},
            "neighbors": {"type": ["object", "null"], "properties": {
                "count": {"type": "integer"}, "weighted_mean": {"type": "number"}}},
            "item_link": {"type": ["object", "null"], "properties": {
                "source_movie_id": {"type": "integer"}, "source_title": {"type": "string"},
                "support": {"type": "integer", "description": "Viewers who rated both films 3.5+."}}},
            "rating_count": {"type": "integer"}, "average_rating": {"type": ["number", "null"]},
            "low_confidence": {"type": "boolean"}}},
        "Recommendation": {"allOf": [_ref("Movie"), {"type": "object", "properties": {
            "rank": {"type": "integer"},
            "rank_score": {"type": "number", "description": "Ordering score only; not a star rating or probability."},
            "rating_count": {"type": "integer"}, "average_rating": {"type": ["number", "null"]},
            "explanation": {"type": "string"}, "evidence": _ref("Evidence")}}]},
        "ChatRequest": {"type": "object", "required": ["user_id", "message"], "properties": {
            "user_id": {"type": "integer"}, "message": {"type": "string", "minLength": 1, "maxLength": 2000}}},
        "ChatResponse": {"type": "object", "properties": {
            "intent": {"type": "string", "enum": ["recommend", "neighbor_opinion", "explain", "blind_spots",
                                                  "help", "clarify", "empty"]},
            "language": {"type": "string", "enum": ["en", "vi"]},
            "answer": {"type": "string"}, "headline": {"type": "string"},
            "recommendations": {"type": "array", "items": _ref("Recommendation")},
            "details": {"type": "object", "description": "Structured data for the intent (opinion, explanation, parsed constraints)."}}},
        "NeighborOpinion": {"type": "object", "properties": {
            "movie": _ref("Movie"), "neighbor_pool": {"type": "integer"},
            "raters": {"type": "array", "items": {"type": "object", "properties": {
                "user_id": {"type": "integer"}, "similarity": {"type": "number"},
                "common_movies": {"type": "integer"}, "rating": {"type": "number"}}}},
            "weighted_mean": {"type": ["number", "null"]}, "liked": {"type": "integer"}, "disliked": {"type": "integer"},
            "overall": {"type": "object", "properties": {"count": {"type": "integer"}, "average": {"type": ["number", "null"]}}},
            "your_rating": {"type": ["number", "null"]}, "text": {"type": "string"}}},
        "Explanation": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["recommended", "already_rated", "not_recommended"]},
            "movie": _ref("Movie"), "text": {"type": "string"}},
            "additionalProperties": True},
        "BlindSpots": {"type": "object", "properties": {
            "user_id": {"type": "integer"}, "text": {"type": "string"},
            "blind_spots": {"type": "array", "items": {"type": "object", "properties": {
                "genre": {"type": "string"}, "count": {"type": "integer"}, "average": {"type": "number"},
                "suggestion": {"oneOf": [_ref("Recommendation"), {"type": "null"}]}}}}}},
    }},
}
