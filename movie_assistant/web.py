"""HTTP API and web UI for the movie discovery assistant (standard library only).

Run locally with `python -m movie_assistant.web`; see docs/API.md or /docs for the API.
Host and port come from --host/--port or the MOVIE_ASSISTANT_HOST/PORT environment variables.
"""

import argparse
import json
import mimetypes
import os
import re
import signal
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from . import __version__, messages
from .api_spec import SPEC
from .dialogue import respond
from .engine import MovieAssistant
from .models import Recommendation
from .storage import load_dataset

STATIC_DIR = Path(__file__).with_name("static")
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/static/app.js": "app.js",
                "/static/style.css": "style.css", "/docs": "docs.html", "/static/docs.js": "docs.js"}
PAGE_CSP = ("default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "base-uri 'none'; frame-ancestors 'none'")
# The API reference page loads Swagger UI from a CDN.
DOCS_CSP = ("default-src 'self'; style-src 'self' https://cdn.jsdelivr.net; script-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self'; img-src 'self' data: https://cdn.jsdelivr.net; base-uri 'none'; frame-ancestors 'none'")
MAX_BODY_BYTES = 16_384
MAX_MESSAGE_CHARS = 2_000


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class LocalHTTPServer(ThreadingHTTPServer):
    """Refuse to share a port with a stale process (matters on Windows)."""

    daemon_threads = True
    allow_reuse_address = not hasattr(socket, "SO_EXCLUSIVEADDRUSE")
    allow_reuse_port = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def recommendation_json(rank: int, rec: Recommendation, lang: str) -> dict:
    return {
        "rank": rank,
        "movie_id": rec.movie.movie_id,
        "title": rec.movie.title,
        "year": rec.movie.year,
        "genres": list(rec.movie.genres),
        "rank_score": round(rec.score, 4),
        "rating_count": rec.rating_count,
        "average_rating": round(rec.average_rating, 3) if rec.rating_count else None,
        "explanation": messages.recommendation_text(rec.movie.title, rec.evidence, lang),
        "evidence": rec.evidence,
    }


def _int(values: dict, name: str, default=None, low=None, high=None):
    raw = values.get(name, [None])[0]
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(400, "bad_request", f"'{name}' must be an integer.") from None
    if (low is not None and value < low) or (high is not None and value > high):
        raise ApiError(400, "bad_request", f"'{name}' must be between {low} and {high}.")
    return value


def _csv(values: dict, name: str) -> list[str]:
    raw = values.get(name, [""])[0]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _lang(values: dict) -> str:
    lang = values.get("lang", ["en"])[0] or "en"
    if lang not in {"en", "vi"}:
        raise ApiError(400, "bad_request", "'lang' must be 'en' or 'vi'.")
    return lang


class MovieWebHandler(BaseHTTPRequestHandler):
    assistant: MovieAssistant
    assistant_lock: Lock
    cors_origins: frozenset = frozenset()
    access_log = True
    server_version = f"MovieAssistant/{__version__}"

    # ----- plumbing -------------------------------------------------------
    def log_request(self, code="-", size="-"):
        if self.access_log:
            super().log_request(code, size)

    def _send(self, status: int, body: bytes, content_type: str, csp: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if csp:
            self.send_header("Content-Security-Policy", csp)
        origin = self.headers.get("Origin")
        if origin and ("*" in self.cors_origins or origin in self.cors_origins):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data):
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, code: str, message: str):
        self._json(status, {"error": {"code": code, "message": message}})

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(400, "bad_request", "Invalid Content-Length.") from None
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "payload_too_large", f"Body must be at most {MAX_BODY_BYTES} bytes.")
        if length < 1:
            raise ApiError(400, "bad_request", "A JSON body is required.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "bad_request", "Body must be valid JSON.") from None
        if not isinstance(payload, dict):
            raise ApiError(400, "bad_request", "Body must be a JSON object.")
        return payload

    def _user(self, raw) -> int:
        try:
            user_id = int(raw)
        except (TypeError, ValueError):
            raise ApiError(400, "bad_request", "'user_id' must be an integer.") from None
        if user_id not in self.assistant.user_ratings:
            raise ApiError(404, "not_found", f"User {user_id} is not present in the rating data.")
        return user_id

    def _movie(self, raw) -> int:
        movie_id = int(raw)
        if movie_id not in self.assistant.dataset.movies:
            raise ApiError(404, "not_found", f"Movie {movie_id} is not in the catalog.")
        return movie_id

    def _genres(self, names: list[str]) -> set[str]:
        catalog = {genre.casefold(): genre for genre in self.assistant.genre_catalog}
        unknown = [name for name in names if name.casefold() not in catalog]
        if unknown:
            raise ApiError(400, "bad_request", f"Unknown genre(s): {', '.join(unknown)}. "
                           f"Known: {', '.join(sorted(self.assistant.genre_catalog))}.")
        return {catalog[name.casefold()] for name in names}

    def _dispatch(self, routes):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            for pattern, handler in routes:
                match = re.fullmatch(pattern, parsed.path)
                if match:
                    handler(query, *match.groups())
                    return
            if any(re.fullmatch(pattern, parsed.path) for pattern, _ in self._all_routes()):
                raise ApiError(405, "method_not_allowed", f"{self.command} is not supported on {parsed.path}.")
            raise ApiError(404, "not_found", f"No route for {parsed.path}.")
        except ApiError as error:
            self._error(error.status, error.code, error.message)
        except Exception as error:  # pragma: no cover - last-resort guard for a long-running service
            self.log_error("Unhandled error on %s: %r", parsed.path, error)
            self._error(500, "internal_error", "Unexpected server error.")

    def _all_routes(self):
        return self.GET_ROUTES + self.POST_ROUTES + self.DELETE_ROUTES

    def do_GET(self):
        self._dispatch([(p, getattr(self, h)) for p, h in self.GET_ROUTES])

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self._dispatch([(p, getattr(self, h)) for p, h in self.POST_ROUTES])

    def do_DELETE(self):
        self._dispatch([(p, getattr(self, h)) for p, h in self.DELETE_ROUTES])

    def do_PUT(self):
        self._dispatch([])

    do_PATCH = do_PUT

    def do_OPTIONS(self):
        self.send_response(204)
        origin = self.headers.get("Origin")
        if origin and ("*" in self.cors_origins or origin in self.cors_origins):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # ----- service --------------------------------------------------------
    def static(self, query, *_):
        path = urlsplit(self.path).path
        name = STATIC_FILES[path]
        try:
            body = (STATIC_DIR / name).read_bytes()
        except OSError:
            raise ApiError(404, "not_found", "Static file is missing.") from None
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in {"application/javascript", "text/javascript"}:
            mime += "; charset=utf-8"
        self._send(200, body, mime, DOCS_CSP if name.startswith("docs") else PAGE_CSP)

    def health(self, query):
        data = self.assistant.dataset
        self._json(200, {"status": "ok", "version": __version__, "movies": len(data.movies),
                         "users": len(self.assistant.user_ratings), "ratings": len(data.ratings)})

    def openapi(self, query):
        self._json(200, SPEC)

    # ----- chat -----------------------------------------------------------
    def chat(self, query):
        payload = self._body()
        user_id = self._user(payload.get("user_id"))
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
            raise ApiError(400, "bad_request", f"'message' must be a non-empty string of at most {MAX_MESSAGE_CHARS} characters.")
        with self.assistant_lock:
            reply = respond(self.assistant, user_id, message)
        self._json(200, {
            "intent": reply.intent, "language": reply.language, "answer": reply.text,
            "headline": reply.headline,
            "recommendations": [recommendation_json(i, rec, reply.language)
                                for i, rec in enumerate(reply.recommendations, 1)],
            "details": reply.details,
        })

    # ----- users ----------------------------------------------------------
    def users(self, query):
        ids = sorted(self.assistant.user_ratings)
        self._json(200, {"count": len(ids), "user_ids": ids})

    def profile(self, query, raw_user):
        user_id = self._user(raw_user)
        ratings = self.assistant.user_ratings[user_id]
        genre_likes: dict[str, int] = {}
        for rating in ratings:
            if rating.value >= 4.0:
                for genre in self.assistant.dataset.movies[rating.movie_id].genres:
                    genre_likes[genre] = genre_likes.get(genre, 0) + 1
        self._json(200, {
            "user_id": user_id,
            "rating_count": len(ratings),
            "average_rating": round(sum(r.value for r in ratings) / len(ratings), 3),
            "favorite_genres": sorted(genre_likes, key=lambda g: (-genre_likes[g], g))[:3],
        })

    def reset_context(self, query, raw_user):
        user_id = self._user(raw_user)
        with self.assistant_lock:
            self.assistant.clear_context(user_id)
        self._send(204, b"", "text/plain")

    def recommendations(self, query, raw_user):
        user_id = self._user(raw_user)
        lang = _lang(query)
        seeds = set()
        for raw in _csv(query, "seed_movie_ids"):
            try:
                seeds.add(self._movie(raw))
            except ValueError:
                raise ApiError(400, "bad_request", "'seed_movie_ids' must be comma-separated integers.") from None
        kwargs = dict(
            limit=_int(query, "limit", 5, 1, 50), query=query.get("q", [""])[0],
            exclude_genres=self._genres(_csv(query, "exclude_genres")),
            required_genres=self._genres(_csv(query, "require_genres")),
            seed_movie_ids=seeds, min_year=_int(query, "min_year"), max_year=_int(query, "max_year"),
            max_ratings=_int(query, "max_ratings", None, 0, None), lang=lang)
        try:
            with self.assistant_lock:
                recs = self.assistant.recommend(user_id, **kwargs)
        except ValueError as error:
            raise ApiError(400, "bad_request", str(error)) from None
        self._json(200, {"user_id": user_id,
                         "recommendations": [recommendation_json(i, r, lang) for i, r in enumerate(recs, 1)]})

    def blind_spots(self, query, raw_user):
        user_id = self._user(raw_user)
        lang = _lang(query)
        with self.assistant_lock:
            rows = self.assistant.blind_spots(user_id, limit=_int(query, "limit", 5, 1, 19), lang=lang)
        self._json(200, {
            "user_id": user_id, "text": messages.blind_spot_text(rows, lang),
            "blind_spots": [{"genre": r["genre"], "count": r["count"], "average": r["average"] if r["count"] else None,
                             "suggestion": recommendation_json(1, r["suggestion"], lang) if r["suggestion"] else None}
                            for r in rows]})

    def neighbor_opinion(self, query, raw_user, raw_movie):
        user_id, movie_id, lang = self._user(raw_user), self._movie(raw_movie), _lang(query)
        with self.assistant_lock:
            data = self.assistant.neighbor_opinion(user_id, movie_id)
        self._json(200, data | {"text": messages.opinion_text(data, lang)})

    def explanation(self, query, raw_user, raw_movie):
        user_id, movie_id, lang = self._user(raw_user), self._movie(raw_movie), _lang(query)
        with self.assistant_lock:
            data = self.assistant.explanation_for(user_id, movie_id)
        self._json(200, data | {"text": messages.explanation_text(data, lang)})

    # ----- movies ---------------------------------------------------------
    def search(self, query):
        text = query.get("q", [""])[0].strip()
        if not text:
            raise ApiError(400, "bad_request", "'q' is required.")
        limit = _int(query, "limit", 10, 1, 50)
        with self.assistant_lock:
            hits = self.assistant.search(text, limit=limit)
        self._json(200, {"results": [self.assistant.movie_summary(mid) | {"relevance": round(score, 4)}
                                     for mid, score in hits]})

    def movie(self, query, raw_movie):
        movie_id = self._movie(raw_movie)
        facts = self.assistant.movie_facts(movie_id)
        facts.pop("genres")
        self._json(200, self.assistant.movie_summary(movie_id) | facts
                   | {"plot": self.assistant.dataset.movies[movie_id].plot})

    GET_ROUTES = [
        (r"/|/index\.html|/static/app\.js|/static/style\.css|/static/docs\.js|/docs", "static"),
        (r"/health", "health"),
        (r"/openapi\.json", "openapi"),
        (r"/api/v1/users", "users"),
        (r"/api/v1/users/(-?\d+)", "profile"),
        (r"/api/v1/users/(-?\d+)/recommendations", "recommendations"),
        (r"/api/v1/users/(-?\d+)/blind-spots", "blind_spots"),
        (r"/api/v1/users/(-?\d+)/movies/(\d+)/neighbor-opinion", "neighbor_opinion"),
        (r"/api/v1/users/(-?\d+)/movies/(\d+)/explanation", "explanation"),
        (r"/api/v1/movies/search", "search"),
        (r"/api/v1/movies/(\d+)", "movie"),
    ]
    POST_ROUTES = [(r"/api/v1/chat", "chat")]
    DELETE_ROUTES = [(r"/api/v1/users/(-?\d+)/context", "reset_context")]


def build_server(assistant: MovieAssistant, host: str, port: int, cors_origins=(),
                 access_log: bool = True) -> LocalHTTPServer:
    handler = type("ConfiguredMovieWebHandler", (MovieWebHandler,), {
        "assistant": assistant, "assistant_lock": Lock(), "cors_origins": frozenset(cors_origins),
        "access_log": access_log})
    return LocalHTTPServer((host, port), handler)


def _stop_on_sigterm(*_):
    # Container runtimes stop services with SIGTERM; shut down like Ctrl+C.
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description="Serve the movie assistant web UI and HTTP API.")
    parser.add_argument("--host", default=os.environ.get("MOVIE_ASSISTANT_HOST", "127.0.0.1"),
                        help="Bind address (default 127.0.0.1; use 0.0.0.0 in a container)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")), help="Port (default 8765)")
    parser.add_argument("--data-dir", default=os.environ.get("MOVIE_ASSISTANT_DATA_DIR"),
                        help="Path to ml-latest-small-filtered CSV files")
    parser.add_argument("--cors-origins", default=os.environ.get("MOVIE_ASSISTANT_CORS_ORIGINS", ""),
                        help="Comma-separated origins allowed to call the API from a browser, or *")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    assistant = MovieAssistant(load_dataset(args.data_dir))
    origins = [origin.strip() for origin in args.cors_origins.split(",") if origin.strip()]
    server = build_server(assistant, args.host, args.port, origins)
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    shown = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Movie assistant: http://{shown}:{args.port}  (API docs: /docs)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping movie assistant.", file=sys.stderr)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
