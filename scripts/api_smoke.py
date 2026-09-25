#!/usr/bin/env python3
"""Start the HTTP API in-process on the real data and record a few request/response pairs."""

import argparse
import json
import sys
import threading
from http.client import HTTPConnection
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_assistant.engine import MovieAssistant
from movie_assistant.storage import load_dataset
from movie_assistant.web import build_server

REQUESTS = [
    ("GET", "/health", None),
    ("GET", "/api/v1/users/15", None),
    ("GET", "/api/v1/users/15/recommendations?limit=3&q=space&exclude_genres=Horror", None),
    ("POST", "/api/v1/chat", {"user_id": 15, "message": "I don't want horror, give me sci-fi"}),
    ("POST", "/api/v1/chat", {"user_id": 15, "message": "Why would I like that?"}),
    ("GET", "/api/v1/users/1/movies/296/neighbor-opinion", None),
    ("GET", "/api/v1/users/1/blind-spots?limit=3", None),
    ("GET", "/api/v1/movies/search?q=dark%20psychological%20thriller&limit=3", None),
    ("GET", "/api/v1/users/9999", None),
    ("GET", "/api/v1/users/15/recommendations?exclude_genres=Cartoon", None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/api_smoke.json"))
    args = parser.parse_args()
    server = build_server(MovieAssistant(load_dataset()), "127.0.0.1", 0, access_log=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    exchanges = []
    try:
        for method, path, body in REQUESTS:
            connection = HTTPConnection(*server.server_address, timeout=30)
            connection.request(method, path, json.dumps(body) if body else None,
                               {"Content-Type": "application/json"} if body else {})
            response = connection.getresponse()
            payload = json.loads(response.read() or b"null")
            connection.close()
            exchanges.append({"request": {"method": method, "path": path, "body": body},
                              "status": response.status, "response": payload})
    finally:
        server.shutdown()
        server.server_close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exchanges, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(exchanges)} exchanges to {args.output}: "
          + ", ".join(str(item["status"]) for item in exchanges))


if __name__ == "__main__":
    main()
