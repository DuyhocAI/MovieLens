#!/usr/bin/env python3
"""Capture real assistant answers for the sample users and queries, with local timings."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_assistant.dialogue import respond
from movie_assistant.engine import MovieAssistant, DEFAULT_ITEM_WEIGHT, DEFAULT_CONTENT_WEIGHT
from movie_assistant.storage import load_dataset

# The six sample queries from the brief, asked by each suggested test user.
BRIEF_QUERIES = [
    "What should I watch tonight?",
    "Why do you think I'd like that?",
    "I want a dark psychological thriller with a twist",
    "What do people with similar taste to mine think about Pulp Fiction?",
    "I liked Toy Story but I'm tired of animated movies — what else?",
    "What's my blind spot? What genres am I missing?",
]
EXTRA = {
    15: ["I don't want horror, give me sci-fi", "movies like Aliens but no horror",
         "Why would I like Arrival?", "Something funny for the weekend",
         "phim hành động không kinh dị", "phim chữa lành không hoạt hình",
         "phim ít người biết", "phim mới từ năm 2022"],
    30: ["người có gu giống tôi nghĩ gì về Inception?", "Recommend a war movie"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    assistant = MovieAssistant(load_dataset())
    initialization_seconds = time.perf_counter() - started
    conversations = []
    for uid in (1, 15, 30):
        for question in BRIEF_QUERIES + EXTRA.get(uid, []):
            started = time.perf_counter()
            reply = respond(assistant, uid, question)
            conversations.append({
                "user_id": uid, "question": question, "intent": reply.intent, "language": reply.language,
                "answer": reply.text, "seconds": round(time.perf_counter() - started, 4)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "item_weight": DEFAULT_ITEM_WEIGHT,
        "content_weight": DEFAULT_CONTENT_WEIGHT,
        "initialization_seconds": round(initialization_seconds, 4),
        "timing_note": "Single local run, including lazy cache work; not a controlled benchmark.",
        "conversations": conversations,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(conversations)} exchanges to {args.output}")


if __name__ == "__main__":
    main()
