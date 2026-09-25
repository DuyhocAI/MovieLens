"""Interactive command-line interface."""

import argparse
import sys

from .dialogue import respond
from .engine import MovieAssistant
from .storage import load_dataset


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="MovieLens local movie discovery assistant")
    parser.add_argument("--user", type=int, help="MovieLens user ID (for example: 1, 15, or 30)")
    parser.add_argument("--data-dir", help="Path to ml-latest-small-filtered CSV files")
    args = parser.parse_args()
    assistant = MovieAssistant(load_dataset(args.data_dir))
    user_id = args.user
    while user_id is None or user_id not in assistant.user_ratings:
        raw = input("MovieLens user ID (try 1, 15, or 30): ").strip()
        try:
            user_id = int(raw)
        except ValueError:
            user_id = None
        if user_id not in assistant.user_ratings:
            print("ID not found; choose a user ID present in ratings.csv.")
            user_id = None
    print(f"Movie assistant ready for user {user_id}. Type :help for examples, :quit to exit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if question.casefold() in {":quit", ":q", "quit", "exit"}:
            print("Bye.")
            break
        if question.casefold().startswith(":user "):
            try:
                next_user = int(question.split(maxsplit=1)[1])
                if next_user not in assistant.user_ratings:
                    raise ValueError
                user_id = next_user
                print(f"Switched to user {user_id}.")
            except ValueError:
                print("Please provide a user ID present in ratings.csv.")
            continue
        try:
            print("\nAssistant: " + respond(assistant, user_id, question).text)
        except ValueError as error:
            print(f"\nAssistant: {error}")


if __name__ == "__main__":
    main()
