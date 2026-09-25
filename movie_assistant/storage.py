"""CSV loading and lightweight data validation."""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import Movie, Rating, Tag


@dataclass
class MovieDataset:
    movies: dict[int, Movie]
    ratings: list[Rating]
    tags: list[Tag]

    @property
    def user_ratings(self) -> dict[int, list[Rating]]:
        result: dict[int, list[Rating]] = defaultdict(list)
        for rating in self.ratings:
            result[rating.user_id].append(rating)
        return dict(result)

    @property
    def movie_ratings(self) -> dict[int, list[Rating]]:
        result: dict[int, list[Rating]] = defaultdict(list)
        for rating in self.ratings:
            result[rating.movie_id].append(rating)
        return dict(result)


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def load_dataset(data_dir: str | Path | None = None) -> MovieDataset:
    """Load the supplied MovieLens CSVs, resolving the default path from this file."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small-filtered"
    data_dir = Path(data_dir)
    required = ("movies_with_plots.csv", "ratings.csv", "tags.csv")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {data_dir}: {', '.join(missing)}")

    movies: dict[int, Movie] = {}
    for row in _rows(data_dir / "movies_with_plots.csv"):
        movie_id = int(row["movieId"])
        year_text = row.get("year", "").strip()
        genres = tuple(g for g in row.get("genres", "").split("|") if g and g != "(no genres listed)")
        movies[movie_id] = Movie(
            movie_id=movie_id,
            title=row["title"].strip(),
            year=int(float(year_text)) if year_text else None,
            genres=genres,
            plot=row.get("plot", "").strip(),
        )

    ratings = [
        Rating(int(r["userId"]), int(r["movieId"]), float(r["rating"]), int(r["timestamp"]))
        for r in _rows(data_dir / "ratings.csv")
        if int(r["movieId"]) in movies
    ]
    tags = [
        Tag(int(r["userId"]), int(r["movieId"]), r["tag"].strip(), int(r["timestamp"]))
        for r in _rows(data_dir / "tags.csv")
        if int(r["movieId"]) in movies
    ]
    return MovieDataset(movies=movies, ratings=ratings, tags=tags)
