from dataclasses import dataclass, field


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    plot: str


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    value: float
    timestamp: int


@dataclass(frozen=True)
class Tag:
    user_id: int
    movie_id: int
    value: str
    timestamp: int


@dataclass(frozen=True)
class Recommendation:
    movie: Movie
    score: float
    explanation: str
    rating_count: int
    average_rating: float
    evidence: dict = field(default_factory=dict, compare=False)
