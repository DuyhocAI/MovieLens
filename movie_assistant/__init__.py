"""Local, dependency-free movie discovery assistant."""

__version__ = "1.0.0"

from .engine import MovieAssistant
from .models import Movie, Rating, Tag
from .storage import MovieDataset, load_dataset

__all__ = ["MovieAssistant", "Movie", "Rating", "Tag", "MovieDataset", "load_dataset"]
