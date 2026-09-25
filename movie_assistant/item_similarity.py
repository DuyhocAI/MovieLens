"""Lazy item-item collaborative filtering using positive rating co-occurrence."""

import heapq
import math
from collections import defaultdict

from .models import Rating


class ItemSimilarity:
    """Activity-weighted cosine with support shrinkage; no dense movie matrix."""

    def __init__(self, user_ratings: dict[int, list[Rating]]):
        self.positive = {
            uid: {r.movie_id for r in rows if r.value >= 3.5}
            for uid, rows in user_ratings.items()
        }
        self.users: dict[int, list[int]] = defaultdict(list)
        self.norms: dict[int, float] = defaultdict(float)
        self.activity = {uid: 1.0 / math.log2(2 + len(mids))
                         for uid, mids in self.positive.items()}
        for uid, mids in self.positive.items():
            for mid in mids:
                self.users[mid].append(uid)
                self.norms[mid] += self.activity[uid]
        self.cache: dict[int, list[tuple[int, float, int]]] = {}

    def neighbors(self, movie_id: int) -> list[tuple[int, float, int]]:
        if movie_id not in self.cache:
            weighted: dict[int, float] = defaultdict(float)
            counts: dict[int, int] = defaultdict(int)
            for uid in self.users.get(movie_id, ()):
                for other in self.positive[uid]:
                    if other != movie_id:
                        weighted[other] += self.activity[uid]
                        counts[other] += 1
            similarities = (
                (mid, weight / math.sqrt(self.norms[movie_id] * self.norms[mid])
                 * counts[mid] / (counts[mid] + 5.0), counts[mid])
                for mid, weight in weighted.items()
            )
            self.cache[movie_id] = heapq.nlargest(
                100, similarities, key=lambda row: (row[1], row[2], -row[0]))
        return self.cache[movie_id]

    def score(self, ratings: list[Rating], seeds: set[int]) -> tuple[
            dict[int, float], dict[int, tuple[int, float, int]]]:
        preferences = {r.movie_id: (r.value - 3.0 if r.value >= 3.5
                                    else -0.25 * (3.0 - r.value)) for r in ratings}
        preferences.update({mid: 2.0 for mid in seeds})
        scores: dict[int, float] = defaultdict(float)
        evidence: dict[int, tuple[int, float, int]] = {}
        for source, preference in sorted(preferences.items()):
            if not preference:
                continue
            for mid, similarity, support in self.neighbors(source):
                contribution = preference * similarity
                scores[mid] += contribution
                if contribution > 0 and contribution > evidence.get(mid, (0, 0.0, 0))[1]:
                    evidence[mid] = (source, contribution, support)
        return {mid: score for mid, score in scores.items() if score > 0}, evidence
