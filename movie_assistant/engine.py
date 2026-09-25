"""Recommendation, content search, and evidence-based explanations."""

import math
import heapq
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from .models import Movie, Rating, Recommendation
from .storage import MovieDataset
from .item_similarity import ItemSimilarity
from .retrieval import BM25Index
from .query import plan_query
from .messages import recommendation_text

DEFAULT_ITEM_WEIGHT = 1.0
DEFAULT_CONTENT_WEIGHT = 0.25  # Selected on validation; see artifacts/evaluation_v2.json.
# MovieLens lists IMAX as a genre, but it is a projection format.
NON_GENRES = {"IMAX"}


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = set("""a an and are as at be been being by can did do does for from had has have he her hers him his how i if in into is it its me my of on or our she that the their them then there these they this those to too was we were what when where which who why will with would you your but about movie film movies films want like watch think tonight people similar tired something else phim toi muon xem gi mot nhung cua va la co den""".split())
QUERY_SYNONYMS = {
    "giat gan": "thriller", "kinh di": "horror", "hai": "comedy",
    "lang man": "romance", "hanh dong": "action", "hoat hinh": "animation",
    "khoa hoc vien tuong": "sci fi", "bi an": "mystery", "toi pham": "crime",
    "tam ly": "psychological", "den toi": "dark",
}


def _ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char)).replace("đ", "d")


def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(_ascii(text)) if len(t) > 2 and t not in STOP_WORDS]


@dataclass
class _Neighbor:
    user_id: int
    similarity: float
    common_count: int


class MovieAssistant:
    """Small-data recommender with collaborative and plot/genre signals."""

    def __init__(self, dataset: MovieDataset):
        self.dataset = dataset
        self.user_ratings = dataset.user_ratings
        self.movie_ratings = dataset.movie_ratings
        self.genre_catalog = {genre for movie in dataset.movies.values() for genre in movie.genres}
        known_years = [movie.year for movie in dataset.movies.values() if movie.year is not None]
        self.year_bounds = (min(known_years), max(known_years)) if known_years else (None, None)
        self.global_mean = sum(r.value for r in dataset.ratings) / max(1, len(dataset.ratings))
        self.user_means = {
            uid: sum(r.value for r in rows) / len(rows)
            for uid, rows in self.user_ratings.items() if rows
        }
        self.movie_stats = {
            mid: (len(rows), sum(r.value for r in rows) / len(rows))
            for mid, rows in self.movie_ratings.items() if rows
        }
        # Bound the document vectors to the most informative terms to keep RAM modest.
        self.features = self._build_features()
        self.inverted_features: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for movie_id, vector in self.features.items():
            for term, value in vector.items():
                self.inverted_features[term].append((movie_id, value))
        self._neighbor_cache: dict[int, list[_Neighbor]] = {}
        # Conversation context is kept per user so concurrent users never share it.
        self._last_recommendations: dict[int, list[Recommendation]] = {}
        self._tag_map: dict[int, list[str]] = defaultdict(list)
        for tag in dataset.tags:
            self._tag_map[tag.movie_id].append(tag.value)
        self.item_similarity = ItemSimilarity(self.user_ratings)
        self.retrieval = BM25Index(dataset.movies, dataset.tags, _tokens)
        self._item_score_cache = {}
        self._last_content_key = None
        self._last_content_scores = {}

    def _build_features(self) -> dict[int, dict[str, float]]:
        doc_terms: dict[int, Counter[str]] = {}
        document_frequency: Counter[str] = Counter()
        for mid, movie in self.dataset.movies.items():
            text = " ".join((movie.title, " ".join(movie.genres), movie.plot))
            terms = Counter(_tokens(text))
            doc_terms[mid] = terms
            document_frequency.update(terms.keys())
        total = max(1, len(doc_terms))
        vectors: dict[int, dict[str, float]] = {}
        for mid, terms in doc_terms.items():
            movie = self.dataset.movies[mid]
            weighted = {
                term: (1.0 + math.log(count)) * (math.log((total + 1) / (document_frequency[term] + 1)) + 1.0)
                for term, count in terms.items()
            }
            selected = dict(sorted(weighted.items(), key=lambda item: item[1], reverse=True)[:80])
            # Keep explicit title and genre signals even when plot text is long.
            for term in _tokens(movie.title + " " + " ".join(movie.genres)):
                selected[term] = weighted[term]
            norm = math.sqrt(sum(value * value for value in selected.values())) or 1.0
            vectors[mid] = {term: value / norm for term, value in selected.items()}
        return vectors

    def _similar_users(self, user_id: int) -> list[_Neighbor]:
        if user_id in self._neighbor_cache:
            return self._neighbor_cache[user_id]
        target = {r.movie_id: r.value for r in self.user_ratings.get(user_id, [])}
        target_mean = self.user_means.get(user_id, self.global_mean)
        neighbors: list[_Neighbor] = []
        if len(target) >= 2:
            for other_id, rows in self.user_ratings.items():
                if other_id == user_id:
                    continue
                other = {r.movie_id: r.value for r in rows}
                common = target.keys() & other.keys()
                if len(common) < 3:
                    continue
                other_mean = self.user_means[other_id]
                left = [target[mid] - target_mean for mid in common]
                right = [other[mid] - other_mean for mid in common]
                denom = math.sqrt(sum(x*x for x in left) * sum(y*y for y in right))
                if not denom:
                    continue
                pearson = sum(x*y for x, y in zip(left, right)) / denom
                # Shrink noisy similarities when users share only a few rated films.
                similarity = pearson * (len(common) / (len(common) + 10.0))
                if similarity > 0:
                    neighbors.append(_Neighbor(other_id, similarity, len(common)))
        neighbors.sort(key=lambda n: n.similarity, reverse=True)
        self._neighbor_cache[user_id] = neighbors[:50]
        return self._neighbor_cache[user_id]

    def _content_profile(self, user_id: int, positive_seeds: set[int] | None = None) -> dict[str, float]:
        profile: Counter[str] = Counter()
        positive_seeds = positive_seeds or set()
        for rating in self.user_ratings.get(user_id, []):
            if rating.value < 3.5 or rating.movie_id in positive_seeds:
                continue
            weight = max(0.25, rating.value - 3.0)
            for term, value in self.features.get(rating.movie_id, {}).items():
                profile[term] += value * weight
        for movie_id in positive_seeds:
            for term, value in self.features.get(movie_id, {}).items():
                profile[term] += value * 2.0
        norm = math.sqrt(sum(v*v for v in profile.values())) or 1.0
        return {term: value / norm for term, value in profile.items()}

    def _content_score(self, movie_id: int, profile: dict[str, float]) -> float:
        return sum(value * profile.get(term, 0.0) for term, value in self.features.get(movie_id, {}).items())

    def _content_scores(self, profile: dict[str, float]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term, weight in profile.items():
            for movie_id, value in self.inverted_features.get(term, ()):
                scores[movie_id] += weight * value
        return scores

    def _bayesian_mean(self, movie_id: int, prior_count: float = 8.0) -> tuple[float, int, float]:
        count, average = self.movie_stats.get(movie_id, (0, self.global_mean))
        score = (count * average + prior_count * self.global_mean) / (count + prior_count)
        return score, count, average

    def last_recommendations_for(self, user_id: int) -> list[Recommendation]:
        return self._last_recommendations.get(user_id, [])

    def clear_context(self, user_id: int) -> None:
        self._last_recommendations.pop(user_id, None)

    def recommend(self, user_id: int, limit: int = 5, exclude_genres: set[str] | None = None,
                  query: str = "", blend: tuple[float, float, float] = (0.70, 0.10, 0.20),
                  seed_movie_ids: set[int] | None = None,
                  item_weight: float = DEFAULT_ITEM_WEIGHT,
                  min_year: int | None = None, max_year: int | None = None,
                  content_weight: float = DEFAULT_CONTENT_WEIGHT,
                  max_ratings: int | None = None,
                  required_genres: set[str] | None = None,
                  lang: str = "en") -> list[Recommendation]:
        if user_id not in self.user_ratings:
            raise ValueError(f"User {user_id} is not present in the rating data.")
        if len(blend) != 3 or any(not math.isfinite(weight) or weight < 0 for weight in blend) or not math.isclose(sum(blend), 1.0, abs_tol=1e-6):
            raise ValueError("blend must contain non-negative collaborative/content/quality weights summing to 1")
        if not 0 <= item_weight <= 1:
            raise ValueError("item_weight must be between 0 and 1")
        if not 0 <= content_weight <= 1:
            raise ValueError("content_weight must be between 0 and 1")
        if max_ratings is not None and max_ratings < 0:
            raise ValueError("max_ratings must be non-negative")
        if min_year is not None and max_year is not None and min_year > max_year:
            raise ValueError("min_year cannot be later than max_year")
        self._last_recommendations[user_id] = []
        watched = {r.movie_id for r in self.user_ratings[user_id]}
        seed_movie_ids = {mid for mid in (seed_movie_ids or set()) if mid in self.dataset.movies}
        excluded = {g.casefold() for g in (exclude_genres or set())}
        required = set(required_genres or ())
        content_key = (user_id, tuple(sorted(seed_movie_ids)))
        if content_key != self._last_content_key:
            self._last_content_scores = self._content_scores(self._content_profile(user_id, seed_movie_ids))
            self._last_content_key = content_key
        content_scores = self._last_content_scores
        seed_genres = [set(self.dataset.movies[mid].genres) for mid in seed_movie_ids]
        query_plan = plan_query(query, _tokens)
        query_scores = dict(self.search(query, limit=len(self.dataset.movies))) if query_plan.groups else {}
        if query_plan.groups and not query_scores:
            return []
        query_candidate_ids = set(query_scores)
        neighbors = self._similar_users(user_id)
        neighbor_votes: dict[int, list[tuple[float, float, int, float]]] = defaultdict(list)
        for neighbor in neighbors:
            for rating in self.user_ratings[neighbor.user_id]:
                if rating.movie_id not in watched:
                    neighbor_votes[rating.movie_id].append((neighbor.similarity, rating.value,
                                                            neighbor.user_id, self.user_means[neighbor.user_id]))

        target_mean = self.user_means[user_id]
        item_scores, item_evidence = {}, {}
        if item_weight:
            cache_key = (user_id, tuple(sorted(seed_movie_ids)))
            if cache_key not in self._item_score_cache:
                self._item_score_cache[cache_key] = self.item_similarity.score(
                    self.user_ratings[user_id], seed_movie_ids)
            item_scores, item_evidence = self._item_score_cache[cache_key]
        ranked: list[tuple[float, Movie, float, list[tuple[float, float, int, float]], float, int]] = []
        for movie_id, movie in self.dataset.movies.items():
            if movie_id in watched or movie_id in seed_movie_ids or any(g.casefold() in excluded for g in movie.genres):
                continue
            if min_year is not None and (movie.year is None or movie.year < min_year):
                continue
            if max_year is not None and (movie.year is None or movie.year > max_year):
                continue
            if not query_plan.accepts(movie.genres):
                continue
            if required and not required.issubset(movie.genres):
                continue
            if query_candidate_ids and movie_id not in query_candidate_ids:
                continue
            bayes, count, average = self._bayesian_mean(movie_id)
            if max_ratings is not None and count > max_ratings:
                continue
            seed_genre_score = max((len(genres & set(movie.genres)) /
                                    math.sqrt(max(1, len(genres) * len(movie.genres)))
                                    for genres in seed_genres), default=0.0)
            content = max(content_scores.get(movie_id, 0.0), query_scores.get(movie_id, 0.0),
                          seed_genre_score * 0.8)
            votes = neighbor_votes.get(movie_id, [])
            total_weight = sum(w for w, _, _, _ in votes)
            residual = sum(w * (value - neighbor_mean) for w, value, _, neighbor_mean in votes)
            # Center neighbor ratings around their own mean before transferring preference.
            cf = min(5.0, max(0.5, target_mean + residual / (total_weight + 4.0)))
            # The content contribution is bounded, and Bayesian quality stabilizes sparse films.
            content_estimate = min(5.0, 3.0 + 2.0 * min(1.0, content))
            personal_estimate = cf if votes else target_mean
            score = blend[0] * personal_estimate + blend[1] * content_estimate + blend[2] * bayes
            ranked.append((score, movie, content, votes, average, count))
        if item_weight and any(row[1].movie_id in item_scores for row in ranked):
            base_order = sorted(ranked, key=lambda row: (-row[0], -row[5], row[1].movie_id))
            item_order = sorted((row for row in ranked if row[1].movie_id in item_scores),
                                key=lambda row: (-item_scores[row[1].movie_id], row[1].movie_id))
            item_ranks = {row[1].movie_id: rank for rank, row in enumerate(item_order, 1)}
            fused = []
            for rank, row in enumerate(base_order, 1):
                item_rank = item_ranks.get(row[1].movie_id)
                # Weighted reciprocal rank fusion avoids mixing incompatible raw scales.
                score = 5 * 61 * ((1 - item_weight) / (60 + rank)
                                 + (item_weight / (60 + item_rank) if item_rank else 0))
                fused.append((score, *row[1:]))
            ranked = fused
        if content_weight and ranked and any(content_scores.get(row[1].movie_id, 0) > 0 for row in ranked):
            # Content has its own candidate ranking, including zero-rating films.
            # Fuse ranks instead of inflating a rating estimate for sparse items.
            content_order = sorted((row for row in ranked if content_scores.get(row[1].movie_id, 0) > 0),
                                   key=lambda row: (-content_scores[row[1].movie_id], row[1].movie_id))
            content_ranks = {row[1].movie_id: i for i, row in enumerate(content_order, 1)}
            personal_order = sorted(ranked, key=lambda row: (-row[0], -row[5], row[1].movie_id))
            ranked = [(5 * 61 * ((1 - content_weight) / (60 + rank)
                       + (content_weight / (60 + content_ranks[row[1].movie_id])
                          if row[1].movie_id in content_ranks else 0)), *row[1:])
                      for rank, row in enumerate(personal_order, 1)]
        if seed_movie_ids:
            # A current liked-film request should not be drowned out by a long history.
            seed_links = {seed: {mid: similarity for mid, similarity, _ in
                                  self.item_similarity.neighbors(seed)} for seed in seed_movie_ids}
            seed_maxima = {seed: max(links.values(), default=1.0) for seed, links in seed_links.items()}
            affinities = {}
            for row in ranked:
                movie = row[1]
                affinity = 0.0
                for seed in seed_movie_ids:
                    source = self.dataset.movies[seed]
                    cosine = self._content_score(movie.movie_id, self.features[seed])
                    genre_overlap = len(set(source.genres) & set(movie.genres)) / math.sqrt(
                        max(1, len(source.genres) * len(movie.genres)))
                    link = seed_links[seed].get(movie.movie_id, 0.0) / seed_maxima[seed]
                    affinity = max(affinity, 0.5 * cosine + 0.3 * genre_overlap + 0.2 * link)
                affinities[movie.movie_id] = affinity
            maximum = max(affinities.values(), default=0.0) or 1.0
            ranked = [(0.4 * row[0] + 3.0 * affinities[row[1].movie_id] / maximum, *row[1:])
                      for row in ranked]
        if query_scores:
            # Explicit current intent takes precedence over general taste/popularity.
            ranked = [(0.4 * row[0] + 3.0 * query_scores.get(row[1].movie_id, 0), *row[1:])
                      for row in ranked]
        ranked = heapq.nlargest(max(0, limit), ranked, key=lambda row: (row[0], row[5], -row[1].movie_id))
        liked_genres = {genre for rating in self.user_ratings[user_id] if rating.value >= 4.0
                        for genre in self.dataset.movies[rating.movie_id].genres}
        result = []
        for score, movie, content, votes, average, count in ranked:
            evidence = self._evidence(user_id, movie, content_scores.get(movie.movie_id, 0.0), votes,
                                      average, count, liked_genres, seed_movie_ids, query, query_plan,
                                      item_scores, item_evidence)
            result.append(Recommendation(movie, score, recommendation_text(movie.title, evidence, lang),
                                         count, average, evidence))
        self._last_recommendations[user_id] = result
        return result

    def _evidence(self, user_id: int, movie: Movie, content: float,
                  votes: list[tuple[float, float, int, float]], average: float, count: int,
                  liked_genres: set[str], seed_movie_ids: set[int], query: str, query_plan,
                  item_scores: dict[int, float], item_evidence: dict) -> dict:
        """Collect the facts behind one recommendation; wording lives in messages.py."""
        seed_matches, content_signal = [], None
        if content >= 0.02:
            movie_vector = self.features.get(movie.movie_id, {})
            for seed_id in sorted(seed_movie_ids):
                seed = self.dataset.movies[seed_id]
                shared_seed_genres = [genre for genre in seed.genres if genre in movie.genres]
                seed_vector = self.features.get(seed_id, {})
                overlap = sorted(((term, seed_vector[term] * movie_vector[term])
                                  for term in seed_vector.keys() & movie_vector.keys()),
                                 key=lambda row: row[1], reverse=True)
                if shared_seed_genres or overlap:
                    seed_matches.append({
                        "movie_id": seed_id, "title": seed.title, "shared_genres": shared_seed_genres,
                        "shared_terms": [] if shared_seed_genres else [term for term, _ in overlap[:2]],
                    })
            if not seed_matches:
                if not seed_movie_ids:
                    content_signal = "profile"
                elif content >= 0.05:
                    content_signal = "history"
        neighbors = None
        if votes:
            weight = sum(w for w, _, _, _ in votes)
            neighbors = {"count": len(votes),
                         "weighted_mean": round(sum(w * v for w, v, _, _ in votes) / weight, 3)}
        item_link = None
        if movie.movie_id in item_scores and movie.movie_id in item_evidence:
            source, _, support = item_evidence[movie.movie_id]
            item_link = {"source_movie_id": source, "source_title": self.dataset.movies[source].title,
                         "support": support}
        has_query = bool(query_plan.groups)
        return {
            "shared_genres": [genre for genre in movie.genres if genre in liked_genres],
            "seed_matches": seed_matches,
            "content_signal": content_signal,
            "content_similarity": round(content, 4),
            "query": query.strip() if has_query and query.strip() else None,
            "query_terms": self.retrieval.evidence(movie.movie_id, query_plan) if has_query else [],
            "tone_caveat": has_query and any(label in {"sad ending", "feel good", "mind bending", "dark", "psychological"}
                                             for label in query_plan.labels),
            "neighbors": neighbors,
            "item_link": item_link,
            "rating_count": count,
            "average_rating": round(average, 3) if count else None,
            "low_confidence": count < 5 or len(self.user_ratings[user_id]) < 20,
        }

    def search(self, query: str, limit: int = 10, expand_query: bool = True) -> list[tuple[int, float]]:
        if expand_query:
            plan = plan_query(query, _tokens)
            scores = self.retrieval.search_groups(plan.groups)
            return sorted(((mid, score) for mid, score in scores.items()
                           if plan.accepts(self.dataset.movies[mid].genres)),
                          key=lambda item: (-item[1], item[0]))[:max(0, limit)]
        normalized_query = _ascii(query)
        for phrase, replacement in sorted(QUERY_SYNONYMS.items(), key=lambda pair: -len(pair[0])):
            normalized_query = re.sub(rf"\b{re.escape(phrase)}\b", replacement, normalized_query)
        scores_by_movie = self.retrieval.search(normalized_query)
        query_tokens = set(_tokens(normalized_query))
        matched_genres = {
            genre for movie in self.dataset.movies.values() for genre in movie.genres
            if set(_tokens(genre.replace("-", " "))).issubset(query_tokens)
            and _tokens(genre.replace("-", " "))
        }
        scores = [(mid, score) for mid, score in scores_by_movie.items() if score > 0
                  and (not matched_genres or matched_genres.intersection(self.dataset.movies[mid].genres))]
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:max(0, limit)]

    def find_movie(self, text: str) -> Movie | None:
        needle = text.casefold().strip().strip('"\'?!., ')
        if not needle:
            return None
        exact = [m for m in self.dataset.movies.values() if m.title.casefold() == needle]
        if exact:
            return exact[0]
        matches = [m for m in self.dataset.movies.values() if needle in m.title.casefold()]
        if not matches:
            words = set(_tokens(needle))
            if words:
                matches = [m for m in self.dataset.movies.values()
                           if words.issubset(set(_tokens(m.title)))]
        matches.sort(key=lambda m: (len(self.movie_ratings.get(m.movie_id, [])), m.title), reverse=True)
        return matches[0] if matches else None

    def movie_summary(self, movie_id: int) -> dict:
        movie = self.dataset.movies[movie_id]
        return {"movie_id": movie.movie_id, "title": movie.title, "year": movie.year, "genres": list(movie.genres)}

    def movie_facts(self, movie_id: int) -> dict:
        bayes, count, average = self._bayesian_mean(movie_id)
        return {
            "genres": list(self.dataset.movies[movie_id].genres),
            "rating_count": count,
            "average_rating": round(average, 3) if count else None,
            "smoothed_rating": round(bayes, 3),
            "tags": sorted(set(self._tag_map.get(movie_id, [])), key=str.casefold),
        }

    def neighbor_opinion(self, user_id: int, movie_id: int, pool_size: int = 20) -> dict:
        """What the user's closest taste neighbours think of one movie."""
        if user_id not in self.user_ratings:
            raise ValueError(f"User {user_id} is not present in the rating data.")
        neighbors = self._similar_users(user_id)[:pool_size]
        by_id = {n.user_id: n for n in neighbors}
        rated = sorted(((by_id[r.user_id], r.value) for r in self.movie_ratings.get(movie_id, [])
                        if r.user_id in by_id), key=lambda pair: (-pair[0].similarity, pair[0].user_id))
        weight = sum(n.similarity for n, _ in rated)
        count, average = self.movie_stats.get(movie_id, (0, 0.0))
        own = next((r.value for r in self.user_ratings[user_id] if r.movie_id == movie_id), None)
        return {
            "status": "ok",
            "movie": self.movie_summary(movie_id),
            "neighbor_pool": len(neighbors),
            "raters": [{"user_id": n.user_id, "similarity": round(n.similarity, 3),
                        "common_movies": n.common_count, "rating": value} for n, value in rated],
            "weighted_mean": round(sum(n.similarity * value for n, value in rated) / weight, 3) if rated else None,
            "liked": sum(value >= 4.0 for _, value in rated),
            "disliked": sum(value <= 2.5 for _, value in rated),
            "overall": {"count": count, "average": round(average, 3) if count else None},
            "your_rating": own,
        }

    def explanation_for(self, user_id: int, movie_id: int | None = None) -> dict:
        """Evidence for why a movie suits the user; defaults to the latest top recommendation."""
        if user_id not in self.user_ratings:
            raise ValueError(f"User {user_id} is not present in the rating data.")
        recent = self.last_recommendations_for(user_id)
        if movie_id is None:
            if not recent:
                return {"status": "no_context"}
            movie_id = recent[0].movie.movie_id
        match = next((r for r in recent if r.movie.movie_id == movie_id), None)
        if match:
            return {"status": "recommended", "movie": self.movie_summary(movie_id), "evidence": match.evidence}
        own = next((r.value for r in self.user_ratings[user_id] if r.movie_id == movie_id), None)
        if own is not None:
            return {"status": "already_rated", "movie": self.movie_summary(movie_id), "your_rating": own}
        movie = self.dataset.movies[movie_id]
        liked_genres = {genre for rating in self.user_ratings[user_id] if rating.value >= 4.0
                        for genre in self.dataset.movies[rating.movie_id].genres}
        return {
            "status": "not_recommended",
            "movie": self.movie_summary(movie_id),
            "shared_genres": [genre for genre in movie.genres if genre in liked_genres],
            "content_similarity": round(self._content_score(movie_id, self._content_profile(user_id)), 4),
            "stats": self.movie_facts(movie_id),
        }

    def blind_spots(self, user_id: int, limit: int = 5, suggestions: int = 3, lang: str = "en") -> list[dict]:
        """Genres the user has rarely rated, each with one personalised film to try."""
        if user_id not in self.user_ratings:
            raise ValueError(f"User {user_id} is not present in the rating data.")
        rows = self.user_ratings[user_id]
        by_genre: dict[str, list[float]] = defaultdict(list)
        for rating in rows:
            for genre in self.dataset.movies[rating.movie_id].genres:
                by_genre[genre].append(rating.value)
        catalog_genres = self.genre_catalog - NON_GENRES
        spots = sorted(((g, len(by_genre[g]), sum(by_genre[g]) / len(by_genre[g]) if by_genre[g] else 0.0)
                        for g in catalog_genres if len(by_genre[g]) < max(3, len(rows) * 0.05)),
                       key=lambda item: (item[1], item[0]))[:limit]
        result, picks = [], []
        for index, (genre, count, average) in enumerate(spots):
            suggestion = None
            if index < suggestions:
                found = self.recommend(user_id, limit=1, required_genres={genre}, lang=lang)
                suggestion = found[0] if found else None
                if suggestion:
                    picks.append(suggestion)
            result.append({"genre": genre, "count": count, "average": round(average, 3), "suggestion": suggestion})
        # "Why would I like that?" should refer to these suggestions.
        self._last_recommendations[user_id] = picks
        return result
