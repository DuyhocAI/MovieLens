"""Full-text BM25 retrieval with boosted title, genre and tag fields."""

import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, movies, tags, tokenize):
        self.tokenize = tokenize
        self.postings = defaultdict(list)
        self.lengths = {}
        tag_terms = defaultdict(set)
        for tag in tags:
            tag_terms[tag.movie_id].update(tokenize(tag.value))
        for mid, movie in movies.items():
            terms = Counter(tokenize(movie.plot))
            terms.update({term: 3 * count for term, count in Counter(tokenize(movie.title)).items()})
            terms.update({term: 4 for term in tokenize(" ".join(movie.genres))})
            terms.update({term: 4 for term in tag_terms[mid]})
            self.lengths[mid] = sum(terms.values())
            for term, count in terms.items():
                self.postings[term].append((mid, count))
        self.total = len(movies)
        self.average_length = sum(self.lengths.values()) / max(1, self.total) or 1.0

    def search(self, text: str) -> dict[int, float]:
        return self.search_groups([{term: 1.0} for term in sorted(set(self.tokenize(text)))])

    def search_groups(self, groups) -> dict[int, float]:
        """Disjunction-max BM25: synonyms share one coverage contribution."""
        scores = defaultdict(float)
        coverage = defaultdict(int)
        for alternatives in groups:
            best = defaultdict(float)
            for term, weight in sorted(alternatives.items()):
                postings = self.postings.get(term, ())
                idf = math.log(1 + (self.total - len(postings) + 0.5) / (len(postings) + 0.5))
                for mid, count in postings:
                    length_factor = 1.2 * (0.25 + 0.75 * self.lengths[mid] / self.average_length)
                    value = weight * idf * count * 2.2 / (count + length_factor)
                    best[mid] = max(best[mid], value)
            for mid, value in best.items():
                scores[mid] += value
                coverage[mid] += 1
        # Favor matching several requested concepts over repeated mentions of one.
        for mid in scores:
            scores[mid] *= 0.5 + 0.5 * coverage[mid] / max(1, len(groups))
        maximum = max(scores.values(), default=1.0)
        return {mid: value / maximum for mid, value in scores.items()}

    def evidence(self, movie_id, plan):
        evidence = []
        for label, alternatives in zip(plan.labels, plan.groups):
            matches = [term for term in alternatives
                       if any(mid == movie_id for mid, _ in self.postings.get(term, ()))]
            if matches:
                evidence.append({"concept": label, "terms": matches[:2]})
        return evidence
