#!/usr/bin/env python3
"""Temporal last-rating holdout evaluation; reports Recall@10 and NDCG@10."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Allow `python scripts/evaluate_recommender.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_assistant.engine import MovieAssistant
from movie_assistant.models import Rating
from movie_assistant.storage import MovieDataset, load_dataset

BLENDS = (
    (0.75, 0.00, 0.25),
    (0.70, 0.10, 0.20),
    (0.65, 0.15, 0.20),
    (0.60, 0.20, 0.20),
    (0.65, 0.10, 0.25),
)


def _compare_configurations(model, holdouts, k, configurations):
    """Evaluate each user's candidates together, reusing their cached content profile."""
    totals = [[0, 0.0] for _ in configurations]
    evaluated = 0
    for uid, target in holdouts.items():
        if target.value < 4.0:
            continue
        evaluated += 1
        for index, (blend, item_weight) in enumerate(configurations):
            recommendations = model.recommend(uid, limit=k, blend=blend, item_weight=item_weight,
                                              content_weight=0.0)
            rank = next((i for i, rec in enumerate(recommendations, 1)
                         if rec.movie.movie_id == target.movie_id), None)
            if rank:
                totals[index][0] += 1
                totals[index][1] += 1.0 / math.log2(rank + 1)
    return evaluated, totals


def _rank_metrics(model: MovieAssistant, holdouts: dict[int, Rating], k: int,
                  blend: tuple[float, float, float] | None = None,
                  collect_failures: bool = False,
                  item_weight: float = 0.0) -> tuple[int, int, float, list[tuple[int, Rating]]]:
    evaluated = hits = 0
    ndcg = 0.0
    misses: list[tuple[int, Rating]] = []
    for user_id, target in holdouts.items():
        if target.value < 4.0:
            continue
        evaluated += 1
        recommendations = model.recommend(user_id, limit=k, blend=blend or (0.70, 0.10, 0.20),
                                          item_weight=item_weight, content_weight=0.0)
        rank = next((i for i, rec in enumerate(recommendations, 1)
                     if rec.movie.movie_id == target.movie_id), None)
        if rank:
            hits += 1
            ndcg += 1.0 / math.log2(rank + 1)
        elif collect_failures:
            misses.append((user_id, target))
    return evaluated, hits, ndcg, misses


def evaluate(dataset: MovieDataset, k: int = 10, validation_only: bool = False) -> dict[str, object]:
    if k < 1:
        raise ValueError("k must be positive")
    by_user: dict[int, list[Rating]] = defaultdict(list)
    for rating in dataset.ratings:
        by_user[rating.user_id].append(rating)
    tune_training: list[Rating] = []
    validation: dict[int, Rating] = {}
    test_training: list[Rating] = []
    test: dict[int, Rating] = {}
    for user_id, rows in by_user.items():
        rows.sort(key=lambda r: (r.timestamp, r.movie_id))
        if len(rows) >= 20:
            validation[user_id] = rows[-2]
            test[user_id] = rows[-1]
            tune_training.extend(rows[:-2])
            test_training.extend(rows[:-1])
        else:
            tune_training.extend(rows)
            test_training.extend(rows)

    # Tags are not used by unqueried ranking. Exclude them from this rating-only
    # diagnostic so held-out interaction timestamps cannot enter retrieval either.
    validation_model = MovieAssistant(MovieDataset(dataset.movies, tune_training, []))
    validation_rows = []
    evaluated, blend_totals = _compare_configurations(
        validation_model, validation, k, [(blend, 0.0) for blend in BLENDS])
    for blend, (hits, ndcg) in zip(BLENDS, blend_totals):
        validation_rows.append((ndcg / evaluated if evaluated else 0.0,
                                hits / evaluated if evaluated else 0.0, blend, evaluated))
    validation_rows.sort(reverse=True)
    _, _, selected_blend, validation_count = validation_rows[0]

    fusion_rows = []
    item_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    evaluated, fusion_totals = _compare_configurations(
        validation_model, validation, k, [(selected_blend, weight) for weight in item_weights])
    for item_weight, (hits, ndcg) in zip(item_weights, fusion_totals):
        fusion_rows.append({"item_weight": item_weight, "hits": hits,
                            "recall": hits / evaluated if evaluated else 0.0,
                            "ndcg": ndcg / evaluated if evaluated else 0.0})
    # On ties prefer the simpler model; selection never consults final test results.
    selected_item_weight = max(fusion_rows, key=lambda row: (row["ndcg"], row["recall"],
                                                            -row["item_weight"]))["item_weight"]
    selection = {"k": k, "validation_users": validation_count,
                 "selected_blend": selected_blend, "selected_item_weight": selected_item_weight,
                 "validation_fusion": fusion_rows,
                 "validation_blends": [{"blend": blend, "ndcg": ndcg, "recall": recall}
                                       for ndcg, recall, blend, _ in validation_rows]}
    if validation_only:
        return selection

    model = MovieAssistant(MovieDataset(dataset.movies, test_training, []))
    evaluated, model_hits, model_ndcg, misses = _rank_metrics(
        model, test, k, selected_blend, collect_failures=True, item_weight=selected_item_weight)
    _, legacy_hits, legacy_ndcg, _ = _rank_metrics(model, test, k, selected_blend)
    baseline_hits = 0
    baseline_ndcg = 0.0
    for user_id, target in test.items():
        if target.value < 4.0:
            continue
        watched = {r.movie_id for r in model.user_ratings[user_id]}
        ranked = sorted(((model._bayesian_mean(mid)[0], model.movie_stats.get(mid, (0, 0.0))[0], mid)
                         for mid in dataset.movies if mid not in watched), reverse=True)
        rank = next((i for i, row in enumerate(ranked[:k], 1) if row[2] == target.movie_id), None)
        if rank:
            baseline_hits += 1
            baseline_ndcg += 1.0 / math.log2(rank + 1)
    if not evaluated:
        return {**selection, "evaluated_users": 0, "validation_users": validation_count,
                "selected_blend": str(selected_blend), "model_recall_at_k": 0.0,
                "model_ndcg_at_k": 0.0, "baseline_recall_at_k": 0.0, "baseline_ndcg_at_k": 0.0,
                "model_hits": 0, "baseline_hits": 0, "miss_examples": [],
                "legacy_hits": 0, "legacy_recall_at_k": 0.0, "legacy_ndcg_at_k": 0.0}
    return {
        **selection,
        "evaluated_users": evaluated,
        "validation_users": validation_count,
        "selected_blend": selected_blend,
        "legacy_hits": legacy_hits,
        "legacy_recall_at_k": legacy_hits / evaluated,
        "legacy_ndcg_at_k": legacy_ndcg / evaluated,
        "model_hits": model_hits,
        "baseline_hits": baseline_hits,
        "miss_examples": [
            f"user {user_id}: {dataset.movies[target.movie_id].title}, "
            f"rating {target.value:.1f}/5, {model.movie_stats.get(target.movie_id, (0, 0.0))[0]} training ratings"
            for user_id, target in sorted(misses,
                key=lambda pair: (model.movie_stats.get(pair[1].movie_id, (0, 0.0))[0], pair[0]))[:3]
        ],
        "model_recall_at_k": model_hits / evaluated,
        "model_ndcg_at_k": model_ndcg / evaluated,
        "baseline_recall_at_k": baseline_hits / evaluated,
        "baseline_ndcg_at_k": baseline_ndcg / evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Path to ml-latest-small-filtered CSV files")
    parser.add_argument("-k", type=int, default=10, help="Recommendation list length")
    parser.add_argument("--validation-only", action="store_true", help="Select configuration without scoring test holdouts")
    parser.add_argument("--output", type=Path, help="Write reproducible metrics as JSON")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("-k must be positive")
    metrics = evaluate(load_dataset(args.data_dir), args.k, args.validation_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Validation fusion candidates: " + json.dumps(metrics["validation_fusion"]))
    print(f"Selected item fusion weight: {metrics['selected_item_weight']}")
    if args.validation_only:
        print(f"Selected blend: {metrics['selected_blend']}")
        return
    print("Temporal validation/test (users with >=20 ratings; relevant rating >=4.0)")
    print(f"Selected collaborative/content/quality weights: {metrics['selected_blend']}")
    print(f"Validation users: {metrics['validation_users']}")
    print(f"Evaluated users: {metrics['evaluated_users']}")
    print(f"Personalized Recall@{args.k}: {metrics['model_recall_at_k']:.4f}")
    print(f"Personalized NDCG@{args.k}: {metrics['model_ndcg_at_k']:.4f}")
    print(f"Personalized hits: {metrics['model_hits']}/{metrics['evaluated_users']}")
    print(f"Previous algorithm Recall@{args.k}: {metrics['legacy_recall_at_k']:.4f}")
    print(f"Previous algorithm NDCG@{args.k}: {metrics['legacy_ndcg_at_k']:.4f}")
    print(f"Popularity Recall@{args.k}: {metrics['baseline_recall_at_k']:.4f}")
    print(f"Popularity NDCG@{args.k}: {metrics['baseline_ndcg_at_k']:.4f}")
    print(f"Popularity hits: {metrics['baseline_hits']}/{metrics['evaluated_users']}")
    print("Missed relevant examples: " + " | ".join(metrics["miss_examples"]))


if __name__ == "__main__":
    main()
