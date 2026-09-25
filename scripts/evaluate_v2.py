"""Validation-only content-fusion selection, cohort diagnostics and paired uncertainty."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movie_assistant.engine import MovieAssistant
from movie_assistant.storage import MovieDataset, load_dataset

CONTENT_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def split_data(dataset):
    grouped = dataset.user_ratings
    training, validation, testing = [], {}, {}
    for uid, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda r: (r.timestamp, r.movie_id))
        if len(ordered) >= 20:
            training.extend(ordered[:-2])
            validation[uid], testing[uid] = ordered[-2:]
        else:
            training.extend(ordered)
    return training, validation, testing


def wilson(hits, n):
    if not n:
        return None
    z = 1.959963984540054
    p = hits / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center - margin, center + margin]


def summarize(rows, catalog_size):
    n = len(rows)
    hits = sum(row['hit'] for row in rows)
    mids = {mid for row in rows for mid in row['recommendations']}
    return {
        'users': n, 'hits': hits, 'recall': hits/n if n else None,
        'ndcg': sum(row['ndcg'] for row in rows)/n if n else None,
        'recall_wilson_95': wilson(hits, n),
        'catalog_coverage': len(mids)/catalog_size if catalog_size else 0,
        'mean_genre_diversity': sum(row['diversity'] for row in rows)/n if n else None,
        'long_tail_recommendation_share': (sum(row['tail_recs'] for row in rows)
                                          / max(1, sum(len(row['recommendations']) for row in rows))),
    }


def collect(model, holdouts, k, weights):
    results = {weight: [] for weight in weights}
    for uid, target in sorted(holdouts.items()):
        if target.value < 4:
            continue
        for weight in weights:
            recs = model.recommend(uid, limit=k, item_weight=1., content_weight=weight)
            rank = next((i for i, r in enumerate(recs, 1) if r.movie.movie_id == target.movie_id), None)
            pairs = [(set(a.movie.genres), set(b.movie.genres))
                     for i, a in enumerate(recs) for b in recs[i+1:]]
            diversity = sum(1 - len(a & b)/max(1, len(a | b)) for a, b in pairs)/max(1, len(pairs))
            results[weight].append({
                'user_id': uid, 'target': target.movie_id,
                'training_ratings': model.movie_stats.get(target.movie_id, (0, 0))[0],
                'profile_size': len(model.user_ratings[uid]),
                'hit': int(rank is not None), 'ndcg': 1/math.log2(rank+1) if rank else 0.,
                'recommendations': [r.movie.movie_id for r in recs],
                'tail_recs': sum(r.rating_count <= 5 for r in recs), 'diversity': diversity,
            })
    return results


def paired_interval(new, old, key, repeats=2000):
    if not new:
        return None
    assert [r['user_id'] for r in new] == [r['user_id'] for r in old]
    differences = [a[key] - b[key] for a, b in zip(new, old)]
    rng = random.Random(2026)
    draws = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences)
                   for _ in range(repeats))
    return [draws[int(.025*repeats)], draws[min(repeats-1, int(.975*repeats))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/evaluation_v2.json'))
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('-k', type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error('k must be positive')
    started = time.perf_counter()
    dataset = load_dataset()
    training, validation, test = split_data(dataset)
    model = MovieAssistant(MovieDataset(dataset.movies, training, []))
    validation_results = collect(model, validation, args.k, CONTENT_WEIGHTS)
    summaries = {w: summarize(rows, len(dataset.movies)) for w, rows in validation_results.items()}
    chosen = max(CONTENT_WEIGHTS, key=lambda w: (summaries[w]['ndcg'] or 0,
                                               summaries[w]['recall'] or 0, -w))
    output = {
        'k': args.k, 'selected_content_weight': chosen,
        'fixed_previous_configuration': {'blend': [.7, .1, .2], 'item_weight': 1.},
        'selection': 'Validation NDCG, then Recall, then smaller content weight. No test-based selection.',
        'validation': summaries,
        'protocol': 'Same per-user temporal split as v1; >=20 ratings; positive holdout >=4. Full unseen catalog. Tags excluded. Existing test has been inspected in earlier development, so it is not a fresh external test.',
        'long_tail_definition': 'At most 5 training ratings; discovery UI instead uses at most 10.',
    }
    print('Validation:', json.dumps(summaries), flush=True)
    print('Selected content weight:', chosen, flush=True)
    if not args.validation_only:
        model = MovieAssistant(MovieDataset(dataset.movies, training + list(validation.values()), []))
        # A content-only ablation describes the tradeoff; it cannot change selection.
        results = collect(model, test, args.k, sorted({0., chosen, 1.}))
        selected, previous = results[chosen], results[0.]
        output['test'] = {w: summarize(rows, len(dataset.movies)) for w, rows in results.items()}
        cohorts = {
            'zero_rating': lambda r: r['training_ratings'] == 0,
            'long_tail_1_to_5': lambda r: 1 <= r['training_ratings'] <= 5,
            'head_over_5': lambda r: r['training_ratings'] > 5,
            'profile_under_50': lambda r: r['profile_size'] < 50,
            'profile_50_plus': lambda r: r['profile_size'] >= 50,
        }
        output['test_cohorts'] = {name: {
            str(w): summarize([r for r in rows if predicate(r)], len(dataset.movies))
            for w, rows in results.items()} for name, predicate in cohorts.items()}
        output['paired_delta_95_bootstrap'] = {
            'recall': paired_interval(selected, previous, 'hit'),
            'ndcg': paired_interval(selected, previous, 'ndcg'),
            'seed': 2026, 'resamples': 2000,
            'note': 'Paired user resampling; users share training data, so this is descriptive uncertainty.'}
        output['per_user_test'] = results
    output['elapsed_seconds'] = round(time.perf_counter() - started, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
