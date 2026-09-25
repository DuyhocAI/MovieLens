"""Developer-authored synthetic intent probes, not a human relevance benchmark."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movie_assistant.engine import MovieAssistant
from movie_assistant.models import Movie
from movie_assistant.storage import MovieDataset


def fixture():
    rows = [
        (1, 'Second Chance', ('Drama',), 'An uplifting heartwarming story of friendship and hope.'),
        (2, 'Last Signal', ('Action', 'Sci-Fi'), 'Astronauts aboard a spaceship face an alien invasion in deep space.'),
        (3, 'Broken Mirror', ('Thriller', 'Mystery'), 'A detective investigates a psychological mystery of paranoia and delusion. A revelation changes the investigation.'),
        (4, 'Homeward', ('Animation', 'Children'), 'An uplifting heartwarming tale of friendship and hope for children.'),
        (5, 'The Crossing', ('Adventure', 'Drama'), 'A stranded explorer struggles to survive. Survival depends on finding water.'),
        (6, 'Final Letter', ('Romance', 'Drama'), 'A tragic romance ends in tragedy and grief.'),
        (7, 'Night Siege', ('Horror', 'Action'), 'A sinister zombie attack turns a dark night into disturbing horror.'),
        (8, 'City Pursuit', ('Action', 'Crime'), 'Police chase a robber through the streets during an action packed crime spree.'),
        (9, 'Yesterday Again', ('Sci-Fi', 'Comedy'), 'A scientist travels in time to change the past and future.'),
        (10, 'Wedding Weekend', ('Romance', 'Comedy'), 'Two people fall in love and plan a funny wedding.'),
    ]
    return MovieDataset({mid: Movie(mid, title, 2000, genres, plot)
                         for mid, title, genres, plot in rows}, [], [])


# These probes were authored with the rules and are intentionally reported as
# behavioral coverage. A disjoint human-judged query set remains future work.
PROBES = [
    ('phim chữa lành không hoạt hình', {1}, {4}),
    ('feel good without animation', {1}, {4}),
    ('người ngoài hành tinh', {2}, set()),
    ('alien invasion', {2}, set()),
    ('phim sinh tồn', {5}, set()),
    ('survival', {5}, set()),
    ('du hành thời gian', {9}, set()),
    ('time travel', {9}, set()),
    ('phim tình cảm kết thúc buồn', {6}, set()),
    ('romance sad ending', {6}, set()),
    ('giật gân tâm lý', {3}, set()),
    ('psychological thriller', {3}, set()),
    ('hành động không kinh dị', {2, 8}, {7}),
    ('action no horror', {2, 8}, {7}),
    ('animation', {4}, set()),
    ('xyzunmatchedtoken', set(), set(range(1, 11))),
    # Added after a review found that English contractions were not treated as negation.
    ("I don't want horror, give me action", {2, 8}, {7}),
    ('I hate horror, action please', {2, 8}, {7}),
    ('action but not a fan of horror', {2, 8}, {7}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/query_probes.json'))
    args = parser.parse_args()
    model = MovieAssistant(fixture())
    output = {'scope': __doc__, 'k': 3, 'catalog_size': 10, 'probe_count': len(PROBES)}
    for name, expand in [('legacy', False), ('expanded', True)]:
        rows = []
        for query, expected, forbidden in PROBES:
            ids = [mid for mid, _ in model.search(query, limit=3, expand_query=expand)]
            satisfied = bool(set(ids) & expected) if expected else not ids
            rows.append({'query': query, 'top_ids': ids, 'expected_any': sorted(expected),
                         'forbidden_ids': sorted(forbidden),
                         'pass': satisfied and not bool(set(ids) & forbidden)})
        output[name] = {'passed': sum(r['pass'] for r in rows), 'cases': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: output[name]['passed'] for name in ('legacy', 'expanded')}))


if __name__ == '__main__':
    main()
