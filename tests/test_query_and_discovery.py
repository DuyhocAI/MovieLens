import unittest
from http.server import BaseHTTPRequestHandler

from movie_assistant.dialogue import release_year_constraints as _release_year_constraints
from movie_assistant.engine import MovieAssistant, _tokens
from movie_assistant.models import Movie, Rating
from movie_assistant.query import plan_query
from movie_assistant.storage import MovieDataset
from movie_assistant.web import LocalHTTPServer
from scripts.evaluate_v2 import wilson, paired_interval, split_data
from test_movie_assistant import _respond, sample_assistant


class QueryAndDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.assistant = sample_assistant()

    def test_negative_only_query_still_recommends_and_excludes_genre(self):
        results = self.assistant.recommend(1, query='không hoạt hình')
        self.assertTrue(results)
        self.assertTrue(all('Animation' not in r.movie.genres for r in results))

    def test_two_requested_genres_are_both_required(self):
        results = self.assistant.search('action thriller')
        self.assertEqual([4], [mid for mid, _ in results])

    def test_explicit_or_allows_either_genre(self):
        results = self.assistant.search('hành động hoặc tình cảm')
        self.assertEqual({1, 2, 3, 4}, {mid for mid, _ in results})

    def test_negated_genre_does_not_become_a_positive_requirement(self):
        plan = plan_query('phim hành động không kinh dị', _tokens)
        self.assertEqual({'Action'}, plan.required)
        self.assertEqual({'Horror'}, plan.excluded)

    def test_bilingual_concepts_find_english_plot_without_shared_query_words(self):
        results = self.assistant.search('người ngoài hành tinh')
        self.assertTrue({1, 2, 4}.issubset({mid for mid, _ in results}))
        self.assertEqual([], self.assistant.search('người ngoài hành tinh', expand_query=False))

    def test_concept_synonyms_count_as_one_group(self):
        plan = plan_query('sinh tồn', _tokens)
        self.assertEqual(1, len(plan.groups))
        self.assertIn('stranded', plan.groups[0])

    def test_hope_alone_is_not_evidence_of_feel_good_tone(self):
        movie = Movie(8, 'The Siege', 2000, ('Horror',),
                      'Friends hope to escape a killer, but everyone dies.')
        model = MovieAssistant(MovieDataset({8: movie}, [], []))
        self.assertEqual([], model.search('phim chữa lành'))

    def test_unrated_movie_can_enter_content_ranking(self):
        dataset = self.assistant.dataset
        movies = dict(dataset.movies)
        movies[8] = Movie(8, 'Space Colony', 2010, ('Action', 'Sci-Fi'),
                          'A space pilot protects a colony from an alien invasion.')
        model = MovieAssistant(MovieDataset(movies, dataset.ratings, []))
        results = model.recommend(1, content_weight=1., lang='vi')
        ids = [r.movie.movie_id for r in results]
        self.assertIn(8, ids)
        self.assertLess(ids.index(8), ids.index(5))
        self.assertIn('chưa có lượt chấm', next(r.explanation for r in results if r.movie.movie_id == 8))

    def test_discovery_enforces_rating_ceiling(self):
        results = self.assistant.recommend(1, content_weight=.75, max_ratings=0)
        self.assertTrue(results)
        self.assertTrue(all(r.rating_count == 0 for r in results))

    def test_seed_keeps_other_query_constraints(self):
        _respond(self.assistant, 1, 'I liked Summer Wedding but no animation')
        recent = self.assistant.last_recommendations_for(1)
        self.assertTrue(recent)
        self.assertTrue(all('Animation' not in r.movie.genres for r in recent))

    def test_query_explanation_contains_actual_index_terms(self):
        result = self.assistant.recommend(1, query='người ngoài hành tinh', lang='vi')[0]
        self.assertIn('Tín hiệu từ nội dung/thẻ:', result.explanation)
        self.assertTrue('alien' in result.explanation)

    def test_exact_year_and_strict_bounds(self):
        self.assertEqual((2003, 2003), _release_year_constraints('phim năm 2003'))
        self.assertEqual((2001, 2009), _release_year_constraints('after 2000 before 2010'))
        self.assertEqual((2000, 2010), _release_year_constraints('từ năm 2000 đến năm 2010'))

    def test_year_only_query_does_not_search_year_as_plot_word(self):
        _respond(self.assistant, 1, 'phim năm 2003')
        self.assertEqual([4], [r.movie.movie_id for r in self.assistant.last_recommendations_for(1)])

    def test_invalid_content_weight_is_rejected(self):
        for weight in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                self.assistant.recommend(1, content_weight=weight)

    def test_uncertainty_handles_empty_and_paired_identical_results(self):
        self.assertIsNone(wilson(0, 0))
        low, high = wilson(5, 10)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        rows = [{'user_id': 1, 'hit': 1}, {'user_id': 2, 'hit': 0}]
        self.assertEqual([0., 0.], paired_interval(rows, rows, 'hit', repeats=100))

    def test_temporal_split_keeps_both_holdouts_out_of_training(self):
        ratings = [Rating(9, i, 4., i) for i in range(22, 0, -1)]
        ratings.append(Rating(10, 1, 5., 1))
        training, validation, testing = split_data(MovieDataset({}, ratings, []))
        self.assertEqual(21, validation[9].movie_id)
        self.assertEqual(22, testing[9].movie_id)
        self.assertTrue(all(r.movie_id <= 20 for r in training if r.user_id == 9))
        self.assertNotIn(10, validation)
        self.assertIn(ratings[-1], training)

    def test_second_server_cannot_silently_share_the_same_port(self):
        with LocalHTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler) as first:
            with self.assertRaises(OSError):
                with LocalHTTPServer(first.server_address, BaseHTTPRequestHandler):
                    pass


if __name__ == '__main__':
    unittest.main()
