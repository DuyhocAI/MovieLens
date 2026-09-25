import unittest

from movie_assistant.engine import MovieAssistant
from movie_assistant.models import Movie, Rating, Tag
from movie_assistant.item_similarity import ItemSimilarity
from movie_assistant.storage import MovieDataset
from movie_assistant.dialogue import movie_mentioned as _movie_mentioned
from movie_assistant.dialogue import release_year_constraints as _release_year_constraints
from movie_assistant.dialogue import respond
from movie_assistant.messages import explanation_text, opinion_text
from scripts.evaluate_recommender import _compare_configurations, _rank_metrics


def _respond(assistant, user_id, message):
    return respond(assistant, user_id, message).text


def sample_assistant() -> MovieAssistant:
    movies = {
        1: Movie(1, "Space Pilot", 2000, ("Action", "Sci-Fi"), "A space pilot fights alien ships across distant galaxies."),
        2: Movie(2, "Robot Frontier", 2001, ("Action", "Sci-Fi"), "A robot hero protects a colony from an alien invasion."),
        3: Movie(3, "Summer Wedding", 2002, ("Romance",), "Two people fall in love and plan a wedding."),
        4: Movie(4, "Star Quest", 2003, ("Action", "Sci-Fi", "Thriller"), "A space crew battles aliens to save a distant planet."),
        5: Movie(5, "Toy Friends", 2004, ("Animation", "Children"), "Toys become friends on a magical adventure."),
        6: Movie(6, "City Detectives", 2005, ("Comedy", "Crime"), "Two detectives solve a funny city mystery."),
        7: Movie(7, "O", 2006, ("Drama",), "A dramatic story about a family."),
    }
    pairs = {
        1: {1: 5.0, 2: 4.0, 3: 1.0, 6: 4.0},
        2: {1: 4.0, 2: 4.5, 3: 1.0, 4: 5.0, 6: 4.0},
        3: {1: 4.5, 2: 5.0, 3: 1.5, 4: 4.5, 6: 4.0},
        4: {1: 1.0, 2: 1.0, 3: 5.0, 5: 5.0, 6: 1.0},
    }
    ratings = [Rating(user_id, movie_id, score, user_id * 100 + movie_id)
               for user_id, values in pairs.items() for movie_id, score in values.items()]
    return MovieAssistant(MovieDataset(movies, ratings, []))


class MovieAssistantTests(unittest.TestCase):
    def setUp(self):
        self.assistant = sample_assistant()

    def test_recommendations_exclude_watched_movies(self):
        watched = {r.movie_id for r in self.assistant.user_ratings[1]}
        results = self.assistant.recommend(1, limit=3)
        self.assertTrue(results)
        self.assertTrue(all(item.movie.movie_id not in watched for item in results))

    def test_genre_constraint_is_applied(self):
        results = self.assistant.recommend(1, limit=5, exclude_genres={"Action"})
        self.assertTrue(all("Action" not in item.movie.genres for item in results))

    def test_plot_search_finds_matching_movie(self):
        results = self.assistant.search("space crew battles aliens", limit=3)
        self.assertIn(4, [movie_id for movie_id, _ in results])

    def test_genre_keyword_search_returns_movies_in_that_genre(self):
        results = self.assistant.search("psychological thriller with a twist", limit=10)
        self.assertTrue(results)
        self.assertTrue(all("Thriller" in self.assistant.dataset.movies[mid].genres for mid, _ in results))

    def test_recommendation_query_honors_explicit_genre(self):
        results = self.assistant.recommend(1, limit=5, query="psychological thriller with a twist")
        self.assertTrue(results)
        self.assertTrue(all("Thriller" in item.movie.genres for item in results))

    def test_similar_user_answer_uses_rating_evidence(self):
        data = self.assistant.neighbor_opinion(1, 4)
        self.assertEqual({2, 3}, {row["user_id"] for row in data["raters"]})
        answer = opinion_text(data, "en")
        self.assertIn("/5", answer)
        self.assertIn("user", answer)

    def test_explanation_followup_uses_last_recommendation(self):
        first = self.assistant.recommend(1, limit=1)[0]
        answer = explanation_text(self.assistant.explanation_for(1), "vi")
        self.assertIn(first.movie.title, answer)
        self.assertIn("lượt chấm", answer)

    def test_unknown_user_is_reported(self):
        with self.assertRaises(ValueError):
            self.assistant.recommend(999)

    def test_explicit_liked_movie_can_override_a_low_historical_rating(self):
        historical = next(r.value for r in self.assistant.user_ratings[1] if r.movie_id == 3)
        self.assertEqual(1.0, historical)
        profile = self.assistant._content_profile(1, {3})
        self.assertIn("wedding", profile)

    def test_follow_up_why_refers_to_last_recommendation(self):
        _respond(self.assistant, 1, "What should I watch tonight?")
        first_title = self.assistant.last_recommendations_for(1)[0].movie.title
        answer = _respond(self.assistant, 1, "Why would I like that?")
        self.assertIn(first_title, answer)

    def test_short_title_is_not_matched_inside_an_ordinary_word(self):
        self.assertIsNone(_movie_mentioned(self.assistant, "Why would I like that?"))

    def test_english_blind_spot_intent_is_recognized(self):
        answer = _respond(self.assistant, 1, "What genres have I not explored?")
        self.assertIn("genres you have barely explored", answer.casefold())

    def test_unmatched_query_does_not_fall_back_to_unrelated_popular_movies(self):
        self.assistant.recommend(1)
        self.assertEqual([], self.assistant.recommend(1, query="xyzunmatchedtoken"))
        self.assertEqual("no_context", self.assistant.explanation_for(1)["status"])

    def test_explicit_seed_is_not_recommended_back_to_user(self):
        results = self.assistant.recommend(1, seed_movie_ids={4})
        self.assertNotIn(4, [rec.movie.movie_id for rec in results])

    def test_explanation_does_not_leak_previous_users_recommendations(self):
        self.assistant.recommend(1)
        self.assertEqual("no_context", self.assistant.explanation_for(4)["status"])

    def test_invalid_ranking_weights_are_rejected(self):
        for blend in ((0.5, 0.5), (float("nan"), 0.1, 0.9), (-1, 1, 1)):
            with self.assertRaises(ValueError):
                self.assistant.recommend(1, blend=blend)
        with self.assertRaises(ValueError):
            self.assistant.recommend(1, item_weight=1.1)

    def test_full_text_retrieval_keeps_terms_beyond_truncated_content_vectors(self):
        plot = " ".join(f"word{index} word{index}" for index in range(100)) + " zeppelin"
        movie = Movie(10, "Long Story", 2000, (), plot)
        assistant = MovieAssistant(MovieDataset({10: movie}, [], []))
        self.assertNotIn("zeppelin", assistant.features[10])
        self.assertEqual(10, assistant.search("zeppelin")[0][0])

    def test_tag_only_query_is_searchable(self):
        dataset = self.assistant.dataset
        assistant = MovieAssistant(MovieDataset(dataset.movies, dataset.ratings,
                                                [Tag(1, 4, "mindbending", 1)]))
        self.assertEqual(4, assistant.search("mindbending")[0][0])

    def test_vietnamese_dark_thriller_query_matches_english(self):
        self.assertEqual(self.assistant.search("dark psychological thriller"),
                         self.assistant.search("giật gân tâm lý đen tối"))

    def test_item_link_uses_positive_co_ratings_and_reports_support(self):
        ratings = {
            1: [Rating(1, 1, 5, 1), Rating(1, 2, 5, 2), Rating(1, 3, 1, 3)],
            2: [Rating(2, 1, 4, 1), Rating(2, 2, 4, 2)],
        }
        index = ItemSimilarity(ratings)
        scores, evidence = index.score([Rating(9, 1, 5, 1)], set())
        self.assertGreater(scores[2], 0)
        self.assertEqual((1, 2), (evidence[2][0], evidence[2][2]))
        self.assertNotIn(3, scores)
        disliked_scores, _ = index.score([Rating(9, 1, 1, 1)], set())
        self.assertNotIn(2, disliked_scores)

    def test_query_relevance_beats_unrelated_movie_in_same_genre(self):
        movies = dict(self.assistant.dataset.movies)
        movies[8] = Movie(8, "Alien Crew", 2008, ("Thriller",), "An alien crew in space.")
        movies[9] = Movie(9, "Dinner", 2009, ("Thriller",), "A wedding dinner in a city.")
        assistant = MovieAssistant(MovieDataset(movies, self.assistant.dataset.ratings, []))
        results = assistant.recommend(1, query="alien crew space thriller")
        ranked = [rec.movie.movie_id for rec in results]
        self.assertLess(ranked.index(8), ranked.index(9))

    def test_batched_evaluation_matches_individual_configurations(self):
        holdouts = {1: Rating(1, 4, 5, 999), 4: Rating(4, 4, 4, 999)}
        configs = [((0.7, 0.1, 0.2), weight) for weight in (0.0, 0.5, 1.0)]
        count, totals = _compare_configurations(self.assistant, holdouts, 3, configs)
        for config, (hits, ndcg) in zip(configs, totals):
            individual = _rank_metrics(self.assistant, holdouts, 3, config[0], item_weight=config[1])
            self.assertEqual((count, hits, ndcg), individual[:3])

    def test_release_year_language_is_parsed_and_applied_as_a_hard_filter(self):
        question = "Muốn xem phim kinh dị Châu Á, bad ending, phim mới từ năm 2022"
        self.assertEqual((2022, None), _release_year_constraints(question))
        answer = _respond(self.assistant, 1, question)
        self.assertIn("dữ liệu hiện chỉ có phim từ 2000 đến 2006", answer)
        self.assertNotIn("199", answer)

    def test_explicit_year_bounds_filter_the_candidate_catalog(self):
        results = self.assistant.recommend(1, min_year=2005)
        self.assertTrue(results)
        self.assertTrue(all(result.movie.year is not None and result.movie.year >= 2005
                            for result in results))
        self.assertEqual([], self.assistant.recommend(1, min_year=2022))

    def test_current_liked_seed_can_promote_unrated_content_match(self):
        movies = dict(self.assistant.dataset.movies)
        movies[8] = Movie(8, "Another Wedding", 2008, ("Romance",),
                          "Two people fall in love and plan a wedding.")
        assistant = MovieAssistant(MovieDataset(movies, self.assistant.dataset.ratings, []))
        # User 1 historically disliked the seed; their new explicit intent wins.
        results = assistant.recommend(1, seed_movie_ids={3}, item_weight=1.0)
        self.assertEqual(8, results[0].movie.movie_id)


if __name__ == "__main__":
    unittest.main()
