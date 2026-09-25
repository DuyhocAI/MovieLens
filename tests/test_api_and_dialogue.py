import json
import re
import threading
import unittest
from http.client import HTTPConnection

from movie_assistant.api_spec import SPEC
from movie_assistant.dialogue import respond
from movie_assistant.engine import MovieAssistant, _tokens
from movie_assistant.messages import detect_language, display_title
from movie_assistant.models import Movie, Rating
from movie_assistant.query import plan_query
from movie_assistant.storage import MovieDataset
from movie_assistant.web import MovieWebHandler, build_server
from test_movie_assistant import sample_assistant


def assistant_with(extra_movies):
    base = sample_assistant().dataset
    movies = dict(base.movies)
    movies.update({movie.movie_id: movie for movie in extra_movies})
    return MovieAssistant(MovieDataset(movies, base.ratings, []))


class NegationTests(unittest.TestCase):
    def test_english_contractions_exclude_the_genre(self):
        for text in ("I don't want horror, give me sci-fi", "I don’t want horror, give me sci-fi",
                     "I dont want horror, give me sci-fi", "I do not want horror, give me sci-fi"):
            plan = plan_query(text, _tokens)
            self.assertEqual({"Horror"}, plan.excluded, text)
            self.assertEqual({"Sci-Fi"}, plan.required, text)
            self.assertNotIn("don", plan.labels, text)

    def test_other_negative_phrasings(self):
        for text in ("I hate horror", "not a fan of horror", "no more horror movies",
                     "I don't want to watch any horror", "tôi ghét phim kinh dị"):
            plan = plan_query(text, _tokens)
            self.assertEqual({"Horror"}, plan.excluded, text)
            self.assertFalse(plan.required, text)

    def test_negated_genre_never_appears_in_chat_results(self):
        model = assistant_with([Movie(8, "Space Terror", 2007, ("Horror", "Sci-Fi"),
                                      "A space crew is hunted by aliens.")])
        reply = respond(model, 1, "I don't want horror, give me sci-fi")
        self.assertTrue(reply.recommendations)
        self.assertTrue(all("Horror" not in r.movie.genres for r in reply.recommendations))


class DialogueTests(unittest.TestCase):
    def setUp(self):
        self.assistant = sample_assistant()

    def test_movies_like_title_becomes_a_seed(self):
        reply = respond(self.assistant, 4, "movies like Star Quest")
        self.assertEqual(4, reply.details["seed_movie_id"])
        self.assertNotIn(4, [r.movie.movie_id for r in reply.recommendations])
        self.assertIn("Star Quest", reply.headline)

    def test_reply_language_follows_the_question(self):
        self.assertEqual("en", respond(self.assistant, 1, "What should I watch tonight?").language)
        vietnamese = respond(self.assistant, 1, "tối nay xem gì?")
        self.assertEqual("vi", vietnamese.language)
        self.assertIn("Dựa trên lịch sử xem", vietnamese.text)

    def test_language_detection(self):
        self.assertEqual("vi", detect_language("toi nay xem gi"))
        self.assertEqual("vi", detect_language("người cùng gu nghĩ gì về Inception?"))
        self.assertEqual("en", detect_language("What do people think of Amélie?"))

    def test_why_about_a_missing_title_does_not_explain_another_movie(self):
        respond(self.assistant, 1, "What should I watch tonight?")
        reply = respond(self.assistant, 1, "Why would I like Arrival?")
        self.assertEqual("not_found", reply.details["status"])
        self.assertIn("Arrival", reply.text)

    def test_context_is_kept_per_user(self):
        first = self.assistant.recommend(1)[0].movie.movie_id
        self.assistant.recommend(4)
        self.assertEqual(first, self.assistant.explanation_for(1)["movie"]["movie_id"])

    def test_neighbor_count_is_the_real_number_of_raters(self):
        movies = {i: Movie(i, f"Film {i}", 2000, ("Drama",), f"plot {i}") for i in range(1, 6)}
        ratings = [Rating(1, 1, 5, 1), Rating(1, 2, 4, 2), Rating(1, 3, 1, 3)]
        for user in range(2, 8):
            ratings += [Rating(user, 1, 5, 1), Rating(user, 2, 4, 2), Rating(user, 3, 1.5, 3),
                        Rating(user, 4, 4.5, 4)]
        model = MovieAssistant(MovieDataset(movies, ratings, []))
        rec = next(r for r in model.recommend(1, limit=5) if r.movie.movie_id == 4)
        self.assertEqual(6, rec.evidence["neighbors"]["count"])
        self.assertIn("6 viewers with similar taste", rec.explanation)

    def test_blind_spots_skip_imax_and_suggest_a_film(self):
        model = assistant_with([Movie(8, "Big Screen", 2008, ("IMAX", "Documentary"), "A documentary.")])
        rows = model.blind_spots(1)
        self.assertNotIn("IMAX", [row["genre"] for row in rows])
        documentary = next(row for row in rows if row["genre"] == "Documentary")
        self.assertEqual(8, documentary["suggestion"].movie.movie_id)

    def test_display_title_moves_trailing_article(self):
        self.assertEqual("The Godfather", display_title("Godfather, The"))
        self.assertEqual("Aliens", display_title("Aliens"))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = build_server(sample_assistant(), "127.0.0.1", 0, access_log=False)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None):
        connection = HTTPConnection(*self.server.server_address, timeout=10)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, json.dumps(body) if body is not None else None, headers)
        response = connection.getresponse()
        raw = response.read()
        connection.close()
        return response.status, json.loads(raw) if raw else None

    def test_health(self):
        status, data = self.call("GET", "/health")
        self.assertEqual((200, "ok", 7), (status, data["status"], data["movies"]))

    def test_recommendations_with_filters(self):
        status, data = self.call("GET", "/api/v1/users/1/recommendations?limit=3&exclude_genres=action&lang=vi")
        self.assertEqual(200, status)
        self.assertTrue(data["recommendations"])
        for rec in data["recommendations"]:
            self.assertNotIn("Action", rec["genres"])
            self.assertIn("evidence", rec)
            self.assertIsInstance(rec["rank_score"], float)

    def test_errors_are_structured(self):
        self.assertEqual((404, "not_found"), self._error("GET", "/api/v1/users/999"))
        self.assertEqual((400, "bad_request"), self._error("GET", "/api/v1/users/1/recommendations?exclude_genres=Cartoon"))
        self.assertEqual((400, "bad_request"), self._error("GET", "/api/v1/users/1/recommendations?limit=0"))
        self.assertEqual((404, "not_found"), self._error("GET", "/api/v1/nope"))
        self.assertEqual((405, "method_not_allowed"), self._error("PUT", "/api/v1/chat"))
        self.assertEqual((400, "bad_request"), self._error("POST", "/api/v1/chat", {"user_id": 1}))

    def _error(self, method, path, body=None):
        status, data = self.call(method, path, body)
        return status, data["error"]["code"]

    def test_chat_then_explain_then_reset(self):
        status, data = self.call("POST", "/api/v1/chat", {"user_id": 1, "message": "What should I watch tonight?"})
        self.assertEqual((200, "recommend", "en"), (status, data["intent"], data["language"]))
        first = data["recommendations"][0]["movie_id"]
        _, why = self.call("POST", "/api/v1/chat", {"user_id": 1, "message": "Why would I like that?"})
        self.assertEqual(first, why["details"]["movie"]["movie_id"])
        self.assertEqual(204, self.call("DELETE", "/api/v1/users/1/context")[0])
        _, after = self.call("POST", "/api/v1/chat", {"user_id": 1, "message": "Why would I like that?"})
        self.assertEqual("no_context", after["details"]["status"])

    def test_neighbor_opinion_and_movie_lookup(self):
        status, data = self.call("GET", "/api/v1/users/1/movies/4/neighbor-opinion")
        self.assertEqual(200, status)
        self.assertEqual(2, len(data["raters"]))
        status, movie = self.call("GET", "/api/v1/movies/4")
        self.assertEqual((200, "Star Quest"), (status, movie["title"]))

    def test_every_documented_path_is_routed(self):
        routes = [pattern for pattern, _ in
                  MovieWebHandler.GET_ROUTES + MovieWebHandler.POST_ROUTES + MovieWebHandler.DELETE_ROUTES]
        for path in SPEC["paths"]:
            concrete = re.sub(r"\{[^}]+\}", "1", path)
            self.assertTrue(any(re.fullmatch(pattern, concrete) for pattern in routes), path)


if __name__ == "__main__":
    unittest.main()
