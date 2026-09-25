"""Turn one chat message into an intent, run the matching engine call, and phrase the answer.

The routing is a small set of transparent rules shared by the CLI and the HTTP API.
"""

import re
from dataclasses import dataclass, field

from . import messages
from .engine import MovieAssistant, DEFAULT_CONTENT_WEIGHT, _tokens
from .models import Recommendation
from .query import plan_query

NEIGHBOR_CUES = ("gu giống", "tương tự tôi", "giống tôi", "similar taste", "similar users",
                 "people with similar", "people like me", "users like me")
WHY_CUES = ("vì sao", "tại sao", "why", "explain")
BLIND_SPOT_CUES = ("blind spot", "blindspot", "thể loại nào", "thể loại gì", "chưa khám phá", "điểm mù",
                   "what genres", "not explored", "genre am i missing", "genres am i missing")
LIKED_CUES = ("liked", "loved", "enjoyed", "thích", "đã xem")
DISCOVERY = r"(?i)hidden gems?|lesser[- ]known|little[- ]known|ít người biết|ít lượt đánh giá"
# "movies like Aliens", "something similar to Heat", "phim giống Toy Story"
SIMILAR_TO = re.compile(r"(?i)\b(?:movies?|films?|something|anything|stuff|more|phim)\s+"
                        r"(?:like|similar to|giống(?:\s+như)?|tương tự(?:\s+như)?)\s+(.+)$")
CLAUSE_END = re.compile(r"(?i)\s*(?:[,;.!?]|\bbut\b|\bwithout\b|\bexcept\b|\bnhưng\b|\bmà\b|\bkhông\b|\bno\b|\bnot\b)")
# "why would I like Arrival?" names a movie; "why would I like that?" refers back.
WHY_TARGET = re.compile(r"(?i)\b(?:like|thích)\s+(.+?)[\s?.!]*$")
BACK_REFERENCES = {"that", "it", "this", "this one", "that one", "them", "these", "those",
                   "phim đó", "phim này", "nó", "đó", "cái đó"}
LEADING_REQUEST = re.compile(r"(?i)^(what should i watch tonight|what should i watch|recommend|suggest|"
                             r"tối nay xem gì|gợi ý phim|tôi muốn xem)\s*")


@dataclass
class Reply:
    intent: str
    language: str
    text: str
    headline: str = ""
    recommendations: list[Recommendation] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def release_year_constraints(text: str) -> tuple[int | None, int | None]:
    """Read simple year bounds such as 'từ năm 2022' or 'before 2000'."""
    year_pattern = r"(?:18|19|20|21)\d{2}"
    between = re.search(rf"(?i)\b(?:from|từ(?:\s+năm)?)\s+({year_pattern})\s+(?:to|through|until|đến(?:\s+năm)?)\s+({year_pattern})\b", text)
    if between:
        return int(between.group(1)), int(between.group(2))
    lower = re.search(
        rf"(?i)\b(?:từ(?:\s+năm)?|kể\s+từ(?:\s+năm)?|from|since|after|sau(?:\s+năm)?)\s+({year_pattern})\b",
        text,
    )
    upper = re.search(
        rf"(?i)\b(?:trước(?:\s+năm)?|before|until|through)\s+({year_pattern})\b", text)
    if lower or upper:
        return (int(lower.group(1)) + (1 if re.match(r"(?i)(?:after|sau)\b", lower.group()) else 0) if lower else None,
                int(upper.group(1)) - (1 if re.match(r"(?i)(?:before|trước)\b", upper.group()) else 0) if upper else None)
    exact = re.search(rf"(?i)\b(?:in|năm)\s+({year_pattern})\b", text)
    return (int(exact.group(1)), int(exact.group(1))) if exact else (None, None)


def movie_mentioned(assistant: MovieAssistant, text: str) -> str | None:
    """Longest catalog title that appears in the text as whole words."""
    lowered = text.casefold()
    candidates = [movie.title for movie in assistant.dataset.movies.values()
                  if re.search(r"(?<![a-z0-9])" + re.escape(movie.title.casefold()) + r"(?![a-z0-9])", lowered)]
    return max(candidates, key=len) if candidates else None


def _similar_to_title(assistant: MovieAssistant, text: str) -> tuple[str, str] | None:
    """Return (title, matched phrase) for requests like 'movies like Aliens'."""
    match = SIMILAR_TO.search(text)
    if not match:
        return None
    raw = CLAUSE_END.split(match.group(1), maxsplit=1)[0]
    phrase = text[match.start():match.start(1) + len(raw)]
    remainder = raw.strip(" \"'")
    title = movie_mentioned(assistant, remainder)
    if not title and remainder:
        # Accept "the terminator" for the catalog form "Terminator, The".
        article = re.match(r"(?i)^(the|a|an)\s+(.+)$", remainder)
        wanted = {remainder.casefold()} | ({f"{article.group(2)}, {article.group(1)}".casefold()} if article else set())
        title = next((m.title for m in assistant.dataset.movies.values() if m.title.casefold() in wanted), None)
    return (title, phrase) if title else None


def respond(assistant: MovieAssistant, user_id: int, message: str) -> Reply:
    text = message.strip()
    lang = messages.detect_language(text)
    lowered = text.casefold()
    if not text:
        return Reply("empty", lang, messages.t(lang, "empty"))
    if lowered in {":help", "help", "giúp", "?"}:
        return Reply("help", lang, messages.t(lang, "help"))

    mentioned = movie_mentioned(assistant, text)
    if any(cue in lowered for cue in NEIGHBOR_CUES):
        if not mentioned:
            return Reply("clarify", lang, messages.t(lang, "ask_title"))
        movie = assistant.find_movie(mentioned)
        data = assistant.neighbor_opinion(user_id, movie.movie_id)
        return Reply("neighbor_opinion", lang, messages.opinion_text(data, lang), details=data)
    if any(cue in lowered for cue in WHY_CUES):
        movie = assistant.find_movie(mentioned) if mentioned else None
        target = WHY_TARGET.search(text)
        if movie is None and target and target.group(1).casefold() not in BACK_REFERENCES:
            data = {"status": "not_found", "query": target.group(1)}
            return Reply("explain", lang, messages.explanation_text(data, lang), details=data)
        data = assistant.explanation_for(user_id, movie.movie_id if movie else None)
        return Reply("explain", lang, messages.explanation_text(data, lang), details=data)
    if any(cue in lowered for cue in BLIND_SPOT_CUES):
        rows = assistant.blind_spots(user_id, lang=lang)
        picks = [row["suggestion"] for row in rows if row["suggestion"]]
        details = {"blind_spots": [{key: value for key, value in row.items() if key != "suggestion"}
                                   | {"suggestion_movie_id": row["suggestion"].movie.movie_id if row["suggestion"] else None}
                                   for row in rows]}
        return Reply("blind_spots", lang, messages.blind_spot_text(rows, lang), recommendations=picks, details=details)
    return _recommend(assistant, user_id, text, lowered, lang, mentioned)


def _recommend(assistant: MovieAssistant, user_id: int, text: str, lowered: str, lang: str,
               mentioned: str | None) -> Reply:
    query = LEADING_REQUEST.sub("", text).strip(" \t\r\n.,!?;:")
    seed = None
    similar = _similar_to_title(assistant, text)
    if similar:
        seed = assistant.find_movie(similar[0])
        query = text.replace(similar[1], " ")
    elif mentioned and any(cue in lowered for cue in LIKED_CUES):
        seed = assistant.find_movie(mentioned)
        query = re.sub(re.escape(mentioned), " ", text, flags=re.IGNORECASE)
    min_year, max_year = release_year_constraints(text)
    discovery = bool(re.search(DISCOVERY, query))
    if discovery:
        query = re.sub(DISCOVERY, " ", query)
    plan = plan_query(query, _tokens)
    recs = assistant.recommend(user_id, limit=5, exclude_genres=plan_query(text, _tokens).excluded,
                               query=query, seed_movie_ids={seed.movie_id} if seed else set(),
                               min_year=min_year, max_year=max_year,
                               content_weight=0.75 if discovery else DEFAULT_CONTENT_WEIGHT,
                               max_ratings=10 if discovery else None, lang=lang)
    details = {"seed_movie_id": seed.movie_id if seed else None, "query": query.strip() or None,
               "excluded_genres": sorted(plan.excluded), "required_genres": sorted(plan.required),
               "min_year": min_year, "max_year": max_year, "discovery": discovery}
    if not recs:
        first_year, last_year = assistant.year_bounds
        if min_year is not None and last_year is not None and min_year > last_year:
            answer = messages.t(lang, "year_unavailable", first=first_year, last=last_year, year=min_year)
        elif max_year is not None and first_year is not None and max_year < first_year:
            answer = messages.t(lang, "year_unavailable", first=first_year, last=last_year, year=max_year)
        else:
            answer = messages.t(lang, "no_match")
        return Reply("recommend", lang, answer, headline=answer, details=details)
    if seed:
        headline = messages.t(lang, "headline_seed", title=messages.display_title(seed.title))
    elif plan.groups:
        headline = messages.t(lang, "headline_query")
    else:
        headline = messages.t(lang, "headline_history")
    if discovery:
        headline += "\n" + messages.t(lang, "discovery_note")
    lines = [headline]
    for index, rec in enumerate(recs, 1):
        year = f" ({rec.movie.year})" if rec.movie.year else ""
        reason = messages.recommendation_text(rec.movie.title, rec.evidence, lang, with_title=False)
        lines.append(f"{index}. {messages.display_title(rec.movie.title)}{year} — "
                     f"{', '.join(rec.movie.genres)}. {reason[:1].upper()}{reason[1:]}")
    return Reply("recommend", lang, "\n".join(lines), headline=headline, recommendations=recs, details=details)
