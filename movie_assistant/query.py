"""Small, auditable bilingual query planner; these rules are not a language model."""

import re
import unicodedata
from dataclasses import dataclass


def normalize(text):
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(c)).replace("đ", "d")


GENRES = {
    "Action": ("action", "hanh dong"),
    "Adventure": ("adventure", "phieu luu"),
    "Animation": ("animation", "animated", "hoat hinh"),
    "Comedy": ("comedy", "comedies", "hai huoc", "hai"),
    "Crime": ("crime", "toi pham"),
    "Documentary": ("documentary", "tai lieu"),
    "Drama": ("drama", "chinh kich"),
    "Fantasy": ("fantasy", "ky ao"),
    "Horror": ("horror", "kinh di"),
    "Mystery": ("mystery", "bi an"),
    "Romance": ("romance", "romantic", "lang man", "tinh cam"),
    "Sci-Fi": ("sci fi", "sci-fi", "science fiction", "khoa hoc vien tuong"),
    "Thriller": ("thriller", "giat gan"),
    "War": ("war", "chien tranh"),
    "Western": ("western", "cao boi"),
    "Children": ("children", "thieu nhi"),
    "Musical": ("musical", "nhac kich"),
    "Film-Noir": ("film noir", "film-noir"),
}

# One concept contributes at most once per document: adding synonyms cannot
# manufacture extra concept coverage. Broad associations carry lower weights.
CONCEPTS = (
    (("feel good", "feel-good", "chua lanh", "am ap"),
     {"heartwarming": 1., "uplifting": 1., "feelgood": 1., "warmhearted": .9}),
    (("plot twist", "twist ending", "cu twist", "bat ngo", "twist"),
     {"twist": 1., "surprise": .75, "revelation": .7, "revealed": .5}),
    (("mind bending", "mind-bending", "hack nao", "xoan nao"),
     {"mindbending": 1., "surreal": .8, "dream": .65, "reality": .6}),
    (("time travel", "du hanh thoi gian", "xuyen khong"),
     {"time": 1., "future": .65, "past": .65, "timetravel": 1.}),
    (("outer space", "ngoai vu tru", "vu tru"),
     {"space": 1., "spaceship": .9, "galaxy": .8, "astronaut": .8}),
    (("alien invasion", "nguoi ngoai hanh tinh"),
     {"alien": 1., "aliens": 1., "extraterrestrial": .9}),
    (("sad ending", "bad ending", "ket thuc buon", "ket buon"),
     {"tragic": 1., "tragedy": .9, "tragically": .9}),
    (("psychological", "tam ly"),
     {"psychological": 1., "obsession": .7, "paranoia": .75, "delusion": .75}),
    (("dark", "den toi", "u am"),
     {"dark": 1., "bleak": .8, "sinister": .7, "disturbing": .7}),
    (("survival", "sinh ton"),
     {"survival": 1., "survive": .9, "stranded": .8}),
    (("revenge", "tra thu"), {"revenge": 1., "vengeance": .9, "avenge": .8}),
    (("detective", "trinh tham"), {"detective": 1., "investigation": .8, "investigate": .7}),
)
# A negation cue, then up to four filler words ("don't want to watch any ..."), then the genre.
NEGATION_CUES = (
    r"khong|dung|tranh|chan|ngan|ghet|no|not|never|without|avoid|avoiding|excluding|except|skip"
    r"|hate|dislike|don'?t|do not|doesn'?t|anything but|other than|tired of|sick of|no more|not a fan of"
)
NEGATION_FILLER = r"(?:\s+(?:want|wanna|like|need|to|see|watch|any|more|a|an|the|fan|of|phim|movies?|films?|xem|muon|them|nua))"
NEGATION = rf"(?:{NEGATION_CUES}){NEGATION_FILLER}{{0,4}}\s+"


@dataclass
class QueryPlan:
    text: str
    groups: list[dict[str, float]]
    labels: list[str]
    required: set[str]
    excluded: set[str]
    any_genre: bool = False

    def accepts(self, genres):
        values = set(genres)
        return not self.excluded.intersection(values) and (
            not self.required or (bool(self.required & values) if self.any_genre
                                  else self.required.issubset(values)))


def plan_query(text, tokenize):
    remaining = normalize(text)
    required, excluded = set(), set()
    groups, labels = [], []
    any_genre = bool(re.search(r"\b(?:or|hoac)\b", remaining))
    for genre, aliases in GENRES.items():
        pattern = "(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + ")"
        negative = rf"\b{NEGATION}{pattern}\b"
        if re.search(negative, remaining):
            excluded.add(genre)
            remaining = re.sub(negative, " ", remaining)
        if re.search(rf"\b{pattern}\b", remaining):
            required.add(genre)
            remaining = re.sub(rf"\b{pattern}\b", " ", remaining)
            groups.append({t: 1. for t in tokenize(genre.replace("-", " "))})
            labels.append(genre)
    for aliases, alternatives in CONCEPTS:
        pattern = r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
        if re.search(pattern, remaining):
            groups.append(alternatives)
            labels.append(aliases[0])
            remaining = re.sub(pattern, " ", remaining)
    # Release dates are enforced separately, never searched as plot words.
    remaining = re.sub(r"\b(?:18|19|20|21)\d{2}\b", " ", remaining)
    filler = set("please looking look something give show find recommend suggest new latest released from since after before until through nam tu sau truoc moi cho goi y toi nay nao hay hon them nua muon xem va nhung khong but not tired animated im liked loved enjoyed thich da don doesn hate dislike need wanna else giong nhu tuong".split())
    for term in sorted(set(tokenize(remaining)) - filler):
        groups.append({term: 1.})
        labels.append(term)
    return QueryPlan(text, groups, labels, required, excluded, any_genre)
