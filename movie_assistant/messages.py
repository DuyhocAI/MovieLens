"""User-facing wording in English and Vietnamese.

The engine returns structured evidence; this module only turns it into text,
so a follow-up question can be answered in a different language than the
turn that produced the recommendation.
"""

import re
import unicodedata

# Letters/marks that occur in Vietnamese but not in French, Spanish or German titles.
_VIETNAMESE_MARKS = {"̛", "̉", "̣", "̆"}  # horn, hook, dot below, breve
_VIETNAMESE_WORDS = {"phim", "xem", "toi", "khong", "nhung", "muon", "nao", "gi", "minh", "giup", "goi"}


def detect_language(text: str) -> str:
    """Return "vi" for Vietnamese input and "en" otherwise."""
    decomposed = unicodedata.normalize("NFD", text)
    if "đ" in text.casefold() or any(char in _VIETNAMESE_MARKS for char in decomposed):
        return "vi"
    ascii_text = "".join(c for c in decomposed.casefold() if not unicodedata.combining(c))
    return "vi" if set(re.findall(r"[a-z]+", ascii_text)) & _VIETNAMESE_WORDS else "en"


TEXT = {
    "en": {
        "help": ("Try: 'What should I watch tonight?', 'I want a dark psychological thriller with a twist', "
                 "'What do people with similar taste to mine think about Inception?', 'Why would I like that?', "
                 "'Movies like Aliens but no horror', 'What's my blind spot?'. Use :user 15 to switch profile."),
        "empty": "Type a question or :help.",
        "ask_title": "Which movie should I check with viewers who share your taste?",
        "movie_not_found": "I could not find '{title}' in the dataset.",
        "no_context": "There is no recent recommendation to explain yet. Ask 'What should I watch tonight?' first.",
        "headline_history": "Based on your rating history:",
        "headline_query": "Picks that match your request:",
        "headline_seed": "Since you mentioned {title}, these fit it and your history:",
        "discovery_note": "Discovery mode: only films with at most 10 ratings in this dataset; content-led, so the rating evidence is thin.",
        "year_unavailable": ("I can't recommend films from that period: the dataset only covers {first}–{last}, "
                             "so nothing released in {year} is available."),
        "no_match": "No movie in the dataset matches those conditions.",
        "why_prefix": "You might like {title} because {reason}",
        # Recommendation evidence
        "shared_genres": "you rated other {genres} films 4/5 or higher",
        "seed_genres": "shares {genres} with {seed}",
        "seed_terms": "plot keywords overlap with {seed}: {terms}",
        "profile_content": "its plot resembles films you rated highly",
        "history_content": "its plot is close to films you rated positively",
        "query_match": "matches your request ‘{query}’",
        "neighbors": "{count} viewers with similar taste rated it {mean:.2f}/5 on average (weighted)",
        "neighbors_one": "1 viewer with similar taste rated it {mean:.2f}/5",
        "neighbors_lukewarm": "; that is lukewarm, so it ranks mainly on co-liked films and plot similarity",
        "stats": "{count} ratings overall, average {average:.2f}/5",
        "no_stats": "no ratings in the dataset yet",
        "query_terms": " Plot/tag evidence: {terms}.",
        "tone_caveat": " These keywords only hint at tone; they do not confirm the mood or the ending.",
        "item_link": " {support} viewers rated both this and {source} (which you liked) 3.5/5 or higher.",
        "item_link_one": " 1 viewer rated both this and {source} (which you liked) 3.5/5 or higher.",
        "item_link_thin": " That link rests on few viewers.",
        "low_confidence": " Low confidence: little rating data for this film or this profile.",
        # Explanation of a named movie
        "already_rated": "You already rated {title} {rating:.1f}/5, so it is in your history rather than a new suggestion.",
        "explain_genres": "{genres} also appear among films you rated 4/5 or higher",
        "explain_no_genres": "none of its genres appear among films you rated 4/5 or higher",
        "explain_content": "; plot similarity to your liked films is {score:.2f}",
        "movie_stats": "{genres}; {count} ratings, average {average:.2f}/5 (smoothed {bayes:.2f}){tags}.",
        "movie_no_stats": "{genres}; no ratings in the dataset{tags}.",
        "tags": "; tags: {tags}",
        "no_genre": "no genre listed",
        # Taste-neighbour opinion
        "opinion_none": ("None of your {pool} closest taste neighbours has rated {title}. "
                         "Across all viewers: {overall}."),
        "opinion": ("Among your {pool} closest taste neighbours, {count} rated {title}: weighted average "
                    "{mean:.2f}/5 ({liked} gave 4+, {disliked} gave 2.5 or less). Across all viewers: {overall}, "
                    "so your taste group rates it {comparison}."),
        "overall": "{count} ratings, average {average:.2f}/5",
        "overall_none": "no ratings",
        "higher": "higher (+{diff:.2f})",
        "lower": "lower ({diff:.2f})",
        "same": "about the same",
        "opinion_examples": " Closest raters: {examples}.",
        "opinion_example": "user {user} (similarity {similarity:.2f}, {common} films in common) gave {rating:.1f}/5",
        "opinion_yours": " You rated it {rating:.1f}/5 yourself.",
        "opinion_no_neighbors": "Your rating history overlaps too little with other viewers to find taste neighbours.",
        # Blind spots
        "blind_none": "I don't see any genre with little data in your history.",
        "blind_intro": "Genres you have barely explored: {rows}.",
        "blind_row": "{genre} ({count} ratings, average {average:.1f}/5)",
        "blind_row_one": "{genre} (1 rating, {average:.1f}/5)",
        "blind_row_zero": "{genre} (0 ratings)",
        "blind_caveat": " A low count means there is not enough data to infer a preference, not that you dislike the genre.",
        "blind_try": " To try one: {items}.",
    },
    "vi": {
        "help": ("Ví dụ: 'Tối nay xem gì?', 'phim giật gân tâm lý đen tối có cú twist', "
                 "'người có gu giống tôi nghĩ gì về Inception?', 'vì sao tôi sẽ thích phim đó?', "
                 "'phim giống Aliens nhưng không kinh dị', 'thể loại nào tôi chưa khám phá?' hoặc :user 15 để đổi hồ sơ."),
        "empty": "Hãy nhập câu hỏi hoặc :help.",
        "ask_title": "Bạn muốn xem đánh giá của người cùng gu về phim nào?",
        "movie_not_found": "Mình không tìm thấy phim '{title}' trong dữ liệu.",
        "no_context": "Chưa có đề xuất gần đây để giải thích. Hãy hỏi 'Tối nay xem gì?' trước.",
        "headline_history": "Dựa trên lịch sử xem của bạn:",
        "headline_query": "Một vài lựa chọn phù hợp:",
        "headline_seed": "Vì bạn nhắc tới {title}, đây là các phim hợp với phim đó và lịch sử của bạn:",
        "discovery_note": "Khám phá phim có tối đa 10 lượt rating trong kho; ưu tiên nội dung, bằng chứng rating còn ít.",
        "year_unavailable": ("Mình chưa thể gợi ý phim theo mốc năm đó: dữ liệu hiện chỉ có phim từ {first} đến {last}, "
                             "nên không có phim từ {year} phù hợp."),
        "no_match": "Không tìm thấy phim phù hợp với điều kiện đó trong dữ liệu.",
        "why_prefix": "Bạn có thể thích {title} vì {reason}",
        "shared_genres": "thể loại {genres} xuất hiện trong các phim bạn chấm từ 4/5",
        "seed_genres": "cùng thể loại {genres} với {seed}",
        "seed_terms": "từ khóa nội dung trùng với {seed}: {terms}",
        "profile_content": "tóm tắt nội dung có tín hiệu tương đồng với hồ sơ phim bạn thích",
        "history_content": "tóm tắt nội dung gần với lịch sử các phim bạn chấm tích cực",
        "query_match": "được tìm thấy theo mô tả ‘{query}’",
        "neighbors": "{count} người dùng có gu tương tự đã chấm, trung bình có trọng số {mean:.2f}/5",
        "neighbors_one": "1 người dùng có gu tương tự đã chấm {mean:.2f}/5",
        "neighbors_lukewarm": "; mức này chưa cao nên phim được xếp chủ yếu nhờ liên kết phim cùng được thích và nội dung",
        "stats": "{count} lượt chấm toàn bộ, trung bình {average:.2f}/5",
        "no_stats": "chưa có lượt chấm trong dữ liệu",
        "query_terms": " Tín hiệu từ nội dung/thẻ: {terms}.",
        "tone_caveat": " Các từ khóa này chỉ gợi ý sắc thái, chưa xác nhận cảm xúc hay kết thúc phim.",
        "item_link": " Có {support} người cùng chấm từ 3.5/5 cho phim này và {source}, phim bạn thích.",
        "item_link_one": " Có 1 người cùng chấm từ 3.5/5 cho phim này và {source}, phim bạn thích.",
        "item_link_thin": " Bằng chứng liên kết giữa hai phim còn ít.",
        "low_confidence": " Độ chắc chắn thấp do ít dữ liệu rating của phim hoặc hồ sơ người dùng.",
        "already_rated": "Bạn đã chấm {title} {rating:.1f}/5; phim này nằm trong lịch sử nên không phải đề xuất mới.",
        "explain_genres": "thể loại {genres} xuất hiện ở phim bạn chấm từ 4/5",
        "explain_no_genres": "chưa thấy thể loại trùng với nhóm phim bạn chấm từ 4/5",
        "explain_content": "; độ tương đồng nội dung là {score:.2f}",
        "movie_stats": "{genres}; {count} lượt chấm, trung bình {average:.2f}/5 (điểm điều chỉnh {bayes:.2f}){tags}.",
        "movie_no_stats": "{genres}; chưa có lượt chấm trong dữ liệu{tags}.",
        "tags": "; thẻ: {tags}",
        "no_genre": "chưa phân loại",
        "opinion_none": ("Trong {pool} người có gu gần bạn nhất, chưa ai chấm {title}. "
                         "Trên toàn bộ người dùng: {overall}."),
        "opinion": ("Trong {pool} người có gu gần bạn nhất, {count} người đã chấm {title}: trung bình có trọng số "
                    "{mean:.2f}/5 ({liked} người chấm từ 4, {disliked} người chấm từ 2.5 trở xuống). "
                    "Trên toàn bộ người dùng: {overall}, nên nhóm cùng gu chấm {comparison}."),
        "overall": "{count} lượt chấm, trung bình {average:.2f}/5",
        "overall_none": "chưa có lượt chấm",
        "higher": "cao hơn (+{diff:.2f})",
        "lower": "thấp hơn ({diff:.2f})",
        "same": "tương đương",
        "opinion_examples": " Người gần nhất: {examples}.",
        "opinion_example": "user {user} (độ tương đồng {similarity:.2f}, {common} phim chung) chấm {rating:.1f}/5",
        "opinion_yours": " Bạn đã tự chấm {rating:.1f}/5.",
        "opinion_no_neighbors": "Lịch sử của bạn trùng quá ít với người khác để tìm người cùng gu.",
        "blind_none": "Chưa thấy thể loại nào có ít dữ liệu trong lịch sử của bạn.",
        "blind_intro": "Thể loại còn ít dữ liệu trong lịch sử của bạn: {rows}.",
        "blind_row": "{genre} ({count} lượt, trung bình {average:.1f}/5)",
        "blind_row_one": "{genre} (1 lượt, {average:.1f}/5)",
        "blind_row_zero": "{genre} (0 lượt)",
        "blind_caveat": " Số lượt thấp nghĩa là chưa đủ dữ liệu để suy ra sở thích, không có nghĩa là bạn không thích.",
        "blind_try": " Gợi ý để thử: {items}.",
    },
}


def t(lang: str, key: str, **values) -> str:
    return TEXT.get(lang, TEXT["en"])[key].format(**values)


_TRAILING_ARTICLE = re.compile(r"^(?P<name>.+?), (?P<article>The|A|An)(?P<rest>\s+\(.*\))?$")


def display_title(title: str) -> str:
    """MovieLens stores 'Godfather, The'; show 'The Godfather'."""
    match = _TRAILING_ARTICLE.match(title)
    return f"{match['article']} {match['name']}{match['rest'] or ''}" if match else title


def recommendation_text(title: str, evidence: dict, lang: str, with_title: bool = True) -> str:
    """Render the evidence of one recommendation as a sentence."""
    parts = []
    if evidence.get("shared_genres"):
        parts.append(t(lang, "shared_genres", genres=", ".join(evidence["shared_genres"][:3])))
    for seed in evidence.get("seed_matches", [])[:2]:
        if seed["shared_genres"]:
            parts.append(t(lang, "seed_genres", genres=", ".join(seed["shared_genres"][:3]), seed=display_title(seed["title"])))
        elif seed["shared_terms"]:
            parts.append(t(lang, "seed_terms", seed=display_title(seed["title"]), terms=", ".join(seed["shared_terms"][:2])))
    if evidence.get("content_signal"):
        parts.append(t(lang, evidence["content_signal"] + "_content"))
    if evidence.get("query"):
        parts.append(t(lang, "query_match", query=evidence["query"]))
    neighbors = evidence.get("neighbors")
    if neighbors:
        key = "neighbors_one" if neighbors["count"] == 1 else "neighbors"
        text = t(lang, key, count=neighbors["count"], mean=neighbors["weighted_mean"])
        if neighbors["weighted_mean"] < 3.5:
            text += t(lang, "neighbors_lukewarm")
        parts.append(text)
    if evidence.get("rating_count"):
        parts.append(t(lang, "stats", count=evidence["rating_count"], average=evidence["average_rating"]))
    else:
        parts.append(t(lang, "no_stats"))
    sentence = "; ".join(parts) + "."
    if evidence.get("query_terms"):
        sentence += t(lang, "query_terms", terms="; ".join(
            f"{row['concept']}: {', '.join(row['terms'][:2])}" for row in evidence["query_terms"][:3]))
    if evidence.get("tone_caveat"):
        sentence += t(lang, "tone_caveat")
    link = evidence.get("item_link")
    if link:
        sentence += t(lang, "item_link_one" if link["support"] == 1 else "item_link",
                      support=link["support"], source=display_title(link["source_title"]))
        if link["support"] < 5:
            sentence += t(lang, "item_link_thin")
    if evidence.get("low_confidence"):
        sentence += t(lang, "low_confidence")
    return f"{display_title(title)}: {sentence}" if with_title else sentence


def movie_stats_text(stats: dict, lang: str) -> str:
    genres = ", ".join(stats["genres"][:4]) or t(lang, "no_genre")
    tags = t(lang, "tags", tags=", ".join(stats["tags"][:4])) if stats["tags"] else ""
    if stats["rating_count"]:
        return t(lang, "movie_stats", genres=genres, count=stats["rating_count"],
                 average=stats["average_rating"], bayes=stats["smoothed_rating"], tags=tags)
    return t(lang, "movie_no_stats", genres=genres, tags=tags)


def explanation_text(data: dict, lang: str) -> str:
    status = data["status"]
    if status == "no_context":
        return t(lang, "no_context")
    if status == "not_found":
        return t(lang, "movie_not_found", title=data["query"])
    title = display_title(data["movie"]["title"])
    if status == "recommended":
        reason = recommendation_text(title, data["evidence"], lang, with_title=False)
        return t(lang, "why_prefix", title=title, reason=reason)
    if status == "already_rated":
        return t(lang, "already_rated", title=title, rating=data["your_rating"])
    reason = (t(lang, "explain_genres", genres=", ".join(data["shared_genres"][:3]))
              if data["shared_genres"] else t(lang, "explain_no_genres"))
    if data["content_similarity"] >= 0.02:
        reason += t(lang, "explain_content", score=data["content_similarity"])
    return f"{title}: {reason}. " + movie_stats_text(data["stats"], lang)


def opinion_text(data: dict, lang: str) -> str:
    if data["status"] == "not_found":
        return t(lang, "movie_not_found", title=data["query"])
    if not data["neighbor_pool"]:
        return t(lang, "opinion_no_neighbors")
    title = display_title(data["movie"]["title"])
    overall = data["overall"]
    overall_text = (t(lang, "overall", count=overall["count"], average=overall["average"])
                    if overall["count"] else t(lang, "overall_none"))
    if not data["raters"]:
        text = t(lang, "opinion_none", pool=data["neighbor_pool"], title=title, overall=overall_text)
    else:
        diff = data["weighted_mean"] - overall["average"] if overall["count"] else 0.0
        comparison = (t(lang, "higher", diff=diff) if diff >= 0.25 else
                      t(lang, "lower", diff=diff) if diff <= -0.25 else t(lang, "same"))
        text = t(lang, "opinion", pool=data["neighbor_pool"], count=len(data["raters"]), title=title,
                 mean=data["weighted_mean"], liked=data["liked"], disliked=data["disliked"],
                 overall=overall_text, comparison=comparison)
        text += t(lang, "opinion_examples", examples=", ".join(
            t(lang, "opinion_example", user=r["user_id"], similarity=r["similarity"],
              common=r["common_movies"], rating=r["rating"]) for r in data["raters"][:3]))
    if data["your_rating"] is not None:
        text += t(lang, "opinion_yours", rating=data["your_rating"])
    return text


def blind_spot_text(rows: list[dict], lang: str) -> str:
    if not rows:
        return t(lang, "blind_none")
    keys = {0: "blind_row_zero", 1: "blind_row_one"}
    listed = "; ".join(t(lang, keys.get(row["count"], "blind_row"), genre=row["genre"],
                         count=row["count"], average=row["average"]) for row in rows)
    text = t(lang, "blind_intro", rows=listed) + t(lang, "blind_caveat")
    suggestions = [f"{row['genre']} → {display_title(row['suggestion'].movie.title)}"
                   + (f" ({row['suggestion'].movie.year})" if row["suggestion"].movie.year else "")
                   for row in rows if row.get("suggestion")]
    if suggestions:
        text += t(lang, "blind_try", items="; ".join(suggestions))
    return text
