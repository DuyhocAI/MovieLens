# Report: Nguyễn An Bảo Duy

## Problem Analysis

**Who are the users of this system? What do they need?**
The primary user is a viewer who identifies with a MovieLens user ID and wants to decide what to watch. They need four things:

1. Suggestions that reflect *their* history rather than global popularity.
2. A way to steer the suggestions with what they want right now ("no horror", "like Toy Story but not animated", "after 2000").
3. Answers to investigative questions that a search box cannot answer ("what do people with my taste think of Inception?", "what am I missing?").
4. A reason they can check.

The secondary user is whoever operates or evaluates the system. They need to see *why* it produced an answer.

**What makes a good movie recommendation in a conversational setting?**

- It fits the user's long-term taste.
- It respects the explicit constraints of the current turn. A "no horror" request answered with a horror film is worse than a mediocre pick.
- It is explained with facts the user can verify: which of their ratings, how many similar viewers, what those viewers gave it.
- It is honest about uncertainty, such as few ratings or a match on keywords only.
- It remembers context, so "Why would I like *that*?" refers to the right film for the right user.

**What are the key technical challenges?**

- **Sparsity on the item side.** 51% of films have fewer than 5 ratings. Collaborative filtering cannot see them. Plot text can, but lexical similarity is a weak proxy for taste.
- **Combining signals on different scales.** Predicted ratings, cosine similarities and co-occurrence counts are not comparable.
- **Understanding free text without a model.** Requests arrive in English and Vietnamese, with negation ("don't want horror") and tone ("dark", "feel-good", "twist").
- **Grounding.** Every sentence of an explanation has to come from the data, not from general movie knowledge.
- **Evaluation.** There are no relevance labels for free-text requests, and offline rating metrics only measure part of what matters.

## Approach

**How did you break down the problem?**
The system has four layers, each testable on its own:

```text
user ID + message
  → dialogue.py   intent (recommend | neighbour opinion | explain | blind spots) + constraints
                  (required/excluded genres, years, seed film, discovery), per-user context
  → engine.py     candidate filtering → user-neighbour CF, item-item CF, content profile, BM25 query match
                  → weighted reciprocal-rank fusion → evidence object per result
  → messages.py   evidence → English or Vietnamese sentences (same language as the question)
  → cli.py / web.py (JSON API + UI)
```

**What methods and tools did you choose? Why?**
Everything is written in the Python standard library, so reviewers can run it offline with no keys or downloads.

| Signal | Method | Why |
|---|---|---|
| Taste neighbours | Mean-centred Pearson on co-rated films, shrunk by `overlap/(overlap+10)`, at least 3 co-rated films, top 50 (top 20 for opinions) | Directly answers "what do people with similar taste think of X" (requirement 2) and produces checkable evidence |
| Co-liked films | Item-item cosine on positive ratings (≥ 3.5), each user weighted `1/log2(2+n_liked)` so heavy raters count less, shrunk by `support/(support+5)`, 100 neighbours per film | Strongest single signal on validation. Gives explanations like "94 viewers rated both this and The Terminator 3.5+" |
| Plot/genre profile | TF-IDF of each film (top 80 terms plus title and genres), summed over the user's liked films, cosine | The only signal that reaches unrated films |
| Free-text requests | BM25 over the full plot, with title ×3 and genre/tag ×4, plus 12 bilingual concept groups (survival, twist, psychological, …) scored with disjunction-max | Transparent, and synonyms cannot inflate how many concepts a film covers |
| Fusion | `score = 5·61·[(1−c)/(60+rank_personal) + c/(60+rank_content)]`, item weight 1.0 and content weight c = 0.25 chosen on validation | Combines rankings without calibrating raw scores |
| Quality prior | Bayesian mean with a prior worth 8 ratings | Keeps sparsely rated films from dominating on one 5-star rating |

`rank_score` only orders results. It is deliberately not shown as "x/5", because it is not a predicted rating.

**What alternatives did you consider? Why did you reject them?**

- **An LLM agent with tool calls.** The LLM would decide which engine function to call and then phrase the answer. This is the most natural fit for "reason about what to look up", and it would handle paraphrase and negation far better than rules. I did not build it in this version for three reasons:
  - it needs an external API and key, and I wanted reviewers to run everything offline;
  - non-deterministic outputs make evaluation harder;
  - an LLM tends to add knowledge that is not in the data, which requirement 3 rules out.

  Instead, the engine functions (`recommend` with filters, `neighbor_opinion`, `explanation_for`, `blind_spots`) are shaped like tools and already exposed through the API. Putting an LLM planner on top is the natural next step.
- **Matrix factorisation (SVD/ALS).** It is competitive on rating prediction, but latent factors are hard to explain and it needs numerical libraries. Item-item CF is comparably strong on data this small and explains itself.
- **Sentence embeddings for plots.** They would match "funny" or "feel-good" semantically. I rejected them because they need a model download, and I have no labelled queries to show they help (see failure 1).
- **Tag-centred retrieval.** With 2,440 tags across 5,135 films, tags are a boost, not a foundation.

### Decision Log

| Decision | Alternative considered | Why I chose this |
|----------|----------------------|-----------------|
| Deterministic rules and retrieval for language understanding; no LLM | An LLM agent that plans tool calls and writes the answer | Runs offline, every claim traces back to data, and the behaviour can be unit-tested. The cost is brittle language understanding: see failure 2, a real bug this choice caused. |
| Fuse rankings by rank, with weights chosen on validation NDCG | Blend raw scores, or pick the weight with the best test Recall | The raw scores live on different scales, and choosing on validation keeps the test set out of selection. I kept the pre-registered criterion even though the chosen weight found 2 fewer validation hits than c = 0.1. |
| Build explanations from a structured evidence object and render them afterwards | Write the explanation text during ranking | One set of facts feeds the CLI, the UI and the API. A follow-up in another language re-renders the same facts. Counts come straight from data (this fixed a bug that always said "3 similar viewers"). The cost is templated, somewhat repetitive prose. |

## Evaluation

**How do you know your system works?**
I split "works" into three questions and used a different kind of evidence for each:

1. **Does the ranking surface films the user will actually like?** A temporal hold-out with ranking metrics, compared with baselines. This is quantitative, reproducible and able to compare variants.
2. **Does it do what the user asked** (intents, constraints, language)? Behavioural probes, plus unit and API tests.
3. **Are the answers sensible and grounded?** A qualitative review of real conversations for the three suggested users, including the bad ones.

I did not rely on metrics alone because rating-based metrics say nothing about free-text requests, and there are no relevance labels for those. Qualitative review alone cannot compare variants, so I needed both.

### 1. Offline ranking

**Protocol.**

- Each user's ratings are sorted by timestamp. For users with at least 20 ratings, the second-to-last rating is the validation target and the last one is the test target.
- A target counts as relevant only if rated ≥ 4.0. That leaves **297 validation users** and **303 test users**.
- The validation model sees neither target. The test model adds the validation rating back.
- Every unseen film in the catalog is a candidate; there is no negative sampling. Tags are excluded.
- With one target per user, Recall@10 equals hit rate, and NDCG@10 = `1/log2(rank+1)` on a hit.

**Selecting the content weight on validation** (`artifacts/evaluation_v2.json`):

| Content weight c | Hits / 297 | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0.00 (previous ranking) | 35 | 0.1178 | 0.0548 |
| 0.10 | 35 | 0.1178 | 0.0584 |
| **0.25 (selected)** | **33** | **0.1111** | **0.0599** |
| 0.50 | 26 | 0.0875 | 0.0487 |
| 0.75 | 11 | 0.0370 | 0.0258 |
| 1.00 (content only) | 7 | 0.0236 | 0.0163 |

**Test:**

| Method | Hits / 303 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Bayesian popularity | 19 | 0.0627 | 0.0351 |
| User-neighbour + content + quality hybrid | 27 | 0.0891 | 0.0466 |
| + item-item CF (previous ranking) | 31 | 0.1023 | 0.0522 |
| **+ content ranking, c = 0.25 (current)** | **35** | **0.1155** | **0.0587** |
| Content only (ablation) | 7 | 0.0231 | 0.0137 |

- The current system finds almost twice as many held-out favourites as popularity.
- Against the previous ranking, it finds 4 more (Recall +12.9% relative). The paired bootstrap 95% interval for that difference (2,000 user resamples, seed 2026) is **[−0.003, +0.033]** for Recall and **[−0.005, +0.020]** for NDCG. **Both include 0, so the improvement is not established.**
- The Wilson 95% interval for Recall is [8.4%, 15.6%].
- The test set was inspected during earlier iterations, so it is not a pristine test set.

**Where it works and where it does not** (test hits by cohort):

| Cohort (training data) | Users | Previous | Current | Content only |
|---|---:|---:|---:|---:|
| Target film has 0 ratings | 6 | 0 | 0 | 0 |
| Target film has 1–5 ratings | 25 | 0 | 0 | 1 |
| Target film has > 5 ratings | 272 | 31 | 35 | 6 |
| Profile < 50 ratings | 129 | 17 | 19 | 3 |
| Profile ≥ 50 ratings | 174 | 14 | 16 | 4 |

- All the gains come from films that already have ratings. The long tail is never recovered in the default top 10.
- Catalog coverage rises only from 2.88% to 3.15%. Genre diversity (mean pairwise Jaccard distance) dips from 0.792 to 0.786.
- Content-only ranking covers 13.1% of the catalog, and half of its picks have ≤ 5 ratings, but its Recall is only 2.3%. That is a stark accuracy/coverage trade-off.

### 2. Behavioural checks

- **Intent probes** (`scripts/evaluate_queries.py`, a 10-film synthetic catalog, 19 English/Vietnamese probes covering paraphrases, negation, genre AND/OR and an unmatched query). The plain-lexical baseline passes **8/19**; the current planner passes **19/19**.
  - Three English-negation probes were added after review found the bug in failure 2. The original code returned *only the forbidden horror film* for all three, which is 16/19.
  - I wrote these probes alongside the rules, so they measure coverage of intended behaviour. They are not an independent benchmark.
- **Tests:** 59 unit, dialogue and HTTP tests (`artifacts/tests.txt`). They cover:
  - excluded genres never returned, including English contractions;
  - seeds and "movies like X";
  - real neighbour counts;
  - per-user context isolation;
  - reply language follows the question;
  - blind spots skip IMAX;
  - an unknown title in a "why" question;
  - the temporal split and uncertainty helpers;
  - the structured API errors, and that every documented route exists.
- **API smoke test on the full data** (`artifacts/api_smoke.json`): 10 real requests, with the expected 200/404/400 statuses.

### 3. Qualitative review: the brief's sample queries

All outputs below come from `artifacts/demo.json`, which holds 28 exchanges.

| Query | User 1 (action/comedy) | User 15 (sci-fi) | User 30 (18 ratings) |
|---|---|---|---|
| What should I watch tonight? | Terminator 2; Ferris Bueller's Day Off; The Godfather | Indiana Jones and the Last Crusade; Ocean's Eleven; Reservoir Dogs | Saving Private Ryan; Monty Python and the Holy Grail; Fight Club |
| Dark psychological thriller with a twist | Memento; Shutter Island; Twelve Monkeys | Shutter Island; The Machinist; Donnie Darko | Memento; Fight Club; Shutter Island |
| Similar taste on Pulp Fiction | 9 of 20 neighbours, weighted 4.22 vs 4.20 overall | 14 of 20, weighted 4.43, 13 gave 4+ | 12 of 20, weighted 4.18 |
| Liked Toy Story, tired of animation | Pirates of the Caribbean; Ferris Bueller; Mary Poppins | Monty Python; Pirates of the Caribbean; The Princess Bride | Monty Python; Pirates; Back to the Future |
| Blind spot | Documentary 0, Film-Noir 1, Western 7 → try Roger & Me, Sin City, Unforgiven | Documentary 0, Film-Noir 0, Western 2 (avg 2.5), … | Documentary, Fantasy, Film-Noir, Musical, Romance: all 0 |

**What works.**

- Personalisation is visible. The same "dark thriller" request gives each user a different order, and user 30's picks lean toward Braveheart/Inception-style drama and war films.
- Explanations cite the user's own ratings, the real number of similar viewers and their weighted mean, and the item link with its support. They also flag low confidence for user 30, who has only 18 ratings.
- Constraints hold, and "that" resolves per user.

**What stands out.** User 30's blind spot lists genres they simply haven't rated. With 18 ratings, almost every genre falls under the threshold of `max(3, 5% of ratings)`. The answer says that a low count is not a dislike, but the analysis is weak for sparse profiles.

### Failure Analysis

**1. Tone is matched by keyword, not by meaning.**

- *Asked (user 15):* "phim chữa lành không hoạt hình" (a feel-good film, no animation).
- *Recommended:* Lucky Break, Good Will Hunting, and at #3 **The Puppet Masters (1994)**, a Horror/Sci-Fi film about alien parasites.
- *Why:* its plot summary mentions a film-within-the-film as a "morally uplifting pot-boiler". "Uplifting" belongs to the feel-good concept group. The user's liking for Sci-Fi then lifts it through the personal ranking, even though the film has 4 ratings and a single similar viewer gave it 3.0. The explanation does carry the caveat that keywords only hint at tone, but the pick is still wrong.
- *A related case:* "Something funny for the weekend" puts Friends with Benefits first (average 3.05). Its plot contains the words "funny" and "weekend"; "funny" is never mapped to Comedy.
- *Fix:*
  - rerank the top 20 with a semantic model (plot embeddings, or an LLM judge asked "is this feel-good?");
  - demote Horror for feel-good intents unless it is requested;
  - map mood words to genres;
  - validate the change on a labelled set of real requests.

**2. A common phrasing broke the constraint parser.** Review found this bug and it is now fixed.

- *Asked (user 15):* "I don't want horror, give me sci-fi".
- *Recommended (original code):* 28 Weeks Later, Heavy Metal, Alien³. All three are Horror.
- *Why:* the normaliser split "don't" into "don t", and the negation pattern only knew "no/not/without/tired of". So "horror" was parsed as a **required** genre, and "don" became a plot search term. The 16/16 probe score did not catch it, because I wrote the probes with the same rules in mind.
- *Fix applied:* the parser now handles apostrophes and a wider set of negation cues ("don't / do not / hate / not a fan of / no more", "ghét"), with up to four filler words ("don't want to watch any horror"). There are 3 new probes and 3 new tests. The system now returns Men in Black, Predator and Blade Runner.
- *Real fix:* an NLU/LLM parser that emits structured constraints, evaluated on held-out phrasings written by people other than me.

**3. Films with few ratings never reach the top 10.** This is a measured failure.

- *Asked (offline):* user 238's next highly rated film was **MacArthur (1977)**, rated 4.0, with 0 ratings in training.
- *Recommended:* Saving Private Ryan, Casablanca, Reservoir Dogs. The war/drama direction is right, but the target never appears.
- *Why:* item-item CF needs co-ratings that do not exist. Content at weight 0.25 only lifts a film within the content list, and personal rank dominates the fusion. Across all 31 test users whose target had ≤ 5 ratings, there were **0 hits**. Content-only ranking recovers 1 of them but loses 29 hits elsewhere.
- *Fix:*
  - reserve an exploration slot (for example, 1 of 10 results) filled from the content ranking of long-tail films, and measure it with its own cohort metric;
  - use a stronger content representation (embeddings);
  - for new users, onboard with a few seed films; the seed mechanism already supports this ("I liked X").

## Reflection

**What works well in your solution?**

- Every sentence is grounded. The same evidence object drives the text, the UI and the JSON API. Explanations name real counts and real films, and admit when evidence is thin.
- The investigative questions the brief highlights all work, in either language:
  - taste-neighbour opinion, compared against all viewers;
  - "why that" for the right user;
  - blind spots, now with a film to try in each.
- The evaluation is honest. Weights are chosen on validation, results are compared with baselines, uncertainty and cohorts are reported, and the cases where the system fails are shown.
- It is easy to run and ship: no dependencies, runs offline, 59 tests, a documented API and a Docker image.

**What doesn't work well? Why?**

- **Language understanding.** It is a set of rules. It covers the brief's queries and their Vietnamese equivalents, but not open paraphrase, and failure 2 shows how easily a rule gap becomes a wrong answer.
- **Tone and mood.** These are lexical guesses (failure 1).
- **The long tail.** It is invisible to the default ranking (failure 3). The explicit discovery mode ("hidden gems", "phim ít người biết") is the only way in, and its relevance is unmeasured.
- **The weight change.** The gain over the previous ranking is within noise. I kept it because it was chosen by a pre-registered validation rule, not because it is proven better.

**What would you do differently with more time or resources?**

1. Collect around 100 real requests written by other people and label them. Without them, no language-understanding change can be measured.
2. Put an LLM planner on top of the existing tools. It would parse constraints into structured filters and choose which analysis to run, while the answer stays restricted to the evidence the tools return. Measure it against the rule parser on that labelled set.
3. Add a semantic reranker for tone, plus an exploration slot for long-tail films, each with its own metric.
4. Run a fresh temporal test (a global time cutoff rather than per-user) to get a cleaner estimate.

## Open Section

**Changes after self-review.** I reviewed the first version against the brief and fixed what I found:

- negation parsing (failure 2);
- "movies like X" is now treated as a seed rather than a keyword search;
- explanations report the real number of similar viewers (they always said 3) and flag when that group is lukewarm;
- the ranking score is no longer printed as "x/5";
- the taste-neighbour answer now compares against all viewers and reports agreement;
- blind spots ignore IMAX and suggest a film per genre;
- "Why would I like Arrival?" says the film is not in the catalog instead of explaining a different one;
- conversation context is stored per user, so concurrent API users cannot see each other's context;
- replies follow the language of the question;
- the optional Bing web search was removed, because it was outside the brief and made outbound calls.

Re-running the full evaluation after these changes reproduced **identical** validation and test metrics, per-user results included, so the ranking itself is unchanged.

**Current intent versus history.** User 15 rated Toy Story 2.5/5, yet asks as someone who liked it. The assistant treats the statement as a seed for this request only, keeps the "no animation" constraint, and does not rewrite the stored rating.

**Honest but questionable picks.** For "Recommend a war movie", user 30 gets Captain America: The First Avenger first. The explanation says why: similar viewers are lukewarm (3.29/5), and the film ranks on its co-liked link to Iron Man. That transparency is useful, but a person would probably have put Saving Private Ryan (similar viewers 4.12/5) first. The item-link signal can outweigh direct neighbour evidence for users with sparse profiles.

**Serving.** `python -m movie_assistant.web` (or `docker compose up`) serves the UI and a versioned JSON API with OpenAPI docs (`docs/API.md`, `/docs`). The per-user conversation context is kept in memory. That is fine for one process; running several replicas would need sticky sessions or a shared store.
