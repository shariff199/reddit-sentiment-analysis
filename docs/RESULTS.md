# Results & Findings

This document summarises the empirical findings from running the pipeline in `sentiment_pipeline.py` on the `reddit_opinion_PSE_ISR.csv` dataset (1,822,582 rows, 24 columns).

---

## 1. Dataset & filtering

| Step | Rows |
|---|---|
| Raw dataset | **1,822,582** |
| After removing Reddit moderation/policy comments | 1,821,276 |
| After filter `>= 2023-10-07` (Oct 7 attacks) | **1,818,575** |
| Dropped by date filter | 2,701 |
| Duplicate posts (same `post_title` + `post_self_text`) | 1,789,074 |
| Duplicate comments (same `self_text`) | 45,749 |

**Interpretation:** the dataset is comment-heavy — most "duplicate posts" come from comments sharing the same parent post (1.8M comments mapped to a much smaller pool of unique posts). Mod/policy notices, generic replies (`"Yes"`, `"Lol"`, `"Source?"`), and ban-evasion notices were the most common removed text patterns; the top 15 generic comments alone account for ~5,000 rows.

---

## 2. Sentiment distribution (VADER)

VADER compound score thresholded at ±0.05.

| Category | Count | Share |
|---|---:|---:|
| Negative | 796,587 | **44.0%** |
| Positive | 635,028 | 35.1% |
| Neutral  | 377,694 | 20.9% |

**Interpretation:** discourse on r/IsraelPalestine and adjacent subreddits is dominated by negative sentiment — unsurprising given the topic (war, casualties, displacement). The fact that *positive* sentiment still reaches 35% reflects expressions of solidarity, support for one side or the other, and gratitude language ("thank you", "support"), rather than positivity about the conflict itself. Neutral comments (21%) are typically factual, news-sharing, or moderation-adjacent.

---

## 3. Stance classification (keyword-based)

The pipeline classifies each comment as **pro-Israel**, **pro-Palestine**, or **neutral** based on appearance of curated keyword lists:

- **Pro-Israel keywords:** `idf`, `israel defense`, `right to defend`, `terrorism`, `self-defense`, `security barrier`, `rocket attacks`, `right to exist`, `two-state solution`, `peace process`, `settlements`, `annexation`, `jewish homeland`, `zionism`, `diaspora`, `temple mount`, `zionist`, `netanyahu`, `israeli`
- **Pro-Palestine keywords:** `palestinian rights`, `occupation`, `freedom`, `apartheid`, `ethnic cleansing`, `intifada`, `freedom fighters`, `resistance`, `bds`, `boycott`, `gaza blockade`, `refugees`, `right of return`, `human rights violations`, `UN resolutions`, `palestine`, `hamas`, `gaza`, `west bank`, `palestinian`

**Interpretation:** keyword-based stance is a *proxy*, not a true stance classifier. A comment criticising Hamas may still be classified as "pro-Palestine" simply because it contains the word `hamas`. The cross-tab of stance × subreddit (see notebook heatmap) shows that subreddits like `r/IsraelPalestine` and `r/worldnews` carry roughly balanced stance distributions, while topical subs (`r/Israel`, `r/Palestine`) skew strongly toward their own side — which validates the keyword approach as at least directionally correct.

---

## 4. Top bigrams in comments

| Rank | Bigram | Count (sample) |
|---:|---|---:|
| 1 | `west bank` | 130 |
| 2 | `war crime` | 58 |
| 3 | `look like` | 54 |
| 4 | `state solution` | 51 |
| 5 | `october th` | 48 |
| 6 | `year ago` | 47 |
| 7 | `ethnic cleansing` | 44 |
| 8 | `middle east` | 44 |
| 9 | `israeli government` | 40 |
| 10 | `palestinian state` | 39 |
| 11 | `jewish people` | 37 |
| 12 | `human right` | 35 |
| 13 | `social medium` | 35 |
| 14 | `international law` | 34 |
| 15 | `palestinian people` | 31 |
| 16 | `jewish state` | 31 |
| 17 | `state israel` | 30 |
| 18 | `woman child` | 29 |
| 19 | `human shield` | 28 |
| 20 | `sound like` | 28 |

(Counts on a 5,000-row sample for tractability.)

**Interpretation:** the dominant bigrams are geographic (`west bank`, `middle east`), legal/normative (`war crime`, `human right`, `international law`, `ethnic cleansing`), and political-solution language (`state solution`, `palestinian state`, `jewish state`). The bigram `october th` (i.e., "October 7th") confirms the dataset's temporal centre of gravity around the Hamas attack.

---

## 5. LDA topic modelling (5 topics, 100k-row sample)

The script labels the five LDA topics on cleaned comments as:

| Topic | Theme |
|---|---|
| 1 | Geopolitical Conflict |
| 2 | Military Actions and Casualties |
| 3 | International Relations |
| 4 | Religious and Cultural Issues |
| 5 | Media and Public Reactions |

**Interpretation:** the topic structure is consistent with what one would expect of war-time discourse — operational (T2), diplomatic (T3), ideological/religious (T4), and meta-discussion about how the conflict is being covered (T5). Word clouds for each topic are rendered inline in the notebook.

---

## 6. Emotion detection (NRCLex)

NRCLex assigns 10 emotion dimensions per comment. Across the corpus the strongest signals are typically **fear** and **anger**, followed by **trust** and **anticipation**, with **joy** the lowest — consistent with a war/conflict topic. The per-subreddit heatmap (see notebook Fig. *Emotion Scores Across Top Subreddits*) shows `r/IsraelPalestine` carries the most balanced emotional spectrum, while news-aggregator subs like `r/worldnews` lean more heavily on fear/anger.

---

## 7. Misinformation flagging

The script flags any comment containing one of these keywords: `fake news`, `hoax`, `false flag`, `conspiracy`, `deep state`, `propaganda`, `misinformation`, `disinformation`, `biased`, `fake`, `lies`, `false`.

**Interpretation:** average sentiment of misinformation-flagged comments is *lower (more negative)* than non-flagged comments. This is consistent with the literature: people accusing others of spreading falsehoods do so in negatively-valenced language. The flag is a coarse upper bound — it captures comments *discussing* misinformation as well as comments *spreading* it.

---

## 8. Time-series forecasting (Prophet)

A weekly-aggregated mean sentiment series is fitted with Prophet and forecast 12 weeks forward.

**Interpretation:** the trend component shows weekly average sentiment dipping around major escalations (e.g., late Oct 2023, again early 2024 around Rafah operations) and recovering between them. The 12-week forecast extends this oscillating-but-weakly-negative pattern; absent a peace event, sentiment is not predicted to shift positive.

---

## 9. Statistical test — sentiment difference across subreddits

A chi-square test of independence on (subreddit × sentiment_category) returns **p < 0.05**, so we reject the null of equal sentiment distribution across subreddits. Subreddit identity is statistically associated with sentiment composition.

**Interpretation:** the platform is not homogeneous. Where a comment is posted (which subreddit) carries information about its likely sentiment — a fact that should temper any "Reddit overall thinks X" generalisation.

---

## 10. Controversiality

Reddit's `controversiality=1` flag (comments with roughly equal upvotes and downvotes, and a minimum activity level) shows:
- Controversial comments have **lower mean sentiment** than non-controversial ones
- Controversial comments have a wider sentiment IQR (i.e., they're either strongly negative or strongly positive)

**Interpretation:** controversy on Reddit, in this dataset, is correlated with negative-skewing or polarising language — not just with topic.

---

## 11. Caveats on these numbers

- **VADER** is a lexicon-based sentiment model trained on social-media text; it is not domain-tuned for war discourse and may misread sarcasm, irony, and code-switching (Arabic/Hebrew/English).
- **Stance classification** is keyword-based, not learned. Numbers should be read as "comments containing pro-X language" rather than "comments holding pro-X stance".
- **LDA** is run on a 100,000-row sample, not the full corpus, for runtime reasons.
- **Misinformation flag** only matches surface lexical patterns; it does not perform claim verification.

See `docs/LIMITATIONS.md` for the full methodological caveats.
