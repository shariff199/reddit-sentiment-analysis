# Methodology

This document explains the design decisions behind each pipeline stage.

---

## 1. Data source & scope

**Source:** Kaggle dataset `reddit_opinion_PSE_ISR.csv`, a public scrape of Reddit comments and post metadata from subreddits discussing Israel and Palestine. 1,822,582 rows, 24 columns spanning comment text, post text, author metadata (karma, account age), engagement (score, ups, downs, controversiality), and timestamps.

**Temporal scope:** 7 October 2023 onwards. The dataset includes some pre-war content, but the dissertation focuses on discourse triggered by the Hamas attack of 7 October 2023 and the subsequent Gaza war. Pre-Oct-7 rows are dropped (~2,701 rows).

---

## 2. Cleaning & deduplication

**Null handling:**

| Column | Strategy | Rationale |
|---|---|---|
| `self_text` (comment body) | Fill with `''` | Comment body can be legitimately empty (e.g., image-only comments) |
| `post_self_text` | Fill with `''` | Posts may have no body, only a title + link |
| `user_*_karma` (5 cols) | Fill with `0` | Missing karma typically = deleted/suspended account; 0 is conservative |
| `user_account_created_time` | Fill with median timestamp | Preserves the column's distribution for any time-based analysis |

**Deduplication strategy:**

- Drop exact duplicates on `(post_title, post_self_text, author_name, created_time)` — same author posting identical content twice.
- Aggregate engagement metrics (`ups`, `downs`, `score`) by sum/mean for the kept row.
- Drop a curated list of generic and moderation comments (`Yes`, `Lol`, `Source?`, `This has been removed for breaking the Reddit rules…`, etc.) — these are not informative for sentiment.
- Drop comments authored by `[deleted]` (suspended/deleted users).

---

## 3. Feature reduction (correlation-based)

Pearson correlation matrix on numerical columns. Columns dropped:

| Column | Reason |
|---|---|
| `ups` | Perfectly correlated with `score` (r = 1.0) |
| `user_awardee_karma`, `user_awarder_karma` | Low correlation, redundant with `user_total_karma` |
| `user_link_karma`, `user_comment_karma` | High correlation with `user_total_karma` |
| `downs` | Mostly NaN/zero (Reddit no longer exposes downs publicly) |
| `post_thumbs_ups` | Duplicates `post_score` |
| `post_total_awards_received` | Sparse/NaN |

---

## 4. Text normalization

Applied identically to `self_text`, `post_self_text`, `post_title`:

1. **Lowercase**
2. **Expand contractions** (`don't` → `do not`) via `contractions` library — important for VADER which weights individual words
3. **Strip URLs** (`http\S+`)
4. **Strip HTML tags** (`<.*?>`)
5. **Strip punctuation & special characters** (`[^\w\s]`)
6. **Strip digits** (`\d+`)
7. **Strip non-ASCII** — removes emoji, Hebrew/Arabic script (acknowledged limitation; see `docs/LIMITATIONS.md`)
8. **Remove English stopwords** (NLTK)
9. **Lemmatize** using WordNet (`WordNetLemmatizer`)

Output stored in three new columns: `cleaned_self_text`, `cleaned_post_self_text`, `cleaned_post_title`.

---

## 5. Sentiment analysis — VADER

**Why VADER:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is rule-based, fast on >1M comments, and explicitly designed for social-media text (handles slang, intensifiers, negations). No training data required.

**Thresholds** (industry-standard):

| Compound score | Category |
|---|---|
| `> 0.05` | Positive |
| `< -0.05` | Negative |
| `-0.05 to 0.05` | Neutral |

VADER is run independently on `cleaned_self_text` (comment) and `cleaned_post_title` (post title), producing `comment_sentiment` and `post_title_sentiment` numeric scores plus `*_category` labels.

---

## 6. Stance classification

**Approach:** keyword-based, two curated lists (~20 terms each) for pro-Israel and pro-Palestine. A comment is classified `pro-israel` if it contains any pro-Israel keyword; `pro-palestine` if it contains any pro-Palestine keyword; otherwise `neutral`. Pro-Israel takes priority if both match (acknowledged simplification).

**Why keyword-based and not a fine-tuned classifier:** stance detection on this topic is genuinely hard — the same keywords (`hamas`, `israeli`, `gaza`) are used by *both* sides in opposing rhetorical frames. A fine-tuned model would need a labelled dataset that does not exist publicly for this exact corpus. The keyword heuristic is documented as a *lower bound* on stance signal, not a final classifier.

---

## 7. TF-IDF & bigrams

- **TF-IDF** (`TfidfVectorizer`, max 100 features, English stopwords) on `cleaned_self_text` and on `cleaned_post_title` — separate runs.
- **Bigrams** via NLTK `bigrams()` on a sampled 1,000,000-row subset (script line ~1245), with the top 20 plotted.

Both are diagnostic, not predictive — used to characterise *what* discourse looks like, not to classify it.

---

## 8. Emotion detection — NRCLex

NRCLex maps tokens to the 10-emotion lexicon (Plutchik + sentiment polarity): `anger, fear, anticipation, trust, surprise, positive, negative, sadness, disgust, joy`. Each comment gets a count vector over these 10 emotions; per-emotion scores are averaged across the corpus and across subreddits.

**Why NRCLex:** lexicon-based, no training, fast on millions of rows. Same trade-offs as VADER (no irony detection, no domain tuning).

---

## 9. Topic modelling — LDA

- `CountVectorizer(max_df=0.9, min_df=5)` to build the document-term matrix
- `LatentDirichletAllocation(n_components=5)` on a **100,000-row sample** (runtime concession on a 1.8M-row corpus)
- 5 topics for comments, 5 topics for post titles
- Output: top 20 words per topic + word cloud

Topics are manually labelled by the analyst based on word clouds; labels are subjective.

---

## 10. Misinformation flagging

A binary flag based on lexical match to a curated list of misinformation/propaganda terms (`fake news`, `hoax`, `false flag`, `conspiracy`, `propaganda`, `disinformation`, etc.). Analysis compares average sentiment between flagged and non-flagged comments.

**Important:** this measures comments *discussing* misinformation, not comments that *are* misinformation. Genuine claim verification is out of scope.

---

## 11. Time-series forecasting — Prophet

- Aggregate `comment_sentiment` to weekly means
- Fit Facebook Prophet (default settings — additive trend + weekly seasonality)
- Forecast 12 weeks ahead

Prophet was chosen over ARIMA for two reasons: (1) handles missing weeks gracefully, (2) decomposes trend vs seasonality which makes interpretation easier for non-statistical readers (a key audience for a dissertation).

---

## 12. Statistical testing

**Chi-square test of independence** on (`subreddit`, `comment_sentiment_category`) cross-tabulation. Tests whether sentiment distribution differs significantly across subreddits.

`scipy.stats.chi2_contingency` returns the χ² statistic, p-value, degrees of freedom, and expected counts. Significance threshold: α = 0.05.

---

## 13. User-level analysis

- Group by `author_name`
- Compute: comment count, mean sentiment, total score
- Identify top-10 users by each metric
- Compute "dominant stance" per user via mode of their per-comment stance labels

---

## 14. Reproducibility notes

- All `random_state=42` where applicable.
- LDA and bigram analyses use `df.sample(n=..., random_state=42)` — results are deterministic given the dataset and seed.
- Pipeline is single-threaded except where libraries (Prophet, sklearn LDA with `n_jobs=-1`) parallelise internally.
