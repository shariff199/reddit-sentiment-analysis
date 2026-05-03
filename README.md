# Reddit Sentiment Analysis — Israel–Palestine Discourse Post Oct 7

> MSc Data Science Dissertation — University of South Wales (2024)
> Student ID: 30109662

An end-to-end NLP pipeline analysing **~1.82 million Reddit comments and posts** discussing the Israel–Palestine conflict from **7 October 2023** onwards (the start of the Gaza war). The project measures sentiment, classifies stance (pro-Israel / pro-Palestine / neutral), detects emotion and misinformation patterns, models topics, and forecasts how sentiment evolves over time.

---

## Dataset

- **Source:** [`reddit_opinion_PSE_ISR.csv`](https://www.kaggle.com/datasets/asaniczka/public-and-arab-opinions-on-israeli-palestinian) (Reddit comments and post metadata)
- **Size:** ~1,820,000 rows × 24 columns
- **Time filter applied:** rows on or after `2023-10-07`
- **Subreddits covered:** `r/IsraelPalestine`, `r/Palestine`, `r/Israel`, `r/worldnews`, and others

The dataset is **not committed** to this repo (too large). To run the pipeline:

```bash
mkdir data
# place reddit_opinion_PSE_ISR.csv inside data/
```

Or set an environment variable:

```bash
export REDDIT_DATA_PATH=/full/path/to/reddit_opinion_PSE_ISR.csv   # macOS/Linux
$env:REDDIT_DATA_PATH = "C:\full\path\to\reddit_opinion_PSE_ISR.csv"  # PowerShell
```

---

## Pipeline

| # | Stage | Method |
|---|---|---|
| 1 | Load & inspect | pandas, missingno |
| 2 | Filter by date | drop everything before 2023-10-07 |
| 3 | Null handling | placeholder fill for text, 0 for karma, median for `user_account_created_time` |
| 4 | Deduplicate | drop dupes by `(post_title, post_self_text, author_name, created_time)`; remove generic mod comments ("Yes", "Lol", removed-by-Reddit notices) |
| 5 | Correlation analysis | drop redundant numerical features (ups, sub-karmas, etc.) |
| 6 | EDA | top subreddits, top users, posting activity over time |
| 7 | Text normalization | lowercase → expand contractions → strip URLs/HTML/punctuation/digits/non-ASCII → stopwords → WordNet lemmatization |
| 8 | Sentiment | VADER (`SentimentIntensityAnalyzer`), categorised as positive / neutral / negative |
| 9 | Stance classification | keyword lists for pro-Israel and pro-Palestine terms |
| 10 | TF-IDF | top 100 terms across all comments |
| 11 | Bigrams | top 20 most common bigrams (NLTK) |
| 12 | Emotion detection | NRCLex — 10 emotion dimensions (anger, fear, joy, trust, etc.) |
| 13 | Topic modelling | LDA (5 topics on comments, 5 on post titles) with word clouds |
| 14 | Misinformation flagging | keyword-based (`fake news`, `propaganda`, `hoax`, etc.) |
| 15 | User behaviour | top users by activity, sentiment, total score |
| 16 | Time-series forecasting | Facebook Prophet — 12-week-ahead weekly sentiment forecast |
| 17 | Statistical testing | chi-square test for sentiment difference across subreddits |
| 18 | Controversiality analysis | sentiment of controversial vs non-controversial comments |

---

## Files

| File | Description |
|---|---|
| `sentiment_pipeline.py` | Full pipeline as a Python script |
| `sentiment_analysis_reddit.ipynb` | Same pipeline as a Jupyter notebook with rendered outputs and plots |
| `requirements.txt` | All Python dependencies |
| `docs/RESULTS.md` | Empirical findings: sentiment, stance, topics, forecasting, with explanations |
| `docs/METHODOLOGY.md` | Detailed methodology and design decisions for each pipeline stage |
| `docs/LIMITATIONS.md` | Known methodological caveats and ethical considerations |
| `LICENSE` | MIT |
| `.gitignore` | Excludes `data/`, virtual envs, and notebook checkpoints |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/shariff199/reddit-sentiment-analysis.git
cd reddit-sentiment-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK + NRCLex data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# 5. Place the dataset in data/reddit_opinion_PSE_ISR.csv (see Dataset section)

# 6. Run
python sentiment_pipeline.py
# or open the notebook
jupyter notebook sentiment_analysis_reddit.ipynb
```

---

## Tech Stack

- **Data processing:** pandas, NumPy
- **NLP:** NLTK, contractions, scikit-learn (TF-IDF, CountVectorizer, LDA)
- **Sentiment & emotion:** VADER (`vaderSentiment`), NRCLex
- **Forecasting:** Prophet
- **Statistics:** SciPy (chi-square)
- **Visualisation:** matplotlib, seaborn, plotly, wordcloud, missingno

---

## Key Findings (summary)

| Metric | Value |
|---|---|
| Comments analysed (post-filter) | 1,818,575 |
| Negative sentiment | **44.0%** (796,587) |
| Positive sentiment | 35.1% (635,028) |
| Neutral sentiment | 20.9% (377,694) |
| Top bigrams | `west bank`, `war crime`, `state solution`, `ethnic cleansing` |
| LDA topics (5) | Geopolitical · Military · International Relations · Religious/Cultural · Media |
| Chi-square (subreddit × sentiment) | p < 0.05 — sentiment significantly differs across subreddits |

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full results writeup with interpretations.

---

## Notes

- The script is a Jupyter export, so it still contains `# In[N]:` cell markers. It runs end-to-end as a single Python script after the dataset and dependencies are in place.
- LDA and Prophet steps sample the data (100k rows) rather than using all 1.8M rows, to keep runtime manageable on a laptop.
- All visualisations open as `plt.show()` windows when run as a script; use the notebook for inline figures.
- Methodological caveats — particularly around the keyword-based stance classifier and the lexical misinformation flag — are documented in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). Findings should be cited with these caveats in mind.

---

*University of South Wales · MSc Data Science · 2024*
