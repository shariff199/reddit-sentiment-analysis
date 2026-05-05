"""Streamlit dashboard for the Reddit Israel-Palestine sentiment analysis.

Run with:
    streamlit run dashboard.py

Reads from results/tables/*.csv if available (cheap), and falls back to
recomputing from the raw CSV in data/ for richer interactive views.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_TABLES = Path("results/tables")
DEFAULT_DATA_PATH = Path("data/reddit_opinion_PSE_ISR.csv")
DATA_PATH = Path(os.environ.get("REDDIT_DATA_PATH", str(DEFAULT_DATA_PATH)))


st.set_page_config(
    page_title="Reddit Sentiment — Israel/Palestine",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    path = RESULTS_TABLES / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_raw_sample(n: int = 50_000) -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    df["post_created_time"] = pd.to_datetime(df["post_created_time"], errors="coerce")
    return df


st.title("Reddit Sentiment Analysis — Israel/Palestine post Oct 7")
st.caption(
    "MSc dissertation pipeline · 1.82M Reddit comments · "
    "VADER sentiment + NRCLex emotion + LDA topics + Prophet forecasting"
)

# -- Sidebar --
st.sidebar.header("Data sources")
have_tables = RESULTS_TABLES.exists() and any(RESULTS_TABLES.glob("*.csv"))
have_raw = DATA_PATH.exists()
st.sidebar.markdown(
    f"- Pre-computed tables: {'✅ found' if have_tables else '❌ missing — run sentiment_pipeline.py first'}"
)
st.sidebar.markdown(
    f"- Raw dataset (`{DATA_PATH}`): {'✅ found' if have_raw else '❌ missing'}"
)

if not have_tables and not have_raw:
    st.warning(
        "No data found. Either run `python sentiment_pipeline.py` (generates "
        "`results/tables/*.csv`) or place the Kaggle CSV at `data/"
        "reddit_opinion_PSE_ISR.csv`."
    )
    st.stop()

# -- Tabs --
tab_overview, tab_sentiment, tab_stance, tab_topics, tab_explore = st.tabs(
    ["Overview", "Sentiment", "Stance", "Topics & terms", "Explore raw"]
)

# ---------- Overview ----------
with tab_overview:
    st.subheader("Headline numbers")
    sentiment_counts = load_table("sentiment_counts")
    if sentiment_counts is not None:
        sentiment_counts.columns = ["category", "count"]
        total = int(sentiment_counts["count"].sum())
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Comments analysed", f"{total:,}")
        for col, cat in zip(
            (col2, col3, col4), ("negative", "positive", "neutral")
        ):
            row = sentiment_counts[sentiment_counts["category"] == cat]
            if not row.empty:
                count = int(row["count"].iloc[0])
                col.metric(cat.capitalize(), f"{count:,}", f"{count/total*100:.1f}%")
    else:
        st.info("Run the pipeline to populate `results/tables/sentiment_counts.csv`.")

    st.subheader("Most popular subreddits")
    popular = load_table("popular_subreddits")
    if popular is not None:
        popular.columns = ["subreddit", "count"]
        st.bar_chart(popular.head(15).set_index("subreddit"))

# ---------- Sentiment ----------
with tab_sentiment:
    st.subheader("Sentiment distribution")
    sentiment_counts = load_table("sentiment_counts")
    if sentiment_counts is not None:
        sentiment_counts.columns = ["category", "count"]
        st.bar_chart(sentiment_counts.set_index("category"))
        st.dataframe(sentiment_counts, use_container_width=True)

    st.subheader("Sentiment vs controversiality")
    avg_by_contro = load_table("avg_sentiment_by_controversial")
    if avg_by_contro is not None:
        st.dataframe(avg_by_contro, use_container_width=True)

    st.subheader("Sentiment differs across subreddits — chi-square test")
    chi = load_table("chi_square_subreddit_sentiment")
    if chi is not None:
        st.dataframe(chi, use_container_width=True)

# ---------- Stance ----------
with tab_stance:
    st.subheader("Overall stance (keyword-based)")
    stance = load_table("stance_summary")
    if stance is not None:
        stance.columns = ["stance", "count"]
        st.bar_chart(stance.set_index("stance"))

    st.subheader("Stance composition by subreddit")
    stance_by_sub = load_table("stance_by_subreddit")
    if stance_by_sub is not None:
        st.dataframe(stance_by_sub, use_container_width=True)

# ---------- Topics & terms ----------
with tab_topics:
    st.subheader("LDA topics — comments")
    lda_comments = load_table("lda_topics_cleaned_self_text")
    if lda_comments is not None:
        st.dataframe(lda_comments, use_container_width=True)

    st.subheader("LDA topics — post titles")
    lda_titles = load_table("lda_topics_cleaned_post_title")
    if lda_titles is not None:
        st.dataframe(lda_titles, use_container_width=True)

    col_l, col_r = st.columns(2)
    tfidf = load_table("tfidf_top_terms_comments")
    bigrams = load_table("top_bigrams")
    with col_l:
        st.subheader("Top 20 TF-IDF terms")
        if tfidf is not None:
            st.dataframe(tfidf.head(20), use_container_width=True)
    with col_r:
        st.subheader("Top 20 bigrams")
        if bigrams is not None:
            st.dataframe(bigrams, use_container_width=True)

# ---------- Explore raw ----------
with tab_explore:
    st.subheader("Browse the raw dataset")
    if not have_raw:
        st.info(
            "Place `reddit_opinion_PSE_ISR.csv` in `data/` (or set "
            "`REDDIT_DATA_PATH`) to enable this tab."
        )
    else:
        n = st.slider("Sample size", 1_000, 200_000, 50_000, step=1_000)
        df = load_raw_sample(n)
        subs = ["(all)"] + sorted(df["subreddit"].dropna().unique().tolist())
        chosen = st.selectbox("Subreddit", subs)
        if chosen != "(all)":
            df = df[df["subreddit"] == chosen]
        st.write(f"Rows: {len(df):,}")
        st.dataframe(
            df[["subreddit", "author_name", "score", "self_text", "created_time"]]
            .head(500),
            use_container_width=True,
        )

st.caption("See `docs/RESULTS.md` for full findings and `docs/LIMITATIONS.md` for caveats.")
