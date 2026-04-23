# Reddit Sentiment & Misinformation Analysis at Scale

> MSc Data Science Dissertation — University of South Wales (2024)  
> Grade: **Distinction**

---

## Overview

An end-to-end NLP pipeline that collects, processes, and analyses **1.8 million Reddit comments** on Public Sector Efficiency and ISR (Institutional Social Responsibility) to uncover public opinion patterns and detect misinformation.

**Key Results:**
- 45% of comments showed **negative sentiment** toward public sector efficiency
- **6.5% misinformation** rate detected across sampled posts
- Multi-model approach: VADER, TextBlob, BERT, and LSTM classifier

---

## Architecture

```
Reddit API (PRAW)
      ↓
Data Collection (1.8M comments)
      ↓
Preprocessing (cleaning, tokenisation, stopword removal)
      ↓
Sentiment Analysis (VADER + TextBlob)
      ↓
Deep Learning Classification (LSTM / BERT)
      ↓
Misinformation Detection
      ↓
Visualisation & Reporting
```

---

## Files

| File | Description |
|------|-------------|
| `sentiment_analysis_reddit.ipynb` | Main Jupyter notebook — full pipeline |
| `sentiment_pipeline.py` | Python script version of the pipeline |

---

## Tech Stack

- **Data Collection:** PRAW (Reddit API), Python
- **NLP:** VADER, TextBlob, BERT (HuggingFace Transformers)
- **Deep Learning:** LSTM (TensorFlow/Keras)
- **Data Processing:** Pandas, NumPy, re, NLTK
- **Visualisation:** Matplotlib, Seaborn, WordCloud

---

## Key Findings

| Metric | Result |
|--------|--------|
| Dataset size | 1.8 million comments |
| Negative sentiment | 45% |
| Neutral sentiment | ~49% |
| Misinformation detected | 6.5% |
| Best classifier accuracy | BERT fine-tuned |

---

## Setup

```bash
pip install praw pandas numpy nltk transformers torch scikit-learn matplotlib seaborn
jupyter notebook sentiment_analysis_reddit.ipynb
```

---

*University of South Wales · MSc Data Science · Student ID: 30109662*
