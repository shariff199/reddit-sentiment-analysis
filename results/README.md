# Generated outputs

This folder is populated automatically when you run `python sentiment_pipeline.py`.

```
results/
├── figures/                 # all matplotlib + plotly plots, auto-saved
│   ├── fig_001.png          # numbered in pipeline order
│   ├── fig_002.png
│   ├── ...
│   ├── funnel_sentiment.png # plotly sentiment funnel chart
│   └── funnel_sentiment.html
└── tables/                  # key result tables as CSV
    ├── popular_subreddits.csv
    ├── sentiment_counts.csv
    ├── stance_summary.csv
    ├── stance_by_subreddit.csv
    ├── unique_users_per_subreddit.csv
    ├── tfidf_top_terms_comments.csv
    ├── top_bigrams.csv
    ├── emotion_by_subreddit.csv
    ├── misinformation_counts.csv
    ├── lda_topics_cleaned_self_text.csv
    ├── lda_topics_cleaned_post_title.csv
    ├── chi_square_subreddit_sentiment.csv
    ├── controversiality_counts.csv
    └── avg_sentiment_by_controversial.csv
```

The figures and tables themselves are gitignored (regenerated on each run). This README and the folder structure are tracked so the layout is documented.
