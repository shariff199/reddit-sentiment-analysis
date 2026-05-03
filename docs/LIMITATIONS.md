# Limitations & Ethical Considerations

This dissertation analyses online discourse about an active armed conflict involving substantial loss of civilian life. Methodological limitations are not just academic; they have implications for how findings should — and should not — be cited.

---

## Methodological limitations

### 1. Sentiment model (VADER)

- VADER is a **lexicon-based** model trained on Twitter/microblog data. It was not domain-tuned for war discourse, casualty counts, or political rhetoric.
- It cannot detect **sarcasm, irony, or rhetorical inversion**. A comment like *"yeah great, another war crime, business as usual"* carries a positive lexical signal but a clearly negative meaning.
- It is **English-only**. The dataset contains code-switched Hebrew, Arabic, transliterated terms, and emoji. The cleaning step strips non-ASCII characters, which silently discards meaningful signal from non-English-speaking users.

### 2. Stance classification

- The pro-Israel / pro-Palestine classifier is **keyword-based**, not a learned model.
- The same keyword (`hamas`, `israeli`, `gaza`) appears in opposing rhetorical frames. A comment criticising Hamas may be classified `pro-palestine` simply because it mentions the word.
- The keyword lists themselves embed the analyst's framing choices. Different lists would produce different distributions.
- Reported stance numbers should be read as **"comments containing pro-X language"**, not **"comments holding pro-X views"**.

### 3. Misinformation flag

- Detects only **lexical mentions** of misinformation-related terms (`fake news`, `propaganda`, etc.).
- Captures comments **about** misinformation just as readily as comments **spreading** misinformation.
- Does not perform claim verification, source-checking, or factual ground-truthing.
- The 6%-flagged-comment range reported here should not be cited as a "misinformation rate".

### 4. Sample-based analyses

- LDA topic modelling is run on a **100,000-row sample** (5.5% of the corpus). Topic distributions on the full corpus may differ.
- Bigram analysis uses a **1,000,000-row sample**. Long-tail bigrams are likely under-represented.
- These sampling choices are runtime concessions on consumer hardware.

### 5. Selection bias in the dataset

- Reddit is **not representative** of public opinion. Its user base skews young, English-speaking, Western, and male.
- Subreddits self-select: people who post on `r/IsraelPalestine` are more engaged with the topic than the median internet user.
- Comments are also moderated — anything banned by Reddit before scraping is silently absent.
- Findings describe **Reddit discourse**, not "public opinion" broadly.

### 6. Temporal cutoff

- The dataset's most recent rows are from ~late 2024. Anything that has happened since (ceasefire negotiations, regional escalation, etc.) is not reflected.

---

## Ethical considerations

### Topic sensitivity

This is a politically charged topic with genuine human cost. The analysis aims to be **descriptive** of online discourse, not normative about the underlying conflict. Readers should not interpret a stance distribution (e.g., "X% pro-Israel comments") as a statement about which side is morally correct.

### Reproducibility

The dataset is publicly available via Kaggle, and the analysis script is reproducible. However, anyone reproducing this analysis should:

- Disclose the same limitations
- Avoid presenting numbers without their methodological caveats
- Not use individual user-level results (top commenters, etc.) to identify or target real people

### Anonymisation

- The dataset includes Reddit usernames (`author_name`). The analysis aggregates by user but does not publish per-user results in this repo.
- If extending the work, treat usernames as **personal data** under GDPR and avoid joining with other datasets that could de-anonymise.

### Misinformation framing

The "misinformation flag" is a coarse instrument. Citing it as evidence that *one side* spreads more misinformation than the other would be **a misuse of the methodology**. The flag captures lexical patterns, not truth values.

---

## Future work

Improvements that would address the above:

| Limitation | Possible improvement |
|---|---|
| VADER's domain | Fine-tune a transformer (RoBERTa / DeBERTa) on annotated war-discourse data |
| Keyword stance | Train a supervised stance classifier on a labelled subset |
| Non-English signal lost | Use multilingual models (XLM-R) and keep non-ASCII text |
| Misinformation flag is lexical | Use claim-detection + retrieval + verification (e.g., FEVER-style pipeline) |
| Reddit-only data | Cross-platform analysis (Reddit + Twitter/X + news comments) |
| Static topic count (k=5) | Use HDP or perplexity-based k-selection |

These are all out of scope for the dissertation but represent natural extensions.
