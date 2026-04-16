# 🎬 Streaming Content Catalog: Netflix, Prime, Disney+ & More

## Overview

**15,000 titles** from **10 major streaming platforms** — Netflix, Amazon Prime Video, Disney+, Hulu, HBO Max, Apple TV+, Paramount+, Peacock, JioCinema, and Crunchyroll. Includes IMDb ratings, Rotten Tomatoes scores, cast, directors, genres, budgets, awards, and watch-time data from 1980–2026.

---

## 🎯 Use Cases

- **Content recommendation systems**: Collaborative filtering, content-based, hybrid recommenders
- **Platform comparison**: Which service has the best library for each genre?
- **Rating prediction**: Regression from metadata to IMDb/RT scores
- **Genre popularity trends**: How has content mix evolved 1980→2026?
- **NLP on descriptions**: Topic modeling, genre classification from plot summaries
- **Budget ROI analysis**: Which budget tiers deliver the best ratings?
- **International content analysis**: K-drama, anime, Bollywood, Telenovela trends
- **Award prediction**: Classify Oscar/Emmy winners from metadata
- **Duration optimization**: Is there an ideal runtime by genre?

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `streaming_catalog.csv` | 15,000 | Individual titles with all metadata |
| `platform_summary.csv` | 10 | Platform-level aggregated stats |
| `genre_summary.csv` | 21 | Genre-level aggregated stats |
| `country_summary.csv` | 24 | Country-level aggregated stats |
| `yearly_release_trends.csv` | 47 | Release year trends (1980–2026) |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("streaming_catalog.csv")

# Top-rated titles
top = df.nlargest(20, 'imdb_rating')[['title', 'platform', 'primary_genre', 'imdb_rating']]
print(top)

# Titles per platform
df['platform'].value_counts().plot.barh()
plt.title("Titles per Streaming Platform")
plt.show()
```

---

## 💡 Suggested Experiments

1. **Content-based recommender**: TF-IDF on descriptions + cosine similarity
2. **IMDb rating regression**: XGBoost from genre, budget, year, platform, duration
3. **Genre classifier from description**: Multiclass NLP (BERT/fasttext)
4. **Platform strategy analysis**: Netflix vs Disney+ content mix comparison
5. **International rise**: Track non-US content share over time
6. **Budget vs rating**: Does spending more equal better reviews?
7. **Movie vs TV Show**: Compare average ratings, awards, longevity
8. **Director & cast network graphs**: Co-occurrence analysis

---

## ⚠️ Disclaimer

This is a **synthetically generated dataset** for educational purposes. Platform names, real actor/director names, and production companies appear for realism but all ratings, budgets, and title details are simulated. Not affiliated with Netflix, Amazon, Disney, or any streaming service. Do not cite as real catalog data.

---

## 📜 License

CC BY-SA 4.0
