# 💻 Developer Survey 2020–2026: Skills, Salary & AI Impact

## Overview

**12,000 developer responses** from **50 countries** across **7 survey years** — covering programming languages, frameworks, databases, cloud platforms, salaries, AI tool adoption, job satisfaction, burnout, and career outlook.

---

## 🎯 Use Cases

- **Salary prediction**: Regression model from role, experience, country, and tech stack
- **Language & framework trends**: Track rise of Rust, TypeScript, AI tools over time
- **AI impact analysis**: How is AI adoption affecting productivity, sentiment, and job threat perception?
- **Job satisfaction modeling**: What factors predict developer happiness?
- **Burnout analysis**: Correlate burnout with work mode, hours, company size
- **Gender pay gap**: Analyze salary differences controlling for role and experience
- **Country comparison**: Developer ecosystems, salary PPP, and tech preferences by nation
- **Career path analysis**: How do skills, roles, and salaries evolve with experience?

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `developer_survey.csv` | 12,000 | Individual responses — demographics, tech stack, salary, AI, satisfaction |
| `language_trends.csv` | 168 | Language popularity by year (24 languages × 7 years) |
| `salary_by_role_country.csv` | ~500 | Median/avg salary by role × country (min 5 respondents) |
| `ai_adoption_trends.csv` | 7 | AI adoption metrics by year (2020–2026) |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("developer_survey.csv")

# Top 10 languages
all_langs = df['languages_used'].str.split(';').explode()
all_langs.value_counts().head(10).plot.barh()
plt.title("Most Popular Languages")
plt.show()

# Salary by role
df[df['annual_salary_usd'] > 0].groupby('role')['annual_salary_usd'].median().sort_values().plot.barh()
plt.title("Median Salary by Role")
plt.show()
```

---

## 💡 Suggested Experiments

1. **Salary prediction**: XGBoost/LightGBM regression from all features
2. **Language trend forecasting**: Time series on language adoption rates
3. **AI adoption clustering**: K-means on developers by AI usage patterns
4. **Burnout classifier**: Predict burnout from work conditions
5. **Tech stack recommender**: Association rules / collaborative filtering
6. **NLP on want_to_learn**: What technologies are developers most excited about?
7. **Satisfaction regression**: Model job satisfaction from salary, work mode, AI sentiment
8. **Open source analysis**: What differentiates OSS contributors?

---

## ⚠️ Disclaimer

This is a **synthetically generated dataset** for educational purposes. Modeled on patterns from Stack Overflow's Annual Developer Survey but all individual responses are simulated. Not affiliated with Stack Overflow. Do not cite as real survey data.

---

## 📜 License

CC BY-SA 4.0
