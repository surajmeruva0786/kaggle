# 🌾 Global Food Security & Nutrition Index (2015–2025)

## Overview

**1,155 country-year records** covering **105 countries** over **11 years** — combining the Global Food Security Index (GFSI) with key nutrition indicators (undernourishment, stunting, wasting, obesity), agricultural metrics (cereal yield, water stress, food waste), and climate vulnerability scores.

---

## 🎯 Use Cases

- **Food security prediction**: Regression from income, climate, agriculture to GFSI
- **Time series analysis**: Track country trajectories and detect deterioration
- **Nutrition triple burden**: Model the co-existence of stunting, wasting, and obesity
- **Climate-food linkage**: Correlate climate vulnerability with food insecurity
- **COVID-19 impact**: Quantify 2020 food security shock across income groups
- **Country clustering**: Group nations by food security profile (K-Means / DBSCAN)
- **Policy impact modeling**: Does government ag spending improve outcomes?
- **Regional disparities**: Sub-Saharan Africa vs South Asia vs Latin America comparison

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `food_security_annual.csv` | 1,155 | Annual country-level GFSI scores + all sub-indicators |
| `food_security_2025_rankings.csv` | 105 | Latest (2025) rankings snapshot with rank column |
| `global_yearly_summary.csv` | 11 | Global averages per year (2015–2025) |
| `regional_summary_2025.csv` | 18 | Regional aggregates for 2025 |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("food_security_annual.csv")
rankings = pd.read_csv("food_security_2025_rankings.csv")

# Top 15 most food secure (2025)
rankings.nsmallest(15,'rank').plot.barh(x='country', y='gfsi_score')
plt.title("Top 15 Most Food Secure Countries (2025)")
plt.show()

# Undernourishment trend
df.groupby('year')['prevalence_of_undernourishment_pct'].mean().plot()
plt.title("Global Undernourishment Rate Over Time")
plt.show()
```

---

## 💡 Suggested Experiments

1. **GFSI predictor**: XGBoost regression from income, climate risk, cereal yield, ag spend
2. **COVID shock analysis**: Compare 2019 vs 2020 across income groups
3. **Triple burden mapping**: Countries with high stunting AND high obesity
4. **Climate risk regression**: Does water stress predict undernourishment independently?
5. **Progress leaders**: Which low-income countries improved fastest 2015–2025?
6. **Food waste paradox**: Do high-waste countries have lower food security?
7. **Geospatial choropleth**: Map GFSI and undernourishment globally
8. **Income group convergence**: Is the food security gap narrowing over time?

---

## ⚠️ Disclaimer

Synthetically generated using distributions calibrated to EIU Global Food Security Index, FAO undernourishment data, and World Bank nutrition reports. All individual data points are simulated. Not from official FAO, EIU, or WHO databases.

---

## 📜 License

CC BY-SA 4.0
