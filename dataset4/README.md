# 🌍 Global Earthquake & Natural Disasters (1990–2026)

## Overview

**15,000 disaster events** across two complementary tables — a detailed **earthquake catalog** (5,000 events, 1990–2026) and a **multi-hazard disaster database** (10,000 events, 2000–2026) covering floods, cyclones, wildfires, landslides, droughts, tsunamis, volcanic eruptions, tornadoes, and avalanches. Includes geospatial coordinates, casualties, economic damage, and climate attribution flags.

---

## 🎯 Use Cases

- **Earthquake magnitude regression**: Predict casualties/damage from magnitude, depth, location
- **Disaster severity classification**: Classify Minor/Moderate/Major/Catastrophic from features
- **Geospatial mapping**: Heatmaps of seismic activity and disaster hotspots (Folium/Plotly)
- **Time series analysis**: Track disaster frequency and damage trends over decades
- **Climate attribution analysis**: Are extreme events becoming more frequent?
- **Tsunami prediction**: Binary classification — which earthquakes generate tsunamis?
- **Economic impact modeling**: Regression from disaster features to USD damage
- **International aid trigger**: Predict when events cross the humanitarian response threshold

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `earthquakes_1990_2026.csv` | 5,000 | Detailed earthquake catalog with seismological data |
| `natural_disasters_2000_2026.csv` | 10,000 | Multi-hazard disaster events across all types |
| `disaster_type_summary.csv` | 10 | Aggregated stats per disaster type |
| `country_summary.csv` | ~45 | Country-level disaster impact profiles |
| `yearly_trends.csv` | 27 | Year-by-year global disaster metrics |
| `earthquake_magnitude_summary.csv` | 6 | Earthquake statistics by magnitude band |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

eq = pd.read_csv("earthquakes_1990_2026.csv")
dis = pd.read_csv("natural_disasters_2000_2026.csv")

# Magnitude distribution
eq['magnitude'].plot.hist(bins=30, color='#e74c3c')
plt.title("Earthquake Magnitude Distribution")
plt.show()

# Deaths by disaster type
dis.groupby('disaster_type')['deaths'].sum().sort_values().plot.barh()
plt.title("Total Deaths by Disaster Type (2000–2026)")
plt.show()
```

---

## 💡 Suggested Experiments

1. **Tsunami classifier**: Predict `tsunami_generated` from magnitude, depth, mechanism
2. **Casualty regression**: XGBoost from seismic features to deaths + damage
3. **Severity classifier**: Multi-class from disaster features
4. **Geospatial hotspot**: Folium choropleth of events and damage by country
5. **Climate trend analysis**: Test if climate-attributed event frequency is rising
6. **Magnitude vs depth**: Joint distribution analysis and depth zone classification
7. **Aftershock modelling**: Predict aftershock count from main shock magnitude (Omori's Law)
8. **International aid predictor**: Binary classifier for `international_aid_triggered`

---

## ⚠️ Disclaimer

Synthetically generated for educational and ML purposes. Magnitude distributions follow the Gutenberg-Richter law; casualty/damage distributions are calibrated to real disaster statistics but all individual events are simulated. Not from USGS, EM-DAT, or any official disaster database.

---

## 📜 License

CC BY-SA 4.0
