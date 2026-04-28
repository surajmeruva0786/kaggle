# 🏅 Olympic Games Performance & Medal History (1992–2026)

## Overview

**18 Olympic Games** (Summer & Winter, 1992–2026) with complete medal tables, **8,000 athlete profiles**, and country-level socioeconomic metadata — enabling analysis of host-nation effects, GDP–medal correlations, athlete career arcs, and the rise of new sporting nations.

---

## 🎯 Use Cases

- **Medal count prediction**: Regression from GDP, population, host status, sports investment
- **Host nation effect**: Quantify how hosting impacts medal outcomes
- **Athlete career analysis**: Games participated vs medals won, peak age by sport
- **Country clustering**: Group nations by sporting profile and medal efficiency
- **Gender gap analysis**: Compare medal rates across male/female athletes
- **Sport-level analysis**: Which disciplines deliver the most medals per nation?
- **Doping impact**: How doping violations correlate with competitive outcomes
- **Winter vs Summer**: Which nations specialize in each season?

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `medal_results.csv` | 910 | Country × Games medal tables with socioeconomic metadata |
| `athletes.csv` | 8,000 | Individual athlete profiles with medals, physicals, career span |
| `games_summary.csv` | 18 | Per-Games statistics (athletes, nations, medals awarded) |
| `alltime_country_rankings.csv` | 60 | All-time country medal rankings with Summer/Winter split |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

medals = pd.read_csv("medal_results.csv")
athletes = pd.read_csv("athletes.csv")

# All-time medal leaders
medals.groupby('country')['gold'].sum().nlargest(15).sort_values().plot.barh()
plt.title("All-Time Gold Medals by Country")
plt.show()

# Host nation effect
host_boost = medals.groupby('host_nation')['total_medals'].mean()
print(host_boost)
```

---

## 💡 Suggested Experiments

1. **Medal prediction**: XGBoost from GDP tier, population, host status, prior medals
2. **Host effect analysis**: DiD (Difference-in-Differences) before/during/after hosting
3. **Athlete physique clustering**: K-Means on height/weight/BMI by sport
4. **Career longevity**: Survival analysis — how many Games do athletes attend?
5. **Medal efficiency**: Which small countries punch above their weight?
6. **GDP vs medals scatter**: Correlation with log(GDP) per capita
7. **Summer vs Winter specialists**: PCA on country sport profiles
8. **Medal points Elo ranking**: Build a dynamic country ranking system

---

## ⚠️ Disclaimer

Synthetically generated for educational purposes. Medal counts, athlete names, and records are simulated based on realistic historical distributions. Not from the IOC, Olympic databases, or any official sports body.

---

## 📜 License

CC BY-SA 4.0
