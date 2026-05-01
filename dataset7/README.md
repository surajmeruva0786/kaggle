# ₿ Cryptocurrency Market Intelligence (2018–2026)

## Overview

**3,834 monthly records** covering **50 major cryptocurrencies** across **23 categories** from January 2018 to March 2026 — spanning 3 complete market cycles (2018 bear, 2020–21 bull, 2022 bear, 2023–24 bull). Includes OHLCV price data, market cap, dominance, on-chain metrics (active addresses, transaction counts), social sentiment scores, Fear & Greed Index, and exchange netflows.

**Special events captured:** Takata-level collapses (LUNA/UST May 2022, FTX/FTT Nov 2022), the AI crypto boom (2023–24), Bitcoin ETF approval surge (2024), and the rise of meme coins (PEPE, WIF, BONK).

---

## 🎯 Use Cases

- **Price prediction**: LSTM / Prophet time series forecasting on close prices
- **Cycle analysis**: Identify bull/bear transitions using technical indicators
- **Sentiment vs price**: Does Fear & Greed Index predict next-month returns?
- **Category comparison**: Smart Contracts vs Memes vs Layer 2 — who wins each cycle?
- **Portfolio optimization**: Markowitz efficient frontier on crypto assets
- **Contagion analysis**: How did LUNA/FTX collapses affect other coins?
- **On-chain intelligence**: Does transaction volume predict price movements?
- **Dominance trends**: BTC dominance shifts across market cycles

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `crypto_monthly_prices.csv` | 3,834 | Monthly OHLCV + market cap + on-chain + sentiment per coin |
| `coin_metadata.csv` | 50 | Coin profiles with ATH, max drawdown, avg returns, metadata |
| `global_market_summary.csv` | 99 | Monthly global crypto market aggregates |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("crypto_monthly_prices.csv")
btc = df[df['symbol'] == 'BTC'].set_index('date')

btc['close_usd'].plot(figsize=(14,5), color='#F7931A')
plt.title("Bitcoin Price 2018–2026")
plt.show()

# Correlation: sentiment vs next month return
df_sorted = df.sort_values(['symbol','date'])
df_sorted['next_return'] = df_sorted.groupby('symbol')['monthly_return_pct'].shift(-1)
print(df_sorted[['social_sentiment_score','fear_greed_index','next_return']].corr())
```

---

## 💡 Suggested Experiments

1. **LSTM price forecasting**: Predict BTC/ETH close using sliding windows
2. **Regime detection**: Hidden Markov Model for bull/bear/neutral regimes
3. **Fear & Greed signal**: Does extreme fear (< 20) = buy signal?
4. **Cross-correlation**: Does BTC dominance predict altcoin performance?
5. **Portfolio backtest**: Equal-weight vs BTC-heavy vs category-diversified
6. **Sentiment momentum**: Can sentiment score predict 30-day returns?
7. **On-chain divergence**: Price up + transactions down = bearish signal?
8. **Collapse detection**: Anomaly detection on FTT/LUNA before collapse

---

## ⚠️ Disclaimer

Synthetically generated using realistic market cycle distributions calibrated to real historical crypto price patterns. LUNA and FTT collapse events are modeled. Not from CoinGecko, CoinMarketCap, or any exchange API. For educational and ML practice only — not for actual trading decisions.

---

## 📜 License

CC BY-SA 4.0
