# 🛒 E-Commerce Customer Behavior & Sales (2020–2026)

## Overview

**25,000 orders** from **8,000 customers** across **20 countries** and **14 product categories** — with complete purchase history, churn labels, RFM signals, session data, and monthly revenue trends spanning 6 years.

---

## 🎯 Use Cases

- **Churn prediction**: Binary classification using RFM + behavioral features
- **Customer segmentation**: K-Means / RFM-based clustering
- **Revenue forecasting**: Time series prediction of monthly GMV
- **Basket analysis**: Association rule mining for cross-sell
- **CLV prediction**: Customer lifetime value regression
- **Return rate analysis**: Predict returns from product/customer features
- **Discount impact**: Measure elasticity and effect on revenue
- **Cohort analysis**: Track retention by registration quarter

---

## 📁 Dataset Files

| File | Records | Description |
|------|---------|-------------|
| `customers.csv` | 8,000 | Customer profiles, RFM metrics, acquisition channel, churn label |
| `orders.csv` | 25,000 | Individual transactions with pricing, discounts, shipping, session data |
| `product_summary.csv` | 140 | Product-level aggregated sales, ratings, and return rates |
| `monthly_revenue.csv` | 75 | Monthly revenue, AOV, discount, and retention metrics |

---

## 🚀 Quick Start

```python
import pandas as pd
import matplotlib.pyplot as plt

customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")

# Churn rate by membership tier
customers.groupby('membership_tier')['churned'].mean().plot.bar()
plt.title("Churn Rate by Membership Tier")
plt.show()

# Revenue by category
orders[orders['order_status']=='Delivered'].groupby('category')['total_amount_usd'].sum().sort_values().plot.barh()
plt.title("Total Revenue by Category")
plt.show()
```

---

## 💡 Suggested Experiments

1. **Churn classifier**: XGBoost/LightGBM from `days_since_last_purchase`, `total_orders`, RFM
2. **RFM segmentation**: Score customers on Recency, Frequency, Monetary value
3. **CLV regression**: Predict `total_spend_usd` from early customer signals
4. **Market basket analysis**: `mlxtend` Apriori on orders per customer
5. **Revenue forecasting**: Prophet / LSTM on monthly revenue
6. **Return rate model**: Predict `returned` from product, price, and customer features
7. **Cohort retention heatmap**: Track 12-month retention by registration cohort
8. **Discount elasticity**: Does discounting increase or cannibalize revenue?

---

## ⚠️ Disclaimer

Synthetically generated for educational and ML practice purposes. All customer names, products, and transactions are simulated. Not from any real retailer.

---

## 📜 License

CC BY-SA 4.0
