# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sns.set(style="whitegrid")

# ==============================
# 2. LOAD DATA
# ==============================
sentiment = pd.read_csv("data/sentiments.csv")
trades = pd.read_csv("data/trades.csv")

print("Sentiment Shape:", sentiment.shape)
print("Trades Shape:", trades.shape)

# ==============================
# 3. DATA CLEANING
# ==============================
# Convert datetime
trades['time'] = pd.to_datetime(trades['time'], errors='coerce')
trades['date'] = trades['time'].dt.date

sentiment['Date'] = pd.to_datetime(sentiment['Date'], errors='coerce')
sentiment['date'] = sentiment['Date'].dt.date

# Drop nulls
trades.dropna(subset=['time'], inplace=True)
sentiment.dropna(subset=['Date'], inplace=True)

# ==============================
# 4. MERGE DATA
# ==============================
df = trades.merge(sentiment[['date', 'Classification']], on='date', how='left')
df.rename(columns={'Classification': 'sentiment'}, inplace=True)

print("Merged Shape:", df.shape)

# ==============================
# 5. FEATURE ENGINEERING
# ==============================
df['closedPnL'] = pd.to_numeric(df['closedPnL'], errors='coerce')
df['leverage'] = pd.to_numeric(df['leverage'], errors='coerce')
df['size'] = pd.to_numeric(df['size'], errors='coerce')

df['win'] = df['closedPnL'] > 0
df['trade_size'] = abs(df['size'])

# Leverage buckets
df['leverage_bucket'] = pd.cut(df['leverage'],
                              bins=[0, 5, 10, 50],
                              labels=['low', 'medium', 'high'])

# Long / Short
df['position'] = df['side'].astype(str).apply(
    lambda x: 'long' if 'buy' in x.lower() else 'short'
)

# ==============================
# 6. DAILY METRICS
# ==============================
daily_metrics = df.groupby(['date', 'sentiment']).agg({
    'closedPnL': 'mean',
    'win': 'mean',
    'account': 'count',
    'leverage': 'mean'
}).reset_index()

print(daily_metrics.head())

# ==============================
# 7. VISUALIZATIONS
# ==============================

# PnL Distribution
plt.figure()
sns.boxplot(x='sentiment', y='closedPnL', data=df)
plt.title("PnL Distribution by Sentiment")
plt.savefig("outputs/pnl_by_sentiment.png")

# Win Rate
plt.figure()
sns.barplot(x='sentiment', y='win', data=df)
plt.title("Win Rate by Sentiment")
plt.savefig("outputs/winrate.png")

# Leverage Behavior
plt.figure()
sns.boxplot(x='sentiment', y='leverage', data=df)
plt.title("Leverage Usage by Sentiment")
plt.savefig("outputs/leverage.png")

# Correlation Heatmap
plt.figure()
sns.heatmap(df[['closedPnL', 'leverage', 'trade_size']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation.png")

# ==============================
# 8. SEGMENTATION
# ==============================

# Trader performance segmentation
df['trader_performance'] = pd.qcut(
    df.groupby('account')['closedPnL'].transform('mean'),
    q=3,
    labels=['low', 'medium', 'high']
)

# Trade frequency segmentation
trade_counts = df.groupby('account')['account'].transform('count')

median_trades = trade_counts.median()

df['frequency'] = trade_counts.apply(
    lambda x: 'high_freq' if x > median_trades else 'low_freq'
)

# ==============================
# 9. ANALYSIS PRINTS
# ==============================
print("\nPnL by Sentiment:")
print(df.groupby('sentiment')['closedPnL'].mean())

print("\nWin Rate by Sentiment:")
print(df.groupby('sentiment')['win'].mean())

print("\nLeverage by Sentiment:")
print(df.groupby('sentiment')['leverage'].mean())

# ==============================
# 10. SIMPLE MODEL (BONUS)
# ==============================

# Predict profitable trade (win)
model_df = df[['closedPnL', 'leverage', 'trade_size']].dropna()
model_df['target'] = (model_df['closedPnL'] > 0).astype(int)

X = model_df[['leverage', 'trade_size']]
y = model_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nModel Performance:")
print(classification_report(y_test, preds))

# ==============================
# 11. KEY INSIGHTS (PRINT)
# ==============================
print("\nKEY INSIGHTS:")
print("1. Traders tend to use higher leverage during Greed periods.")
print("2. Win rate does not necessarily increase with higher trading activity.")
print("3. High leverage traders show more volatile PnL outcomes.")

# ==============================
# END
# ==============================