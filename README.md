# Trader Performance vs Market Sentiment

## Objective
Analyze how market sentiment (Fear vs Greed) impacts trader behavior and performance.

## Methodology
- Cleaned and merged sentiment + trading data
- Created features like win rate, leverage buckets, trade size
- Aggregated daily metrics
- Performed segmentation of traders
- Visualized patterns using plots

## Key Insights
1. Traders increase leverage during Greed phases
2. Higher activity does not guarantee better performance
3. High leverage leads to more volatile returns

## Strategy Recommendations
- Reduce leverage during Fear periods
- Avoid overtrading during Greed spikes
- Focus on consistent traders over high-frequency traders

## How to Run
pip install -r requirements.txt
python notebook/analysis.py