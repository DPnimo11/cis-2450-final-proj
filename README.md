# updates
- switched to finbert w/ gpu
- more tickers
- change to left join and forward fill to get posts outside of trading hours

# stuff to implement

Hybrid Model (to fix class imbalance):
The Overnight Model (Strategy 1): Aggregate all sentiment from 4:00 PM to 9:30 AM. Use this to predict the Overnight Return (Open price at 9:30 AM minus Close price at 4:00 PM the previous day).
The Intraday Model (Strategy 2): For hours between 9:30 AM and 4:00 PM, use the sentiment at hour $t$ to predict the return at hour $t+1$. (maybe try ema or rolling average or smth, also pool within hour)

maybe address neutral signals w/ z-scores or bull index