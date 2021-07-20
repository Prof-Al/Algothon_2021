import numpy as np
import pandas as pd

def getMyPosition(prc_history):
    lookback = 10
    dlrPosLimit = 10000

    # Initialize stock allocation vector
    final_alloc = np.zeros(len(prc_history))

    # Loop through last 50 stocks
    for i in range(50, 100):

        # Generate required indicators (standard deviation, exponential moving average)
        stock_price = pd.DataFrame(prc_history[i])
        stock_pct_change = stock_price.pct_change()
        stock_pct_cum = stock_pct_change.cumsum()
        std = stock_pct_change.std()
        stock_rolling_eav = stock_pct_cum.ewm(adjust=False, span=lookback).mean()

        # Compute ideal allocation level
        allocation_eav = (stock_rolling_eav - stock_pct_cum) / std
        curr_alloc = allocation_eav.iloc[-1].values[0]

        lookback_day = 1

        # Restrict unprofitable trades with low probabilities of success
        while curr_alloc not in (1, -1):
            if allocation_eav.iloc[-lookback_day].values[0] > 0.05:
                curr_alloc = 1
            elif allocation_eav.iloc[-lookback_day].values[0] < -0.12:
                curr_alloc = -1
            else:
                # Check previous days for the last time the allocation was above the confidence bounds
                lookback_day += 1

        final_alloc[i] = (curr_alloc * dlrPosLimit) // stock_price.iloc[-1]
    return final_alloc
