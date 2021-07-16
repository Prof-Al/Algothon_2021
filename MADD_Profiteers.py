import numpy as np
import pandas as pd

def getMyPosition(prc_history):
    lookback = 10
    dlrPosLimit = 10000
    
    # Initialize stock allocation vector
    final_alloc = np.zeros(len(prc_history))
    
    # Loop through last 50 stocks
    for i in range(50, len(prc_history)):
        
        # Generate required indicators (standard deviation, exponential moving average)
        stock_price = pd.DataFrame(prc_history[i])
        stock_pct_change = stock_price.pct_change()
        stock_pct_cum = stock_pct_change.cumsum()
        std = stock_pct_change.std()
        stock_rolling_eav = stock_pct_cum.ewm(adjust=False, span=lookback).mean()
        
        # Compute ideal allocation level
        allocation_eav = (stock_rolling_eav - stock_pct_cum) / std
        curr_alloc = allocation_eav.iloc[-1].values[0]
        
        # Restrict unprofitable trades with low probablities of success
        if curr_alloc > 0.05:
            curr_alloc = 1
        elif curr_alloc < -0.12:
            curr_alloc = -1
        else:
            curr_alloc = allocation_eav.iloc[-2].values[0]
        
        final_alloc[i] = (curr_alloc * dlrPosLimit) // stock_price.iloc[-1]
    return final_alloc
