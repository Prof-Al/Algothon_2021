def getMyPosition(prc_history):
    lookback = 10
    final_alloc = np.zeros(len(prc_history))

    for i in range(50, len(prc_history)):
        stock_price = pd.DataFrame(prc_history[i])
        stock_pct_change = stock_price.pct_change()
        stock_pct_cum = stock_pct_change.cumsum()
        std = stock_pct_change.std()
        stock_rolling_eav = stock_pct_cum.ewm(adjust=False, span=lookback).mean()

        allocation_eav = (stock_rolling_eav - stock_pct_cum) / std
        curr_alloc = allocation_eav.iloc[-1].values[0]

        if curr_alloc > 0.05:
            curr_alloc = 1
        elif curr_alloc < -0.12:
            curr_alloc = -1
        else:
            curr_alloc = allocation_eav.iloc[-2].values[0]

        final_alloc[i] = (curr_alloc * dlrPosLimit) // stock_price.iloc[-1]
    return final_alloc
