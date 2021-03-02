import numpy as np
import pandas as pd
import deep_evolution_strategy as des
import rolling_agent as ra

def get_actions(price):
    ## MA 
    short_window = int(0.025 * len(price))
    long_window = int(0.05 * len(price))
    
    signals = pd.DataFrame(index=price.index)
    signals['signal'] = 0.0
    signals['short_ma'] = price.rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_ma'] = price.rolling(window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:] 
                                                > signals['long_ma'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()
    MA_signals = signals
    
    # MA_date = MA_signals[signals.positions != 0][-1:].index
    MA_sign = MA_signals[signals.positions != 0].positions[-1:]
    
    ## RSI
    RSI_price = price[-100:]
    U = np.mean(RSI_price.pct_change()[price.pct_change() > 0])
    D = np.abs(np.mean(RSI_price.pct_change()[price.pct_change() < 0]))
    RSI = 100 - (100/(1+U/D))
    
    ## Evolution
    close = price.values.tolist()
    window_size = 30
    skip = 1
    initial_money = 10000
    
    model = des.Model(input_size = window_size, layer_size = 500, output_size = 3)
    agent = des.Agent(model = model, 
                  window_size = window_size,
                  trend = close,
                  skip = skip,
                  initial_money = initial_money)
    agent.fit(iterations = 100, checkpoint = 10)
    states_buy, states_sell, total_gains, invest = agent.buy()
    
    buy_str = sum(np.array(states_buy) > len(price)-30)
    sell_str = sum(np.array(states_sell) > len(price)-30)
    des_sign = buy_str - sell_str
    
    ## Rolling Agent 
    states_buy, states_sell, total_gains, invest = ra.buy_stock(price, initial_state = 1, 
                                                             delay = 4, initial_money = 10000)
    states_sell = np.array(states_sell)
    states_buy = np.array(states_buy)
    states_buy= states_buy[(states_buy)>0]
    
    if states_buy[-1] > states_sell[-1]:
        ra_sign = 1
    else:
        ra_sign = -1
    
    ## Output conditioning 
    if MA_sign[0] > 0:
        MA_action = "BUY"
    else:
        MA_action = "SELL"
        
    if RSI > 70:
        RSI_action = "BUY"
    elif RSI < 30:
        RSI_action = "SELL"
    else:
        RSI_action = "HOLD"
        
    if ra_sign > 0:
        RA_action = "BUY"
    else:
        RA_action = "SELL"
        
    if des_sign > 1:
        DES_action = "BUY"
    elif des_sign < -1:
        DES_action = "SELL"
    else:
        DES_action = "HOLD"
        
    return MA_action, RA_action, DES_action, RSI_action