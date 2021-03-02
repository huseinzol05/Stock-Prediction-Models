import pandas as pd

df = pd.read_csv('dataset/uk_stock_prices.csv')
df = df.set_index("Date")
df.head()

price = df['AZN']

import get_actions as ga

MA_action, RA_action, DES_action, RSI_action = ga.get_actions(price)

