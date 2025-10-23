import pandas as pd

# [NOTE] Load all connection data but avoid interpreting 'NaN-like' values as such
df = pd.read_csv('./data/raw/connections/2025.csv', na_filter = False)


CONNECTIONS = df
CONNECTIONS.to_csv('./data/connections.csv', index = False)
