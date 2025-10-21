import pandas as pd

from preprocess.connections import CONNECTIONS

df = pd.read_csv('./data/raw/stations/2023-09.csv', usecols = ['code', 'name_long', 'country', 'geo_lat', 'geo_lng'], index_col = 'code', na_filter = False)
df = df.rename(columns = { 'name_long': 'name', 'geo_lat': 'lat', 'geo_lng': 'lng' })

# [NOTE] Filter out any stations not in The Netherlands
df = df[df['country'] == 'NL']
df = df.drop(columns = ['country'])

# [NOTE] Determine the station's neighbours based on connections
neighbours = pd.concat([ CONNECTIONS, CONNECTIONS.rename(columns = { 'to': 'from', 'from': 'to' }) ]).groupby('from')['to']
df['neighbours'] = df.index.map(neighbours.apply(lambda x: sorted(set(x))))


STATIONS = df
STATIONS.to_csv('./data/stations.csv')
