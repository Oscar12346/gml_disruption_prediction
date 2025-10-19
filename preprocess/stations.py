import pandas as pd

df = pd.read_csv('./data/raw/stations/2023-09.csv', usecols = ['code', 'name_long', 'country', 'geo_lat', 'geo_lng'], index_col = 'code', na_filter = False)
df = df.rename(columns = { 'name_long': 'name', 'geo_lat': 'lat', 'geo_lng': 'lng' })

# [NOTE] Filter out any stations not in The Netherlands
df = df[df['country'] == 'NL']
df = df.drop(columns = ['country'])


STATIONS = df
STATIONS.to_csv('./data/stations.csv')
