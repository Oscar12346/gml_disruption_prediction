import pandas as pd

df = pd.read_csv('./data/raw/weather_stations/2023.csv', usecols = ['STN', 'LON', 'LAT', 'NAME'], index_col = 'STN')

df.index = df.index.rename('code')
df = df.rename(columns = { 'LON': 'lng', 'LAT': 'lat', 'NAME': 'name' })

df = df[['name', 'lat', 'lng']]


WEATHER_STATIONS = df
WEATHER_STATIONS.to_csv('./data/weather_stations.csv')
