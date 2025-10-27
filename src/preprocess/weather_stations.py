import pandas as pd

from src.preprocess.weather import INCOMPLETE_WEATHER_STATIONS

df = pd.read_csv('./data/raw/weather_stations/2023.csv', usecols = ['STN', 'LON', 'LAT', 'NAME'], index_col = 'STN')

df.index = df.index.rename('code')
df = df.rename(columns = { 'LON': 'lng', 'LAT': 'lat', 'NAME': 'name' })[['name', 'lat', 'lng']]

# [NOTE] Drop all weather stations which have incomplete measurements for the desired weather metrics
df = df[~df.index.isin(INCOMPLETE_WEATHER_STATIONS)]


WEATHER_STATIONS = df
WEATHER_STATIONS.to_csv('./data/weather_stations.csv')
