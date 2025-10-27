import numpy as np
import pandas as pd

from src.preprocess.connections import CONNECTIONS
from src.preprocess.weather_stations import WEATHER_STATIONS

df = pd.read_csv('./data/raw/train_stations/2023-09.csv', usecols = ['code', 'name_long', 'country', 'geo_lat', 'geo_lng'], index_col = 'code', na_filter = False)
df = df.rename(columns = { 'name_long': 'name', 'geo_lat': 'lat', 'geo_lng': 'lng' })

# [NOTE] Filter out any stations not in The Netherlands
df = df[df['country'] == 'NL']
df = df.drop(columns = ['country'])

# [NOTE] Determine the station's neighbours based on connections
neighbours = pd.concat([ CONNECTIONS, CONNECTIONS.rename(columns = { 'to': 'from', 'from': 'to' }) ]).groupby('from')['to']
df['neighbours'] = df.index.map(neighbours.apply(lambda x: sorted(set(x))))

# [NOTE] Compute the closest weather station for some train station based on haversine distance
def find_closest_weather_station(row) -> str:
	def haversine(lat, lng, lat_, lng_) -> int:
		lat, lng, lat_, lng_ = map(np.radians, [lat, lng, lat_, lng_])
		a = np.sin((lat_ - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat_) * np.sin((lng_ - lng) / 2) ** 2
		return 6371 * 2 * np.arcsin(np.sqrt(a))

	distances = haversine(row['lat'], row['lng'], WEATHER_STATIONS['lat'], WEATHER_STATIONS['lng'])
	return WEATHER_STATIONS.index[np.argmin(distances)] # type: ignore

df['weather_station'] = df.apply(find_closest_weather_station, axis = 'columns')


TRAIN_STATIONS = df
TRAIN_STATIONS.to_csv('./data/train_stations.csv')
