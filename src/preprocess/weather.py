from glob import glob
import pandas as pd

from parameters import WEATHER_FEATURES

dfs = [ pd.read_csv(f, index_col = 'code') for f in glob('./data/raw/weather/*.csv') ]
df = pd.concat(dfs)

# [NOTE] Determine the weather stations which have incomplete measurements for the desired weather metrics
INCOMPLETE_WEATHER_STATIONS = df[df[WEATHER_FEATURES].isna().any(axis = 'columns')].index.unique()
df = df.drop(index = INCOMPLETE_WEATHER_STATIONS)

# [NOTE] Convert the given date and hour into a proper start and end timestamp
df['start'] = pd.to_datetime(df['date'].astype(str) + (df['hour'] - 1).astype(str).str.zfill(2), format = '%Y%m%d%H')
df['end'] = df['start'] + pd.Timedelta(hours = 1) # type: ignore
df = df.drop(columns = ['date', 'hour'])

# [NOTE] Rain measurements which have the value '-1' denote rain has been measured but is < 0.05 mm
df['rain'] = df['rain'].replace(-1, 0.5)

# [NOTE] Convert all weather measurements to their desired units
df[['wind', 'wind_max', 'temperature', 'rain']] = df[['wind', 'wind_max', 'temperature', 'rain']] / 10.0
df['rain_duration'] = df['rain_duration'] * 6
df[['fog', 'snow', 'thunder', 'ice']] = df[['fog', 'snow', 'thunder', 'ice']].astype(bool)

df = df[['start', 'end', 'wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'thunder', 'snow', 'ice']]
df = df.sort_values('start', kind = 'stable')


WEATHER = df
WEATHER.to_csv('./data/weather.csv')
