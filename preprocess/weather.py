from glob import glob
import pandas as pd

# [NOTE]
WEATHER_FEATURES = ['wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'snow', 'thunder', 'ice']

# [NOTE]
dfs = [ pd.read_csv(f, index_col = 'code') for f in glob('./data/raw/weather/*.csv') ]
df = pd.concat(dfs)

# [NOTE]
INCOMPLETE_WEATHER_STATIONS = df[df[WEATHER_FEATURES].isna().any(axis = 'columns')].index.unique()
df = df.dropna(subset = WEATHER_FEATURES)

# [NOTE]
df['start'] = pd.to_datetime(df['date'].astype(str) + (df['hour'] - 1).astype(str).str.zfill(2), format = '%Y%m%d%H')
df['end'] = df['start'] + pd.Timedelta(hours = 1) # type: ignore
df = df.drop(columns = ['date', 'hour'])

# [NOTE]
df['rain'] = df['rain'].replace(-1, 0.5)

# [NOTE]
df[['wind', 'wind_max', 'temperature', 'rain']] = df[['wind', 'wind_max', 'temperature', 'rain']] / 10.0
df['rain_duration'] = df['rain_duration'] * 6
df[['fog', 'snow', 'thunder', 'ice']] = df[['fog', 'snow', 'thunder', 'ice']].astype(bool)

# [NOTE]
df = df[['start', 'end', 'wind', 'wind_max', 'temperature', 'rain', 'rain_duration', 'fog', 'thunder', 'snow', 'ice']]
df = df.sort_values('start', kind = 'stable')


WEATHER = df
WEATHER.to_csv('./data/weather.csv')
