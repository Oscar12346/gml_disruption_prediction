from glob import glob
import pandas as pd

from src.preprocess.distances import DISTANCES
from src.preprocess.train_stations import TRAIN_STATIONS

# [NOTE]
dfs = [ pd.read_csv(f, usecols = ['rdt_station_codes', 'cause_en', 'statistical_cause_en', 'start_time', 'end_time'], na_filter = False) for f in glob('./data/raw/disruptions/*.csv') ]
df = pd.concat(dfs, ignore_index = True)

df = df.rename(columns = { 'rdt_station_codes': 'codes', 'cause_en': 'cause', 'statistical_cause_en': 'original_cause', 'start_time': 'start', 'end_time': 'end' })

# [NOTE] Filter out any values that are NaN
df = df.dropna(subset = ['codes', 'start', 'end'])

# [NOTE] Merge the causes into a single column, avoiding any duplicates
df['cause'] = df['cause'].where(df['cause'] == df['original_cause'], df['cause'] + ', ' + df['original_cause'])
df = df.drop(columns = ['original_cause'])

# [TODO] Filter out any unrelated causes

# [NOTE] Convert station codes into list of Dutch station codes and date strings into datetimes
df['codes'] = df['codes'].str.split(', ').apply(lambda cs: [ c for c in cs if c in TRAIN_STATIONS.index ])
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])

# [NOTE] Compute the exact duration of a disruption
df['duration'] = (df['end'] - df['start']).dt.total_seconds() / 60

# [TODO] Filter out disruptions on a single station or with incorrect timestamps

# [NOTE] Explode disruptions for a list of station codes into a disruption per station code
rows = []
for _, row in df.iterrows():
	other = { c: row[c] for c in df.columns if c != 'codes' }
	visited = set()

	# [NOTE]
	for u in row['codes']:
		if (neighbours := set(TRAIN_STATIONS.loc[u, 'neighbours']).intersection(row['codes'])): # type: ignore
			visited.add(u)
			rows.extend([ { 'from': u, 'to': v, **other } for v in neighbours - visited ])

	# [NOTE]
	for u in set(row['codes']) - visited:
		target = DISTANCES.loc[u, list(visited) or row['codes']].idxmin()
		visited.add(u)

		while u != target:
			closest = DISTANCES.loc[TRAIN_STATIONS.loc[u, 'neighbours'], target].idxmin()
			visited.add(u)

			rows.append({ 'from': u, 'to': closest, **other })
			u = closest

df = pd.DataFrame(rows)

# [NOTE] Segment disruptions from exact timestamps into hours
rows = []
for _, row in df.iterrows():
	other = { c: row[c] for c in df.columns if c not in ['start', 'end', 'duration'] }

	current = row['start'].floor('h')
	last = row['end'].ceil('h')

	while current < last:
		next = current + pd.Timedelta(hours = 1)

		# [NOTE] Compute number of minutes of disruption that occurred within this time slot
		duration = (min(next, row['end']) - max(row['start'], current)).total_seconds() / 60

		rows.append({ 'start': current, 'end': next, 'duration': duration, **other })
		current = next

df = pd.DataFrame(rows)

# [NOTE] Aggregate disruptions between same stations and within same hour
df = df.groupby(['start', 'end', 'from', 'to'], as_index = False).agg({ 'duration': 'sum', 'cause': lambda cs: ', '.join(sorted(set(cs))) })


DISRUPTIONS = df
DISRUPTIONS.to_csv('./data/disruptions.csv', index = False)
