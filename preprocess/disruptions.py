import pandas as pd

from preprocess.stations import STATIONS

df = pd.read_csv('./data/raw/disruptions/2023.csv', usecols = ['rdt_station_codes', 'cause_en', 'statistical_cause_en', 'start_time', 'end_time'], na_filter = False)
df = df.rename(columns = { 'rdt_station_codes': 'codes', 'cause_en': 'cause', 'statistical_cause_en': 'original_cause', 'start_time': 'start', 'end_time': 'end' })

# [NOTE] Filter out any values that are NaN
df = df.dropna(subset = ['codes', 'start', 'end'])

# [NOTE] Merge the causes into a single column, avoiding any duplicates
df['cause'] = df['cause'].where(df['cause'] == df['original_cause'], df['cause'] + ', ' + df['original_cause'])
df = df.drop(columns = ['original_cause'])

# [TODO] Filter out any unrelated causes

# [NOTE] Convert station codes into list of codes and date strings into datetimes
df['codes'] = df['codes'].str.split(', ')
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])

# [NOTE] Compute the exact duration of a disruption
df['duration'] = (df['end'] - df['start']).dt.total_seconds() / 60

# [NOTE] Explode disruptions for a list of station codes into a disruption per station code
rows = []
for _, row in df.iterrows():
	other = { c: row[c] for c in df.columns if c != 'codes' }
	rows.extend([ { 'from': row['codes'][i], 'to': row['codes'][i + 1], **other } for i in range(len(row['codes']) - 1) ])

df = pd.DataFrame(rows)

# [TODO] Confirm all dropped edges are to stations abroad
# [NOTE] Filter out any disruptions between train stations not in The Netherlands
df = df[df['from'].isin(STATIONS.index) & df['to'].isin(STATIONS.index)]

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
DISRUPTIONS.to_csv('./data/disruptions.csv')
