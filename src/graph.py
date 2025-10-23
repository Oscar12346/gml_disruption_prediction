import pandas as pd
import networkx as nx

from src.base_graph import BASE_GRAPH
from parameters import EPOCH, HORIZON, WEATHER_FEATURES
from src.preprocess.disruptions import DISRUPTIONS
from src.preprocess.weather import WEATHER


SNAPSHOTS: dict[int, nx.Graph] = { }
for t in range(int((HORIZON - EPOCH).total_seconds() / 3600)):
	G = BASE_GRAPH.copy()

	G.graph['start'] = EPOCH + pd.Timedelta(hours = t)
	G.graph['end'] = G.graph['start'] + pd.Timedelta(hours = 1)

	SNAPSHOTS[t] = G

# [NOTE] Include all disruptions that occur within the configured time window
for _, row in DISRUPTIONS[(DISRUPTIONS['start'] >= EPOCH) & (DISRUPTIONS['end'] <= HORIZON)].iterrows():
	t = (row['start'] - EPOCH).total_seconds() / 3600
	SNAPSHOTS[t].edges[row['from'], row['to']].update({ 'type': 'DISRUPTION', 'duration': row['duration'] })

# [NOTE] Include all weather measurements that occur within the configured time window
for code, row in WEATHER[(WEATHER['start'] >= EPOCH) & (WEATHER['end'] <= HORIZON)].iterrows():
	t = (row['start'] - EPOCH).total_seconds() / 3600
	SNAPSHOTS[t].nodes[code].update(row[WEATHER_FEATURES].to_dict())


# [NOTE] Helper function to obtain the node's associated weather measurements in a given graph
def get_weather(G: nx.Graph, node: str) -> dict[str, float | bool]:
	return { k: G.nodes[G.nodes[node]['weather_station']][k] for k in WEATHER_FEATURES }
