import pandas as pd
import networkx as nx

from src.base_graph import BASE_GRAPH
from parameters import EPOCH, HORIZON, WEATHER_FEATURES
from src.preprocess.disruptions import DISRUPTIONS
from src.preprocess.weather import WEATHER


# [NOTE]
T = int((HORIZON - EPOCH).total_seconds() / 3600)

SNAPSHOTS: dict[int, nx.Graph] = { }
for t in range(T):
	G = BASE_GRAPH.copy()

	G.graph['start'] = EPOCH + pd.Timedelta(hours = t)
	G.graph['end'] = G.graph['start'] + pd.Timedelta(hours = 1)

	SNAPSHOTS[t] = G


for _, row in DISRUPTIONS.iterrows():
	# [NOTE]
	if (t := (row['start'] - EPOCH).total_seconds() / 3600) >= 0 and t < T:
		SNAPSHOTS[t].edges[row['from'], row['to']].update({ 'type': 'DISRUPTION', 'duration': row['duration'] })

	else: break

for code, row in WEATHER.iterrows():
	# [NOTE]
	if (t := (row['start'] - EPOCH).total_seconds() / 3600) >= 0 and t < T:
		SNAPSHOTS[t].nodes[code].update(row[WEATHER_FEATURES].to_dict())

	else: break


# [NOTE]
def get_weather(G: nx.Graph, node: str) -> dict[str, float | bool]:
	return { k: G.nodes[G.nodes[node]['weather_station']][k] for k in WEATHER_FEATURES }
