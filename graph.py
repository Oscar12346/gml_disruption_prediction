import pandas as pd
import networkx as nx

from preprocess.connections import CONNECTIONS
from preprocess.disruptions import DISRUPTIONS
from preprocess.stations import STATIONS


# [NOTE]
EPOCH = pd.Timestamp('2023-01-01 00:00:00')
HORIZON = pd.Timestamp('2024-01-01 00:00:00')


# [NOTE]
BASE_GRAPH = nx.Graph()
BASE_GRAPH.add_nodes_from([ (code, row.to_dict()) for code, row in STATIONS.iterrows() ])

for _, row in CONNECTIONS.iterrows():
	BASE_GRAPH.add_edge(row['from'], row['to'])


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
	if (t := (row['start'] - EPOCH).total_seconds() / 3600) < T:
		SNAPSHOTS[t].add_edge(row['from'], row['to'], weight = row['duration'])

	else: break
