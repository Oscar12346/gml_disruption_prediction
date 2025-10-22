import networkx as nx

from preprocess.connections import CONNECTIONS
from preprocess.train_stations import TRAIN_STATIONS

# [NOTE]
BASE_GRAPH = nx.Graph()
BASE_GRAPH.add_nodes_from([ (code, row.to_dict()) for code, row in TRAIN_STATIONS.iterrows() ])

for _, row in CONNECTIONS.iterrows():
	BASE_GRAPH.add_edge(row['from'], row['to'])
