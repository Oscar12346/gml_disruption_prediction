import networkx as nx

from preprocess.connections import CONNECTIONS
from preprocess.train_stations import TRAIN_STATIONS
from preprocess.weather_stations import WEATHER_STATIONS

# [NOTE]
graph = nx.Graph()

graph.add_nodes_from([ (code, row.to_dict() | { 'type': 'TRAIN' }) for code, row in TRAIN_STATIONS.iterrows() ])
graph.add_nodes_from([ (code, row.to_dict() | { 'type': 'WEATHER' }) for code, row in WEATHER_STATIONS.iterrows() ])

for _, row in CONNECTIONS.iterrows():
	graph.add_edge(row['from'], row['to'], type = 'TRAIN')

for code, row in TRAIN_STATIONS.iterrows():
	graph.add_edge(code, row['weather_station'], type = 'WEATHER')


BASE_GRAPH = graph.subgraph([ n for n, t in graph.nodes.data('type') if t == 'TRAIN' ])
COMBINED_GRAPH = graph
