import networkx as nx

from src.preprocess.connections import CONNECTIONS
from src.preprocess.train_stations import TRAIN_STATIONS
from src.preprocess.weather_stations import WEATHER_STATIONS

# [NOTE]
BASE_GRAPH = nx.Graph()

BASE_GRAPH.add_nodes_from([ (code, row.to_dict() | { 'type': 'TRAIN' }) for code, row in TRAIN_STATIONS.iterrows() ])
BASE_GRAPH.add_nodes_from([ (code, row.to_dict() | { 'type': 'WEATHER' }) for code, row in WEATHER_STATIONS.iterrows() ])

for _, row in CONNECTIONS.iterrows():
	BASE_GRAPH.add_edge(row['from'], row['to'], type = 'TRAIN')

for code, row in TRAIN_STATIONS.iterrows():
	BASE_GRAPH.add_edge(code, row['weather_station'], type = 'WEATHER')


TRAIN_STATION_NODES = sorted([ n for n, t in BASE_GRAPH.nodes.data('type') if t == 'TRAIN' ])
WEATHER_STATION_NODES = sorted([ n for n, t in BASE_GRAPH.nodes.data('type') if t == 'WEATHER' ])
