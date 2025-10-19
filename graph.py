import matplotlib.pyplot as plt
import networkx as nx

from preprocess.connections import CONNECTIONS
from preprocess.stations import STATIONS


# [NOTE]
G = nx.Graph()
G.add_nodes_from([ (code, row.to_dict()) for code, row in STATIONS.iterrows() ])

for _, row in CONNECTIONS.iterrows():
	G.add_edge(row['from'], row['to'])


# [NOTE]
nx.draw(G, pos = { node: (data['lng'], data['lat']) for node, data in G.nodes(data = True) }, with_labels = True)
plt.show()
