import pandas as pd
import networkx as nx

from base_graph import BASE_GRAPH

# [NOTE]
nodes = sorted(BASE_GRAPH.nodes)
df = pd.DataFrame(nx.floyd_warshall_numpy(BASE_GRAPH, nodelist = nodes), index = nodes, columns = nodes, dtype = int)


DISTANCES = df
DISTANCES.to_csv('./data/distances.csv', index = False)
