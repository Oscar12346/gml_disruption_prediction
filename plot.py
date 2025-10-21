from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx

from graph import BASE_GRAPH, SNAPSHOTS


# [NOTE]
plot_options = {
	'node_size': 25,
	'pos': { node: (data['lng'], data['lat']) for node, data in BASE_GRAPH.nodes(data = True) },
}


fig, ax = plt.subplots()

# [NOTE]
def plot_snapshot(frame: int) -> None:
	ax.clear()

	G = SNAPSHOTS[frame]

	ax.set_title(f'{G.graph['start']} - {G.graph['end']}')
	edge_colours = [ 'red' if w else 'black' for _, _, w in G.edges.data('weight') ]

	nx.draw(G, edge_color = edge_colours, **plot_options)

_ = FuncAnimation(fig, plot_snapshot, frames = len(SNAPSHOTS.keys()), interval = 25, repeat = True) # type: ignore

# [NOTE]
plt.tight_layout()
plt.show()
