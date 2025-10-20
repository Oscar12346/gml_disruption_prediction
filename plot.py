from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx

from graph import BASE_GRAPH, SNAPSHOTS


# [NOTE]
plot_options = {
	'node_size': 50,
	'pos': { node: (data['lng'], data['lat']) for node, data in BASE_GRAPH.nodes(data = True) },
}


fig, ax = plt.subplots()

# [NOTE]
def plot_snapshot(frame: int) -> None:
	ax.clear()
	ax.set_title(str(frame))

	edge_colours = [ 'red' if w else 'black' for _, _, w in SNAPSHOTS[frame].edges.data('weight') ]

	nx.draw(SNAPSHOTS[frame], edge_color = edge_colours, **plot_options)

_ = FuncAnimation(fig, plot_snapshot, frames = len(SNAPSHOTS.keys()), interval = 250, repeat = True) # type: ignore

# [NOTE]
plt.tight_layout()
plt.show()
