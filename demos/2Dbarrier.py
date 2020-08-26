from scipy.stats import unitary_group
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Import common files
import sys
sys.path.append('../common')
from network import ScatteringNetwork, Node


# Encode a complex number x as an RGB tuple for visualization
def complex_to_rgb(x):
    mag = np.absolute(x)
    arg = np.angle(x)+np.pi
    return colorsys.hls_to_rgb(0, (2/np.pi)*np.arctan(5*mag), 1)

def draw(lattice):
    pos = {node: node.metadata['position'] for node in lattice.network.nodes}
    edge_colors = [complex_to_rgb(x[2]['amplitude']) for x in lattice.network.edges(data=True)]
    node_colors = [x.metadata['color'] for x in lattice.network]
    nx.draw(lattice.network, pos, connectionstyle='arc3, rad = 0.1', node_color=node_colors, edge_color=edge_colors, width=2)

def onclick(event):
    plt.clf()
    lat.step()
    draw(lat)
    plt.draw()


lat = ScatteringNetwork()
np.random.seed(0)
multiport_matrix1 = unitary_group.rvs(4)
multiport_matrix2 = unitary_group.rvs(4)
mirror_matrix = np.array([[1]])
nodes = [([None]*10) for _ in range(10)]

for i in range(0, 10):
    for j in range(0, 10):
        if not (((i == 0) or (i == 9)) and ((j == 0) or (j == 9))):
            # Janky stuff to get the boundaries to look nice
            if (i == 0) or (i == 9) or (j == 0) or (j == 9):
                node = Node(mirror_matrix)
                node.metadata['color'] = 'gray'
                lat.network.add_node(node)
                if (i == 9):
                    lat.network.add_edge(node, nodes[i-1][j], in_order=2, out_order=0)
                    lat.network.add_edge(nodes[i-1][j], node, in_order=0, out_order=2)
                if (j == 9):
                    lat.network.add_edge(node, nodes[i][j-1], in_order=1, out_order=0)
                    lat.network.add_edge(nodes[i][j-1], node, in_order=0, out_order=1)
            else:
                if (i < 5):
                    node = Node(multiport_matrix1)
                    node.metadata['color'] = 'green'
                else:
                    node = Node(multiport_matrix2)
                    node.metadata['color'] = 'blue'
                lat.network.add_node(node)
                lat.network.add_edge(node, nodes[i-1][j], in_order=2, out_order=0)
                lat.network.add_edge(nodes[i-1][j], node, in_order=0, out_order=2)
                lat.network.add_edge(node, nodes[i][j-1], in_order=1, out_order=3)
                lat.network.add_edge(nodes[i][j-1], node, in_order=3, out_order=1)
            node.metadata['position'] = (i, j)
            nodes[i][j] = node

nx.set_edge_attributes(lat.network, 0, 'amplitude')

lat.network[nodes[7][4]][nodes[7][3]]['amplitude'] = 1j

draw(lat)
plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
