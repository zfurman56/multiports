import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import product


class Node:
    def __init__(self, scatter_matrix):
        self._scatter_matrix = scatter_matrix
        self.metadata = {}

    def _scatter(self, inputs):
        return self._scatter_matrix.dot(inputs)

    def _to_inputs(self, input_edges):
        return [e['amplitude'] for e in sorted(input_edges, key=lambda x: x['in_order'])]

    def _to_outputs(self, outputs, output_edges):
        for i, edge in enumerate(sorted(output_edges, key=lambda x: x['out_order'])):
            edge['amplitude'] = outputs[i]

    def update(self, input_edges, output_edges):
        inputs = self._to_inputs(input_edges)
        self._to_outputs(self._scatter(inputs), output_edges)

class Detector(Node):
    def __init__(self, scatter_matrix):
        super().__init__(scatter_matrix)
        self.amplitudes = np.zeros(np.shape(scatter_matrix)[1])

    def _scatter(self, inputs):
        self.amplitudes += inputs
        return super()._scatter(inputs)
        
class ScatteringNetwork:
    def __init__(self):
        self.network = nx.DiGraph()

    def step(self):
        # Create temp copy network to modify
        network_mut = deepcopy(self.network)
        nx.set_edge_attributes(network_mut, 0, 'amplitude')

        for rnode, wnode in zip(self.network.nodes, network_mut.nodes):
            wnode.update([self.network[a][b] for a,b in self.network.in_edges(rnode)], [network_mut[a][b] for a,b in network_mut.out_edges(wnode)])

        self.network = network_mut

    # Convert lattice model into quantum Markov model
    def output_markov(self):
        markov_network = nx.DiGraph()
        # Turn edges into nodes
        for edge in self.network.edges:
            markov_network.add_node(edge)
        # Create an edge in the markov graph for every pair of edges in the
        # original graph whose source matches the other's desination
        for edge in self.network.edges:
            for out_edge in self.network.out_edges(edge):
                markov_network.add_edge(edge, out_edge)

        return markov_network

