import scipy as sp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import networkx as nx

class RhythmicNetwork:
    def __init__(self, **kwargs):
        self.tau = kwargs.get('tau', 1)
        self.average_degree = kwargs.get('average_degree', 10)
        self.num_nodes = kwargs.get('num_nodes', 100)
        self.link_dist = kwargs.get('link_dist', 'discrete')
        self.omega0 = kwargs.get('omega0', 0.01)
        self.input_weight = kwargs.get('input_weight', 120e-2)
        self.epsilon1 = kwargs.get('epsilon1', 0.2)
        self.epsilon2 = kwargs.get('epsilon2', 0.6)
        self.leakage = kwargs.get('leakage', 0.0)
        self.spectral_radius = kwargs.get('spectral_radius', 0.6)
        self.bias = kwargs.get('bias', 0)
        self.rhythmic_link_ratio = kwargs.get('rhythmic_link_ratio', 1)
        self.link_strength_change_ratio = kwargs.get('link_strength_change_ratio', 0.6)
        self.regularization = kwargs.get('regularization', 1e-20)
        self.bias_phase = kwargs.get('bias_phase', 0)
        self.model_seed = kwargs.get('model_seed', 0)
        self.input_dims = kwargs.get('input_dims', None)

        self.average_degree_links = self.num_nodes // 2
        self.link_network = self.gen_link_network()
        self.nonzero_link_idxs = np.where(np.ndarray.flatten(self.link_network.toarray())!=0)[0]
        self.incidence_matrix_T, self.incidence_normalization = self.gen_incidence_matrix_T()

    def gen_link_network(self):
        unbounded_links = sparse.random(self.num_nodes, self.num_nodes, density=self.average_degree/self.num_nodes, random_state=self.model_seed)
        bounded_links = 2*unbounded_links - unbounded_links.ceil()
        link_eigenvalues = linalg.eigs(bounded_links, k=1, return_eigenvectors=False)
        return self.spectral_radius/np.abs(link_eigenvalues[0])*bounded_links

    def gen_incidence_matrix_T(self):
        link_graph = nx.from_numpy_array(self.link_network, parallel_edges=True, create_using=nx.DiGraph())
        nodelist = list(link_graph)
        if link_graph.is_multigraph():
            edgelist = list(link_graph.edges(keys=True))
        else:
            edgelist = list(link_graph.edges())
        A = sp.sparse.lil_array((len(nodelist), len(edgelist)))
        node_index = {node: i for i, node in enumerate(nodelist)}
        for ei, e in enumerate(edgelist):
            (u, v) = e[:2]
            if u == v: # self loops give zero column ---> CHANGED PERSONALLY TO EQUAL 1 with the 2 lines of code below (otherwise, just 'continue')
                A[u, ei] = 1 #I set it to 1; can change to 2, which is what some conventions use. 
                A[v, ei] = 1       
                continue  
            try:
                ui = node_index[u]
                vi = node_index[v]
            except KeyError as err:
                raise nx.NetworkXError(
                    f"node {u} or {v} in edgelist but not in nodelist"
                ) from err
            wt = 1
            A[ui, ei] = wt
            A[vi, ei] = wt
        incidence_matrix = A.asformat("csc")
        incidence_matrix_T = incidence_matrix.toarray().T
        incidence_normalization = np.zeros((incidence_matrix_T.shape[0]))
        for i in range(incidence_matrix_T.shape[0]):
            incidence_normalization[i] = np.count_nonzero(incidence_matrix_T[i])
        return incidence_matrix_T, incidence_normalization


