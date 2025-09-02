import scipy as sp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import pinv
import networkx as nx
from sklearn.linear_model import Ridge

class RhythmicNetwork:
    def __init__(self, **kwargs):
        self.dt = kwargs.get('dt', 1)
        self.average_degree_nodes = kwargs.get('average_degree_nodes', 10)
        self.num_nodes = kwargs.get('num_nodes', 100)
        self.link_dist = kwargs.get('link_dist', 'discrete')
        self.omega0 = kwargs.get('omega0', 0.01)
        self.omega0_mean = kwargs.get('omega0_mean', self.omega0)
        self.omega0_spread = kwargs.get('omega0_spread', self.omega0/3)
        self.input_weight = kwargs.get('input_weight', 120e-2)
        self.epsilon1 = kwargs.get('epsilon1', -0.2)
        self.epsilon2 = kwargs.get('epsilon2', 0.6)
        self.leakage = kwargs.get('leakage', 0.0)
        self.spectral_radius = kwargs.get('spectral_radius', 0.6)
        self.bias_nodes = kwargs.get('bias_nodes', 0)
        self.rhythmic_link_ratio = kwargs.get('rhythmic_link_ratio', 1)
        self.link_strength_change_ratio = kwargs.get('link_strength_change_ratio', 0.6)
        self.regularization = kwargs.get('regularization', 1e-20)
        self.bias_phase = kwargs.get('bias_phase', 0)
        self.model_seed = kwargs.get('model_seed', 0)
        self.input_dims = kwargs.get('input_dims', None)
        self.frozen = False
        self.mean_phase_threshold = kwargs.get('mean_phase_threshold', np.pi)
        self.mean_phase_tolerance = kwargs.get('mean_phase_tolerance', 1e-3)
        self.error_threshold = kwargs.get('error_threshold', 1e-3)
        self.error_tolerance = kwargs.get('error_tolerance', 1e-3)

        self.average_degree_links = self.num_nodes // 2
        self.node_adj_matrix = self.gen_node_adj_matrix()
        self.nonzero_adj_idxs = np.where(np.ndarray.flatten(self.node_adj_matrix.toarray())!=0)[0]
        self.incidence_T, self.incidence_norm = self.gen_incidence_T()
        self.input_weights = self.gen_input_weights()
        self.output_weights = None
        self.num_links = np.count_nonzero(self.node_adj_matrix.toarray())
        self.link_adj_matrix, self.link_adj_norm = self.gen_link_adj_matrix()
        self.natural_frequencies = self.gen_natural_frequencies()
        self.node_states, self.link_states = self.gen_initial_states()
        self.node_states_history, self.link_states_history, self.training_data_history, self.prediction_history = [], [], [], []

    def gen_node_adj_matrix(self):
        unbounded_links = sparse.random(self.num_nodes, self.num_nodes, density=self.average_degree_nodes/self.num_nodes, random_state=self.model_seed)
        bounded_links = 2*unbounded_links - unbounded_links.ceil()
        link_eigenvalues = linalg.eigs(bounded_links, k=1, return_eigenvectors=False)
        return self.spectral_radius/np.abs(link_eigenvalues[0])*bounded_links

    def gen_incidence_T(self):
        link_graph = nx.from_numpy_array(self.node_adj_matrix, parallel_edges=True, create_using=nx.DiGraph())
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

    def gen_input_weights(self):
        qq = int(np.floor(self.num_nodes/self.input_dims))
        input_weights = np.zeros((self.num_nodes, self.input_dims))
        for i in range(self.input_dims):
            np.random.seed(i)
            ip = 2*np.random.rand(qq) - 1
            input_weights[i*qq:(i+1)*qq, i] = self.input_weight*ip
        return input_weights

    def gen_link_adj_matrix(self):
        link_adj_matrix = sparse.csr_matrix.ceil(sparse.random(self.num_links, self.num_links, density=self.average_degree_links/self.num_links, random_state=self.model_seed+3))
        link_adj_matrix_norm = np.sum(link_adj_matrix.toarray(), axis=1)
        if np.all(link_adj_matrix.toarray()[np.where(link_adj_matrix_norm==0)]==0)==1:
            link_adj_matrix_norm[np.where(link_adj_matrix_norm==0)]=1000
        return link_adj_matrix, link_adj_matrix_norm

    def gen_natural_frequencies(self):
        if self.link_dist == 'discrete':
            natural_frequencies = sparse.random(1, self.num_links, density=self.rhythmic_link_ratio, random_state=self.model_seed+5)
            natural_frequencies = np.matrix.flatten(np.ceil(natural_frequencies.toarray()))*self.omega0
        elif self.link_dist == 'normal':
            natural_frequencies = sparse.random(1, self.num_links, density=self.rhythmic_link_ratio, random_state=self.model_seed+5)
            natural_frequencies = np.matrix.flatten(np.ceil(natural_frequencies.toarray()))
            natural_frequencies[np.where(natural_frequencies!=0)[0]] = np.random.normal(loc=self.omega0_mean, scale=self.omega0_spread, size=np.where(natural_frequencies!=0)[0].shape[0])
        return natural_frequencies

    def gen_initial_states(self, seed_offset=0):
        node_states = np.zeros((self.num_nodes))
        link_states = np.zeros((self.num_links))
        for i in range(self.num_links):
            np.random.seed(i+seed_offset)
            link_states[i] = np.random.rand(1)[0]*2*np.pi
        return node_states, link_states

    def advance_nodes(self, input_state, save_history=True):
        link_phases = np.zeros((self.num_nodes**2))
        link_phases[self.nonzero_adj_idxs] = self.link_states
        link_phases = np.reshape(link_phases, (self.num_nodes, self.num_nodes))
        modulated_node_adj_matrix = self.node_adj_matrix.toarray()*(1-(self.link_strength_change_ratio/2)*(1+np.sin(link_phases)))
        self.node_states = self.leakage*self.node_states + (1-self.leakage)*np.tanh(modulated_node_adj_matrix.dot(self.node_states) + self.input_weights @ input_state + self.bias_nodes)
        if save_history:
            self.node_states_history.append(np.copy(self.node_states))
            self.training_data_history.append(np.copy(input_state))

    def advance_links(self, save_history=True, freezing=False):
        r_x_local = self.link_adj_matrix.dot(np.cos(self.link_states)) * (1/self.link_adj_norm)
        r_x_global = np.average(np.cos(self.link_states), axis=0)
        r_y_local = self.link_adj_matrix.dot(np.sin(self.link_states)) * (1/self.link_adj_norm)
        r_y_global = np.average(np.sin(self.link_states), axis=0)
        global_mean_phase = np.arctan2(r_y_global, r_x_global)
        local_mean_phase = np.arctan2(r_y_local, r_x_local)

        if not freezing:
            forcing = (self.epsilon1 + self.epsilon2*((self.incidence_T @ (self.node_states+1)/2)) * (1/self.incidence_norm)) * np.sin(local_mean_phase-self.link_states+self.bias_phase)
            self.link_states = self.link_states + self.dt*(self.natural_frequencies + forcing)
        else:
            self.frozen = self.frozen or (np.abs(global_mean_phase-self.mean_phase_threshold) < self.mean_phase_tolerance and self.prediction_error < self.error_tolerance)
            if not self.frozen:
                self.link_states = self.link_states + self.dt*self.omega0
        
        if save_history:
            self.link_states_history.append(np.copy(self.link_states))

    def train(self, training_data, warmup_time=0, type='auto'):
        if type != 'manual':
            for t in range(warmup_time):
                self.advance_nodes(training_data[:, t], save_history=False)
                self.advance_links(save_history=False)
            self.node_states_history.append(np.copy(self.node_states))
            for t in range(warmup_time, training_data.shape[1]-1):
                self.advance_nodes(training_data[:, t])
                self.advance_links()
            self.training_data_history.append(np.copy(training_data[:, training_data.shape[1]-1]))
            ridge_model = Ridge(alpha=self.regularization)
            ridge_model.fit(np.asarray(self.node_states_history), np.asarray(self.training_data_history))
            self.output_weights = ridge_model.coef_
        else:
            for t in range(warmup_time):
                self.advance_nodes(training_data[:, t], save_history=False)
                self.advance_links(save_history=False)
            self.node_states_history.append(np.copy(self.node_states))
            for t in range(warmup_time, training_data.shape[1]-1):
                self.advance_nodes(training_data[:, t])
                self.advance_links()
            self.training_data_history.append(np.copy(training_data[:, training_data.shape[1]-1]))

            identity1 = self.regularization*sparse.identity(self.num_nodes)
            training_data = np.asarray(self.training_data_history).T
            node_history = np.asarray(self.node_states_history).T
            identity2 = np.identity(training_data.shape[1])
            wout=np.zeros((training_data.shape[0],self.num_nodes))

            thingy1= np.matmul(np.matmul(training_data,identity2),np.transpose(node_history))
            # Y_target * X^T

            thingy2= pinv(np.matmul(np.matmul(node_history,identity2),np.transpose(node_history))+ identity1)
            wout= np.matmul(thingy1,thingy2) 
            self.output_weights = wout

    def get_history(self):
        return np.asarray(self.node_states_history).T, np.asarray(self.link_states_history).T, np.asarray(self.training_data_history).T, np.asarray(self.prediction_history).T

    def get_global_parameters(self):
        R_x = np.average(np.cos(np.asarray(self.link_states_history).T), axis=0)
        R_y = np.average(np.sin(np.asarray(self.link_states_history).T), axis=0)
        global_synchrony, global_mean_phase = (R_x**2 + R_y**2)**(1/2), np.arctan(R_y, R_x)
        return global_synchrony, global_mean_phase

    def predict(self, test_data, warmup_time=0, freezing_time=float('inf'), prediction_time=0):
        self.node_states, self.link_states = self.gen_initial_states(seed_offset=2)
        self.prediction_history.append(np.copy(self.output_weights @ self.node_states))
        self.node_states_history, self.link_states_history = [], []
        self.node_states_history.append(np.copy(self.node_states))
        self.link_states_history.append(np.copy(self.link_states))
        for t in range(warmup_time):
            self.prediction_error = np.sum(self.prediction_history[-1]-test_data[:, t], axis=0)**2
            self.advance_nodes(test_data[:, t])
            if t < freezing_time:
                self.advance_links()
            else:
                self.advance_links(freezing=True)
            self.prediction_history.append(np.copy(self.output_weights @ self.node_states))
        for t in range(warmup_time, warmup_time+prediction_time):
            self.advance_nodes(self.prediction_history[-1])
            self.advance_links(freezing=True)
            self.prediction_history.append(np.copy(self.output_weights @ self.node_states))

            

