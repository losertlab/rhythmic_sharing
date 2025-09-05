import scipy as sp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
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
        self.input_weight_assign_to = kwargs.get('input_weight_assign_to', None)
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
        self.mean_phase_tolerance = kwargs.get('mean_phase_threshold', 1e-3)
        self.error_threshold = kwargs.get('error_threshold', 1e-3)
        self.error_tolerance = kwargs.get('error_tolerance', 1e-3)

        if not np.isscalar(self.input_weight) and not self.input_weight_assign_to:
            assert False, "Must pass in 'input_weight_assign_to' if 'input_weight' is a list"
        elif not np.isscalar(self.input_weight):
            assert np.sum(self.input_weight_assign_to) == self.input_dims, "Sum of 'input_weight_assign_to' must equal 'input_dims'"
            self.input_weight_assign_to = np.cumsum(self.input_weight_assign_to)

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
        self.node_to_link_coupling_history, self.local_mean_phase_history = [], []

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
        qq = self.num_nodes // self.input_dims
        input_weights = np.zeros((self.num_nodes, self.input_dims))
        for i in range(self.input_dims):
            np.random.seed(i)
            ip = 2*np.random.rand(qq) - 1
            if np.isscalar(self.input_weight):
                input_weights[i*qq:(i+1)*qq, i] = self.input_weight*ip
            else:
                
                input_weights[i*qq:(i+1)*qq, i] = self.input_weight[np.sort(np.where(self.input_weight_assign_to > i)[0])[0]]*ip
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

    def gen_initial_states(self):
        node_states = np.zeros((self.num_nodes))
        link_states = np.zeros((self.num_links))
        for i in range(self.num_links):
            np.random.seed(i)
            link_states[i] = np.random.rand(1)[0]*2*np.pi
        return node_states, link_states

    def advance_nodes(self, input_state, save_history=True):
        link_phases = np.zeros((self.num_nodes**2))
        link_phases[self.nonzero_adj_idxs] = self.link_states
        link_phases = np.reshape(link_phases, (self.num_nodes, self.num_nodes))
        modulated_node_adj_matrix = self.node_adj_matrix.toarray()*(1-(self.link_strength_change_ratio/2)*(1+np.sin(link_phases)))
        self.node_states = self.leakage*self.node_states + (1-self.leakage)*np.tanh(modulated_node_adj_matrix.dot(self.node_states) + self.input_weights @ input_state + self.bias_nodes)
        if save_history:
            self.node_states_history.append(self.node_states)
            self.training_data_history.append(input_state)
            self.node_to_link_coupling_history.append((self.incidence_T @ ((self.node_states+1)/2)) / self.incidence_norm)

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
        elif self.frozen or (np.abs(global_mean_phase-self.mean_phase_threshold) < self.mean_phase_tolerance and self.prediction_error < self.error_tolerance):
            self.frozen = True
        else:
            self.link_states = self.link_states + self.dt*self.omega0
        
        if save_history:
            self.local_mean_phase_history.append(local_mean_phase)
            self.link_states_history.append(self.link_states)
                
    def train(self, training_data, warmup_time=0, reset=True, save_link_history=True):
        # Reset training data
        if reset:
            self.node_states, self.link_states = self.gen_initial_states()
            self.node_states_history, self.training_data_history = [], []
       
        # Run training
        for t in range(warmup_time):
            self.advance_nodes(training_data[:, t], save_history=False)
            self.advance_links(save_history=False)
        for t in range(warmup_time, training_data.shape[1]):
            self.advance_nodes(training_data[:, t])
            self.advance_links(save_history=save_link_history)
        
        # Fit output weights
        ridge_model = Ridge(alpha=self.regularization, fit_intercept=False)
        ridge_model.fit(np.asarray(self.node_states_history), np.asarray(self.training_data_history))
        self.output_weights = ridge_model.coef_
        
        return self.output_weights

    def get_history(self):
        return np.asarray(self.node_states_history).T, np.asarray(self.link_states_history).T, np.asarray(self.training_data_history).T, np.asarray(self.prediction_history).T

    def get_global_parameters(self):
        R_x = np.average(np.cos(np.asarray(self.link_states_history).T), axis=0)
        R_y = np.average(np.sin(np.asarray(self.link_states_history).T), axis=0)
        return (R_x**2 + R_y**2)**(1/2), np.arctan2(R_y, R_x)

    def get_input_parameters(self):
        inp_rs = []
        inp_ph = []
        for inp in range(self.input_weights.shape[1]):
            links_from_input = []
            inp_nodes = self.input_weights[:, inp].nonzero()[0]
            for i in inp_nodes:
                _, connected_nodes = self.node_adj_matrix[i, :].nonzero()
                for j in connected_nodes:
                    flat_idx = i * self.num_nodes + j
                    links_from_input.append(np.where(flat_idx == self.nonzero_adj_idxs)[0][0])
            links_from_input = np.asarray(links_from_input).astype(int)
            R_x = np.average(np.cos(np.asarray(self.link_states_history).T[links_from_input]), axis=0)
            R_y = np.average(np.sin(np.asarray(self.link_states_history).T[links_from_input]), axis=0)
            inp_rs.append((R_x**2 + R_y**2)**(1/2))
            inp_ph.append(np.arctan2(R_y, R_x))
        return np.asarray(inp_rs), np.asarray(inp_ph)

    def predict(self, training_data, warmup_time=0, freezing_time=float('inf'), reset=True, channels_using_sample=None):
        # Reset prediction data
        if reset:
            self.prediction_history = []
            self.node_states, self.link_states = self.gen_initial_states()
        
        # Run prediction loop (warmup then regular)
        self.prediction_history.append(self.output_weights @ self.node_states)
        for t in range(warmup_time):
            self.predict_single_sample(np.copy(training_data[:, t]), freeze=t>freezing_time, channels_using_sample=range(training_data.shape[0]))

        for t in range(warmup_time, training_data.shape[-1]):
            self.predict_single_sample(np.copy(training_data[:, t]), freeze=t>freezing_time, channels_using_sample=channels_using_sample)
        
        return np.asarray(self.prediction_history).T

    def predict_single_sample(self, sample, freeze=False, save_link_history=True, channels_using_sample=None):
        self.prediction_error = np.sum(self.prediction_history[-1]-sample, axis=0)**2

        if channels_using_sample:
            for i in range(sample.shape[0]):
                if i not in channels_using_sample:
                    sample[i] = self.prediction_history[-1][i]
        else:
            sample = self.prediction_history[-1]
        
        self.advance_nodes(sample)
        self.advance_links(freezing=freeze, save_history=save_link_history)
        self.prediction_history.append(self.output_weights @ self.node_states)
        
        return self.prediction_history[-1], self.prediction_error

    # def measure_coupling(self, training_data, percentile=95, warmup_time=np.inf):
    #     # Feeds in the data's warmup samples to estimate max((Q^T n*) / incidence_norm).
    #     # (max((Q^T n*) / incidence_norm) * epsilon2) + epsilon1 > omega0 for convergence to 1 (section 4.2.1)
        
    #     self.node_states, self.link_states = self.gen_initial_states()

    #     for t in range(min(warmup_time, training_data.shape[1])):
    #         self.advance_nodes(training_data[:, t], save_history=False)
    #         self.advance_links(save_history=False)

    #     n_star = (self.node_states + 1) / 2
    #     QTn = (self.incidence_T @ n_star) / self.incidence_norm
    #     return np.percentile(QTn, percentile)

    # def estimate_best_epsilons(self, coupling_one, coupling_two, min_e1=-1.0, max_e1=1.0, max_e2=3.0, tol=1e-6):
    #     # Performs a binary search to find the best e1 and e2 values
        
    #     e2 = lambda y: (2 * y)/(coupling_one - coupling_two)
    #     e1 = lambda y: self.omega0 - y - (e2(y) * coupling_two)
    #     good_solution = lambda y: e2(y) < max_e2 and min_e1 < e1(y) < max_e1

    #     low, high = 0, (max_e2 * (coupling_one - coupling_two))/2
    #     while high - low > tol:
    #         y = 0.5 * (high + low)
    #         if good_solution(y):
    #             low = y
    #         else:
    #             high = y

    #     self.epsilon1 = e1(y)
    #     self.epsilon2 = e2(y)
    #     return self.epsilon1, self.epsilon2

    # def online_predict_and_train(self, training_data, train_warmup_time=0, train_every=20, pretrain_with=100, prev_trainsteps_to_remember=5, reset=True):
    #         # Checks and setup
    #         assert train_warmup_time < pretrain_with and train_warmup_time < train_every * prev_trainsteps_to_remember, "Warmup time for training exceeds training data passed"
    #         assert pretrain_with < training_data.shape[1], "Pretraining is longer than actual data"
    #         assert pretrain_with >= train_every * prev_trainsteps_to_remember, "Not enough pretrain data for initial retrain step"
    #         w_out_hist, errs = [], []
    #         if reset:
    #            self.prediction_history = [] 
            
    #         # Perform pretraining
    #         w_out_hist.append(self.train(training_data[:, :pretrain_with], warmup_time=train_warmup_time))
            
    #         # Prediction loop
    #         self.node_states, self.link_states = self.gen_initial_states()
    #         self.prediction_history.append(self.output_weights @ self.node_states)
    #         for t in range(pretrain_with, training_data.shape[1]):
    #             # Calculate prediction
    #             errs.append(self.predict_single_sample(training_data[:, t])[1])
                
    #             # Periodically retrain
    #             if t % train_every == 0 and t > pretrain_with:
    #                 w_out_hist.append(self.train(training_data[:, t-(prev_trainsteps_to_remember*train_every):t], warmup_time=train_warmup_time, save_link_history=False))
            
    #         return np.asarray(self.prediction_history).T, np.asarray(w_out_hist), self.get_global_parameters()[0], np.asarray(errs)
