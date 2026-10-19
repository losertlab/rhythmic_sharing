import scipy as sp
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
from scipy.linalg import pinv
import networkx as nx
from sklearn.linear_model import Ridge
import warnings
from tqdm import tqdm

from lm_neuron_eq import initial_states, clip_voltage, single_unit


class SpikingNetwork:
	def __init__(self, **kwargs):
		self.average_degree_nodes = kwargs.get('average_degree_nodes', 10)
		self.num_nodes = kwargs.get('num_nodes', 100)
		self.input_weight = kwargs.get('input_weight', 120e-2)
		self.input_weight_assign_to = kwargs.get('input_weight_assign_to', None)
		self.leakage = kwargs.get('leakage', 0.0)
		self.spectral_radius = kwargs.get('spectral_radius', 0.6)
		self.bias_nodes = kwargs.get('bias_nodes', 0)
		self.link_strength_change_ratio = kwargs.get('link_strength_change_ratio', 0.6)
		self.regularization = kwargs.get('regularization', 1e-20)
		self.model_seed = kwargs.get('model_seed', 0)
		self.input_dims = kwargs.get('input_dims', 3)
		self.frozen = False
		self.lsqrs = False
		self.mean_phase_threshold = kwargs.get('mean_phase_threshold', np.pi)
		self.mean_phase_tolerance = kwargs.get('mean_phase_tolerance', 0.005)
		self.error_threshold = kwargs.get('error_threshold', 1e-3)
		self.error_tolerance = kwargs.get('error_tolerance', 1e-3)

		# Spiking config
		self.lock_threshold = kwargs.get('lock_threshold', 0.85)
		self.omega0 = kwargs.get('omega0', 0.01)
		self.I_ref = kwargs.get('I_ref', 39.96 + 15)
		self.link_coupling = kwargs.get('link_coupling', 1.0)
		self.epsilon1 = kwargs.get('epsilon1', -0.2)
		self.epsilon2 = kwargs.get('epsilon2', 0.6)
		self.subtract_dc = kwargs.get('subtract_dc', True)

		if not np.isscalar(self.input_weight) and not self.input_weight_assign_to:
			assert False, "Must pass in 'input_weight_assign_to' if 'input_weight' is a list"
		elif not np.isscalar(self.input_weight):
			assert np.sum(self.input_weight_assign_to) == self.input_dims, "Sum of 'input_weight_assign_to' must equal 'input_dims'"
			self.input_weight_assign_to = np.cumsum(self.input_weight_assign_to)

		self.average_degree_links = self.num_nodes // 2
		self.node_adj_matrix = self.gen_node_adj_matrix()
		self.incidence_T, self.incidence_norm = self.gen_incidence_T()
		self.input_weights = self.gen_input_weights()
		self.output_weights = np.zeros((self.input_dims, self.num_nodes))
		self.num_links = np.count_nonzero(self.node_adj_matrix.toarray())
		self.link_adj_matrix, self.link_adj_norm = self.gen_link_adj_matrix()

		self.h_max = 0.1
		self.T_ref = self.reference_period(self.I_ref, h=self.h_max)
		self.dt_link = self.T_ref * self.omega0 / (2 * np.pi)
		self.I_coup_mean = np.zeros(self.num_links)
		self.n_sub = int(np.ceil(self.dt_link/self.h_max))
		self.coupling_I_max = kwargs.get("coupling_I_max", 3)

		cycle_points = kwargs.get('cycle_points', 256)
		v_cyc, n_cyc = self.reference_cycle(self.I_ref, h=self.h_max)
		idx = np.linspace(0, len(v_cyc), cycle_points, endpoint=False).astype(int)
		self.v_cycle, self.n_cycle = v_cyc[idx], n_cyc[idx]
		self.v_scale, self.n_scale = np.ptp(self.v_cycle), np.ptp(self.n_cycle)

		self.mod_tau = kwargs.get('mod_tau', 30)

		self.reset_initial_states()
		self.reset_history()

	def change_spectral_rad_and_leakage(self, spectral_radius, leakage):
		self.leakage = leakage
		self.spectral_radius = spectral_radius
		self.node_adj_matrix = self.gen_node_adj_matrix()
		self.link_adj_matrix, self.link_adj_norm = self.gen_link_adj_matrix()

		self.reset_initial_states()
		self.reset_history()
	
	def gen_node_adj_matrix(self):
		unbounded_links = sparse.random(self.num_nodes, self.num_nodes, density=self.average_degree_nodes/self.num_nodes, random_state=self.model_seed)
		bounded_links = 2*unbounded_links - unbounded_links.ceil()
		rho = np.max(np.abs(np.linalg.eigvals(bounded_links.toarray())))
		return self.spectral_radius / rho * bounded_links

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

	def reset_initial_states(self, offset=1):
		rng = np.random.default_rng(self.model_seed + offset)
		k = rng.integers(0, len(self.v_cycle), self.num_links)
		self.node_states = np.zeros((self.num_nodes,))
		self.link_states = np.stack([self.v_cycle[k], self.n_cycle[k]], axis=1)
		self.reset_link_bookkeeping()

	def reset_history(self):
		self.node_states_history, self.link_states_history, self.link_phase_history, self.training_data_history, self.prediction_history = [], [], [], [], []
		self.prediction_history.append(self.output_weights @ self.node_states)

	def advance_nodes(self, input_state, save_history=True):
		# c = self.link_strength_change_ratio
		psi = np.angle(np.nanmean(np.exp(1j * self.link_phase)))
		rel = np.exp(1j * (self.link_phase - psi))
		if self.rel_lp is None:
			self.rel_lp = rel
		elif not self.frozen:
			self.rel_lp = self.rel_lp + (rel - self.rel_lp) / self.mod_tau   # held fixed once frozen
		
		if self.lock_threshold:
			lock = np.clip((np.abs(self.rel_lp) - self.lock_threshold) / (1 - self.lock_threshold), 0, 1)
			modulation = 1 - (self.link_strength_change_ratio / 2) * (1 - lock)
		else:
			modulation = 1 - (self.link_strength_change_ratio / 2) * (1 - np.abs(self.rel_lp))
		
		copied_node_adj_matrix = self.node_adj_matrix.copy()
		copied_node_adj_matrix.data *= modulation

		self.node_states = self.leakage*self.node_states + (1-self.leakage)*np.tanh(copied_node_adj_matrix.dot(self.node_states) + self.input_weights @ input_state + self.bias_nodes)
		if save_history:
			self.node_states_history.append(np.copy(self.node_states))
			self.training_data_history.append(np.copy(input_state))

	def advance_links(self, save_history=True, freezing=False):
		if not freezing:
			# Node activity sets the coupling gain; every link receives the same current, so rates do not encode input
			node_drive = (self.incidence_T @ ((self.node_states + 1)/2)) * (1/self.incidence_norm)
			gain = self.link_coupling * (self.epsilon1 + self.epsilon2 * node_drive)
			self.integrate_and_update(self.I_ref, gain)
			self.link_phase = self.measured_phase()
		elif not self.frozen:
			# Uncoupled free-running at I_ref: exactly a rigid rotation at omega0 per step
			gap_before = self.mean_phase_gap()
			self.integrate_and_update(self.I_ref, None)
			self.link_phase = self.measured_phase()
			gap_after = self.mean_phase_gap()
			if gap_before < 0 <= gap_after and gap_after - gap_before < np.pi:
				self.frozen = True
		# frozen: hold (v, n) and phase

		if save_history:
			self.link_states_history.append(np.copy(self.link_states))
			self.link_phase_history.append(np.copy(self.link_phase))
		
	def advance(self, input_state, save_history=True, freezing=False):
		self.advance_nodes(input_state, save_history=save_history)
		self.advance_links(save_history=save_history, freezing=freezing)

	def train(self, training_data, warmup_time=0):
		for t in range(warmup_time):
			self.advance(training_data[:, t], save_history=False)
		
		t_range = range(warmup_time, training_data.shape[1])
		if len(t_range) > 10_000:
			for t in tqdm(t_range):
				self.advance(training_data[:, t])
		else:
			for t in t_range:
				self.advance(training_data[:, t])
		self.compute_weights()

	def create_model(self):
		if not self.lsqrs:
			ridge_model = Ridge(alpha=self.regularization, fit_intercept=False, solver="auto")
		else:
			ridge_model = Ridge(alpha=self.regularization, fit_intercept=False, solver="lsqr")
		return ridge_model

	def compute_weights(self):
		reg_node_states = np.asarray(self.node_states_history)[:-1]
		training_data = np.asarray(self.training_data_history)[1:]
		ridge_model = self.create_model()
		
		with warnings.catch_warnings(record=True) as warning_list:
			warnings.simplefilter("always")
			ridge_model.fit(reg_node_states, training_data)
			if warning_list:
				print("Switching to lsqr")
				self.lsqrs = not self.lsqrs
				ridge_model = self.create_model()
				ridge_model.fit(reg_node_states, training_data)
		self.output_weights = ridge_model.coef_

	def compute_predict_error(self, state):
		self.prediction_error = np.sum(self.prediction_history[-1]-state, axis=0)**2

	def get_history(self):
		return np.asarray(self.node_states_history).T, np.asarray(self.link_states_history).T, np.asarray(self.link_phase_history).T, np.asarray(self.training_data_history).T, np.asarray(self.prediction_history).T

	def get_global_parameters(self):
		z = np.nanmean(np.exp(1j * np.asarray(self.link_phase_history)), axis=1)
		return np.abs(z), np.angle(z)

	def get_input_parameters(self):
		inp_mag = []
		inp_ph = []
		for inp in range(self.input_weights.shape[1]):
			links_from_input = []
			inp_nodes = self.input_weights[:, inp].nonzero()[0]

			for i in inp_nodes:
				_, connected_nodes = self.node_adj_matrix[i, :].nonzero()
				for j in connected_nodes:
					flat_idx = i * self.num_nodes + j
					nonzero_adj_idxs = np.where(np.ndarray.flatten(self.node_adj_matrix.toarray())!=0)[0]
					links_from_input.append(np.where(flat_idx == nonzero_adj_idxs)[0][0])

			links_from_input = np.asarray(links_from_input).astype(int)
			z = np.nanmean(np.exp(1j * np.asarray(self.link_phase_history)[:, links_from_input]), axis=1)
			
			inp_mag.append(np.abs(z))
			inp_ph.append(np.angle(z))

		return np.asarray(inp_mag), np.asarray(inp_ph)

	def get_output(self):
		output = self.output_weights @ self.node_states
		self.prediction_history.append(output)
		return output

	def predict(self, test_data, warmup_time=0, freezing_time=None, r_burn=None, r_window=None):
		prediction_time = test_data.shape[1] - warmup_time
		r_burn = warmup_time // 4 if r_burn is None else r_burn
		r_window = warmup_time // 2 if r_window is None else r_window

		self.reset_initial_states()
		self.reset_history()

		for t in range(warmup_time):
			self.compute_predict_error(test_data[:, t])
			if freezing_time:
				self.advance(test_data[:, t], freezing=t>=freezing_time)
			else:
				self.advance(test_data[:, t], freezing=False)
			self.get_output()

		self.frozen = True # hold phases and rel_lp immediately; rotation is unnecessary
		for t in range(warmup_time, warmup_time + prediction_time):
			self.advance(self.prediction_history[-1], freezing=True)
			self.get_output()

		return np.asarray(self.prediction_history[warmup_time - 1:-1]).T

	# Neuron model helper functions
	def rk4_substep(self, v, n, I_ext, h=0.1, tau_syn=2):
		rhs = lambda v, n: single_unit(v, n, I_ext(v) if callable(I_ext) else I_ext)

		k1 = rhs(v, n)
		k2 = rhs(v + 0.5*h*k1[0], n + 0.5*h*k1[1])
		k3 = rhs(v + 0.5*h*k2[0], n + 0.5*h*k2[1])
		k4 = rhs(v + h*k3[0], n + h*k3[1])

		v = v + (h/6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
		n = n + (h/6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
			
		return v, n

	def reference_period(self, I_ref, t_trans=1000, t_meas=1500, h=0.1):
		v, n = np.asarray([-60.0]), np.asarray([0.0])
		t, spikes = 0, []

		while t < t_trans + t_meas:
			v_new, n = self.rk4_substep(v, n, I_ref, h=h)
			if v[0] < 0.0 <= v_new[0] and t > t_trans:
				spikes.append((t + h * (-v[0]))/(v_new[0] - v[0]))
			v, t = v_new, t + h

		if len(spikes) < 2:
			raise ValueError("I_ref %.3f does not fire, raise it" % I_ref)
		return np.mean(np.diff(spikes))

	def reference_cycle(self, I_ref, t_trans=1000, h=0.1):
		v, n, t = np.asarray([-60.0]), np.asarray([0.0]), 0.0
		while t < t_trans:
			v, n = self.rk4_substep(v, n, I_ref, h=h)
			t += h
 
		crossings, v_samples, n_samples = 0, [], []
		while crossings < 2:
			v_new, n = self.rk4_substep(v, n, I_ref, h=h)
			if v[0] < 0.0 <= v_new[0]:
				crossings += 1
			if crossings == 1:
				v_samples.append(v_new[0])
				n_samples.append(n[0])
			v = v_new


		return np.asarray(v_samples), np.asarray(n_samples)

	def integrate_and_update(self, I_ext, gain):
		# gain: per-receiving-link coupling conductance, or None for uncoupled
		h = self.dt_link / self.n_sub
		v, n = self.link_states[:, 0], self.link_states[:, 1]
		if gain is None:
			current = I_ext
			coup = None
		else:
			I_max = self.coupling_I_max
			coup = lambda v: np.clip(gain * (self.link_adj_matrix.dot(v) / self.link_adj_norm - v), -I_max, I_max)
			offset = np.copy(self.I_coup_mean) if self.subtract_dc else 0.0
			current = lambda v: I_ext + coup(v) - offset

		for _ in range(self.n_sub):
			v_new, n = self.rk4_substep(v, n, current, h=h)
			fired = (v < 0) & (v_new >= 0)

			if np.any(fired):
				t_spk = self.link_time + h * (0 - v[fired]) / (v_new[fired] - v[fired])
				self.last_isi[fired] = t_spk - self.last_spike[fired]
				self.last_spike[fired] = t_spk

			v = v_new

			if coup is not None:
				self.I_coup_mean += (h / self.T_ref) * (coup(v) - self.I_coup_mean)
			self.link_time += h

		self.link_states = np.stack([v, n], axis=1)

	def reset_link_bookkeeping(self, h_max=0.1):
		self.link_time = 0.0
		self.last_spike = np.ones(self.num_links) * -np.inf
		self.last_isi = np.ones(self.num_links) * np.nan
		self.frozen = False
		self.link_phase = self.continuous_phase()
		self.I_coup_mean = np.zeros(self.num_links)
		self.rel_lp = None
		
	def spike_phase(self):
		phase = 2 * np.pi * np.mod((self.link_time - self.last_spike)/self.last_isi, 1.0)
		phase[~np.isfinite(self.last_isi)] = np.nan
		return phase

	def silent_links(self, silence_factor=2.0):
		return ~((self.link_time - self.last_spike) < silence_factor * self.last_isi)

	def mean_phase_gap(self):
		z = np.nanmean(np.exp(1j * self.link_phase))
		return np.angle(z * np.exp(-1j * self.mean_phase_threshold))
 
	def phase_to_cycle(self, cycle):
		grid = 2 * np.pi * np.arange(len(cycle)) / len(cycle)
		return np.interp(self.link_phase, grid, cycle, period=2 * np.pi)

	def continuous_phase(self):
		# All links run on the I_ref cycle, so nearest point on it is an accurate phase (up to coupling perturbations)
		if not np.all(np.isfinite(self.link_states)):
			raise FloatingPointError("link states diverged at t=%.1f" % self.link_time)

		v, n = self.link_states[:, 0], self.link_states[:, 1]
		d = ((v[:, None] - self.v_cycle[None, :]) / self.v_scale)**2 \
		  + ((n[:, None] - self.n_cycle[None, :]) / self.n_scale)**2
		return 2 * np.pi * np.argmin(d, axis=1) / len(self.v_cycle)

	def measured_phase(self):
		# Linear-in-time between spikes: uniform phase velocity, so no static offset in z
		phase = self.spike_phase()					 # NaN until a link has a finite ISI
		undefined = np.isnan(phase)
		if np.any(undefined):
			phase[undefined] = self.continuous_phase()[undefined]   # only during the first period or so
		return phase

	# def initial_frozen_phase(self):
	#	 phase = self.spike_phase()

	#	 undefined = ~np.isfinite(self.last_isi)
	#	 print("Getting initial frozen phase: %.2f%% undefined" % ((np.sum(undefined)/len(undefined)) * 100))
	#	 if np.any(undefined):
	#		 v_scale = np.ptp(self.v_cycle)
	#		 n_scale = np.ptp(self.n_cycle)
	#		 dv = (self.link_states[undefined, 0][:, None] - self.v_cycle[None, :]) / v_scale
	#		 dn = (self.link_states[undefined, 1][:, None] - self.n_cycle[None, :]) / n_scale
	#		 idx = np.argmin(dv**2 + dn**2, axis=1)
	#		 phase[undefined] = 2 * np.pi * idx / len(self.v_cycle)

	#	 return phase

