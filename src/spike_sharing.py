import numpy as np
import scipy.sparse as sparse
from sklearn.linear_model import Ridge
import warnings
from tqdm import tqdm

from lm_neuron_eq import single_unit

I_C = 39.96
I_FLOOR = I_C + 1.0 
H_MAX = 0.1
CYCLE_POINTS = 256

class SpikingNetwork:
	def __init__(self, **kwargs):
		# Reservoir (ESN)
		self.num_nodes = kwargs.pop('num_nodes', 100)
		self.average_degree_nodes = kwargs.pop('average_degree_nodes', 10)
		self.spectral_radius = kwargs.pop('spectral_radius', 0.6)
		self.input_dims = kwargs.pop('input_dims', 3)
		self.input_weight = kwargs.pop('input_weight', 1.2)
		self.input_weight_assign_to = kwargs.pop('input_weight_assign_to', None)
		self.leakage = kwargs.pop('leakage', 0.0)
		self.bias_nodes = kwargs.pop('bias_nodes', 0.0)
		self.regularization = kwargs.pop('regularization', 1e-8)	 # chosen on validation
		self.model_seed = kwargs.pop('model_seed', 0)

		# Links: class I units; drive sets frequency, constant attractive diffusive coupling
		self.I_ref = kwargs.pop('I_ref', I_C + 15)				  # operating point on the f-I curve
		self.g_freq = kwargs.pop('g_freq', 10.0)					# drive -> current gain (detuning)
		self.link_coupling = kwargs.pop('link_coupling', 0.63)	  # K; kappa = K / g_freq ~ 0.063
		self.coupling_I_max = kwargs.pop('coupling_I_max', 10.0)	# clip on coupling current
		self.omega0 = kwargs.pop('omega0', 0.4)

		# Context and modulation
		self.link_strength_change_ratio = kwargs.pop('link_strength_change_ratio', 1.0)   # c, keep <= 1
		self.mod_tau = kwargs.pop('mod_tau', 60)					# low-pass of the relative phasor, in steps

		self.frozen = False

		if not np.isscalar(self.input_weight):
			if self.input_weight_assign_to is None:
				raise ValueError("Must pass 'input_weight_assign_to' if 'input_weight' is a list")
			if np.sum(self.input_weight_assign_to) != self.input_dims:
				raise ValueError("Sum of 'input_weight_assign_to' must equal 'input_dims'")
			self.input_weight_assign_to = np.cumsum(self.input_weight_assign_to)

		# Check for unused keys (i.e. arguments that don't work)
		if len(kwargs):
			raise ValueError("Keyword argument(s) are not implemented in this class: %s" % ", ".join(kwargs.keys()))

		# Graphs
		self.average_degree_links = self.num_nodes // 2
		self.node_adj_matrix = self.gen_node_adj_matrix()
		self.num_links = self.node_adj_matrix.nnz
		self.incidence_T, self.incidence_norm = self.gen_incidence_T()
		self.input_weights = self.gen_input_weights()
		self.output_weights = np.zeros((self.input_dims, self.num_nodes))
		self.link_adj_matrix, self.link_adj_deg = self.gen_link_adj_matrix()

		# Link time scale and reference cycle
		self.T_ref = self.reference_period(self.I_ref, h=H_MAX)
		self.dt_link = self.T_ref * self.omega0 / (2 * np.pi)
		self.n_sub = int(np.ceil(self.dt_link / H_MAX))
		v_cyc, n_cyc = self.reference_cycle(self.I_ref, h=H_MAX)
		idx = np.linspace(0, len(v_cyc), CYCLE_POINTS, endpoint=False).astype(int)
		self.v_cycle, self.n_cycle = v_cyc[idx], n_cyc[idx]
		self.v_scale, self.n_scale = np.ptp(self.v_cycle), np.ptp(self.n_cycle)

		self.reset_initial_states()
		self.reset_history()
	
	def gen_node_adj_matrix(self):
		unbounded_links = sparse.random(self.num_nodes, self.num_nodes, density=self.average_degree_nodes/self.num_nodes, random_state=self.model_seed)
		bounded_links = 2*unbounded_links - unbounded_links.ceil()
		rho = np.max(np.abs(np.linalg.eigvals(bounded_links.toarray())))
		adj_mat = self.spectral_radius / rho * bounded_links
		adj_mat.sort_indices()
		return adj_mat

	def gen_incidence_T(self):
		rows, cols = self.link_coords()
		e = np.arange(self.num_links)
		other = rows != cols                                         # self-loop: single entry, norm 1
		r = np.concatenate([e, e[other]])
		c = np.concatenate([rows, cols[other]])
		incidence_T = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(self.num_links, self.num_nodes))
		return incidence_T, 1.0 + other

	def gen_input_weights(self):
		qq = self.num_nodes // self.input_dims
		input_weights = np.zeros((self.num_nodes, self.input_dims))
		for i in range(self.input_dims):
			rng = np.random.default_rng(self.model_seed + i)
			ip = 2 * rng.random(qq) - 1
			if np.isscalar(self.input_weight):
				input_weights[i*qq:(i+1)*qq, i] = self.input_weight*ip
			else:
				input_weights[i*qq:(i+1)*qq, i] = self.input_weight[np.sort(np.where(self.input_weight_assign_to > i)[0])[0]]*ip
		return input_weights

	def gen_link_adj_matrix(self):
		A = sparse.random(self.num_links, self.num_links, density=self.average_degree_links / self.num_links, random_state=self.model_seed + 3, format="csr").ceil()
		deg = np.asarray(A.sum(axis=1)).ravel()
		return A, deg

	def reset_initial_states(self, offset=1):
		rng = np.random.default_rng(self.model_seed + offset)
		k = rng.integers(0, len(self.v_cycle), self.num_links)
		self.node_states = np.zeros((self.num_nodes,))
		self.link_states = np.stack([self.v_cycle[k], self.n_cycle[k]], axis=1)
		self.reset_link_bookkeeping()

	def reset_history(self):
		self.node_states_history, self.link_states_history, self.link_phase_history, self.training_data_history, self.rel_lp_history = [], [], [], [], []

	def advance_links(self, save_history=True):
		node_drive = (self.incidence_T @ ((self.node_states + 1) / 2)) / self.incidence_norm
		I_ext = np.maximum(self.I_ref + self.g_freq * (node_drive - 0.5), I_FLOOR)
		self.integrate_and_update(I_ext, self.link_coupling)
		self.link_phase = self.measured_phase()

		if save_history:
			self.link_states_history.append(np.copy(self.link_states))
			self.link_phase_history.append(np.copy(self.link_phase))

		
	def modulation(self):
		# m = 1 - (c/2)(1 + Im rel_lp): per-link scale on W, in [1 - c, 1]
		return 1 - (self.link_strength_change_ratio / 2) * (1 + np.imag(self.rel_lp))

	def advance_nodes(self, input_state, save_history=True):
		# relative phasor: each link's phase measured from the population mean phase psi
		valid = np.isfinite(self.link_phase)
		psi = np.angle(np.mean(np.exp(1j * self.link_phase[valid]))) if valid.any() else 0.0
		rel = np.zeros(self.link_phase.shape, dtype=complex)
		rel[valid] = np.exp(1j * (self.link_phase[valid] - psi))      # undefined phase -> 0 (midpoint modulation)

		# context: low-passed relative phasor, held once frozen
		if self.rel_lp is None:
			self.rel_lp = rel
		elif not self.frozen:
			self.rel_lp = self.rel_lp + (rel - self.rel_lp) / self.mod_tau

		W = self.node_adj_matrix.copy()
		W.data *= self.modulation()                                    # link order = CSR order of node_adj_matrix
		self.node_states = self.leakage * self.node_states + (1 - self.leakage) * np.tanh(
			W.dot(self.node_states) + self.input_weights @ input_state + self.bias_nodes)

		if save_history:
			self.node_states_history.append(np.copy(self.node_states))
			self.training_data_history.append(np.copy(input_state))
			self.rel_lp_history.append(self.rel_lp.copy())

	def advance(self, input_state, save_history=True):
		self.advance_nodes(input_state, save_history=save_history)
		self.advance_links(save_history=save_history)

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

	def create_model(self, lsqrs=False):
		if not lsqrs:
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
				ridge_model = self.create_model(lsqrs=True)
				ridge_model.fit(reg_node_states, training_data)
		self.output_weights = ridge_model.coef_

	def get_history(self):
		return np.asarray(self.node_states_history).T, np.asarray(self.link_states_history).T, np.asarray(self.link_phase_history).T, np.asarray(self.training_data_history).T

	def get_global_parameters(self):
		z = np.nanmean(np.exp(1j * np.asarray(self.link_phase_history)), axis=1)
		return np.abs(z), np.angle(z)

	def link_coords(self):
		# (row, col) of every link in CSR order, which is the link order used everywhere else
		rows = np.repeat(np.arange(self.num_nodes), np.diff(self.node_adj_matrix.indptr))
		return rows, self.node_adj_matrix.indices

	def get_input_parameters(self, endpoint="row"):
		rows, cols = self.link_coords()
		phases = np.asarray(self.link_phase_history)

		R, psi = [], []
		for d in range(self.input_dims):
			driven = np.flatnonzero(self.input_weights[:, d])

			if endpoint == "row":
				mask = np.isin(rows, driven)
			elif endpoint == "col":
				mask = np.isin(cols, driven)
			else:
				mask = np.isin(rows, driven) | np.isin(cols, driven)
			
			z = np.nanmean(np.exp(1j * phases[:, mask]), axis=1)
			R.append(np.abs(z)); psi.append(np.angle(z))
		
		return np.asarray(R), np.asarray(psi)

	def predict(self, test_data, warmup_time, reentrain_time=50, prediction_time=None, offset=1):
		if prediction_time is None:
			prediction_time = test_data.shape[1] - warmup_time - reentrain_time

		# 1. settle: nodes and links both driven by the data; rel_lp low-passes the context
		self.reset_initial_states(offset)
		self.reset_history()
		for t in range(warmup_time):
			self.advance(test_data[:, t], save_history=False)

		# 2. freeze: rel_lp (and so the modulation) is held from here on; links are no longer advanced
		self.frozen = True

		# 3. re-entrain: open-loop node updates under the frozen modulation
		t = warmup_time
		for _ in range(reentrain_time):
			self.advance_nodes(test_data[:, t], save_history=False)
			t += 1

		# 4. closed loop
		out = self.output_weights @ self.node_states			# predicts test_data[:, t]
		preds = np.empty((self.input_dims, prediction_time))
		for k in range(prediction_time):
			preds[:, k] = out
			self.advance_nodes(out, save_history=False)
			out = self.output_weights @ self.node_states

		return preds, t

	# Neuron model helper functions
	def rk4_substep(self, v, n, I_ext, h=H_MAX):
		rhs = lambda v, n: single_unit(v, n, I_ext(v) if callable(I_ext) else I_ext)

		k1 = rhs(v, n)
		k2 = rhs(v + 0.5*h*k1[0], n + 0.5*h*k1[1])
		k3 = rhs(v + 0.5*h*k2[0], n + 0.5*h*k2[1])
		k4 = rhs(v + h*k3[0], n + h*k3[1])

		v = v + (h/6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
		n = n + (h/6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
			
		return v, n

	def reference_period(self, I_ref, t_trans=1000, t_meas=1500, h=H_MAX):
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

	def reference_cycle(self, I_ref, t_trans=1000, h=H_MAX):
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
		# diffusive coupling: gain * (mean neighbour v - v), clipped; zero for links with no neighbours
		h = self.dt_link / self.n_sub
		v, n = self.link_states[:, 0], self.link_states[:, 1]
		I_max = self.coupling_I_max
		has_nbr = self.link_adj_deg > 0
		deg_safe = np.maximum(self.link_adj_deg, 1)
		coup = lambda v: np.clip(gain * (self.link_adj_matrix.dot(v) / deg_safe - has_nbr * v), -I_max, I_max)
		offset = np.copy(self.I_coup_mean)                           # DC subtraction, held over the step
		current = lambda v: I_ext + coup(v) - offset

		for _ in range(self.n_sub):
			v_new, n = self.rk4_substep(v, n, current, h=h)
			fired = (v < 0) & (v_new >= 0)
			self.spike_counts += fired.astype(int)
			if np.any(fired):
				t_spk = self.link_time + h * (0 - v[fired]) / (v_new[fired] - v[fired])
				self.last_isi[fired] = t_spk - self.last_spike[fired]
				self.last_spike[fired] = t_spk
			v = v_new
			self.I_coup_mean += (h / self.T_ref) * (coup(v) - self.I_coup_mean)
			self.link_time += h

		self.link_states = np.stack([v, n], axis=1)

	def reset_link_bookkeeping(self):
		self.link_time = 0.0
		self.last_spike = np.ones(self.num_links) * -np.inf
		self.last_isi = np.ones(self.num_links) * np.nan
		self.spike_counts = np.zeros(self.num_links).astype(int)
		self.frozen = False
		self.link_phase = self.continuous_phase()
		self.I_coup_mean = np.zeros(self.num_links)
		self.rel_lp = None
		
	def spike_phase(self):
		phase = 2 * np.pi * np.mod((self.link_time - self.last_spike)/self.last_isi, 1.0)
		phase[~np.isfinite(self.last_isi)] = np.nan
		return phase

	def continuous_phase(self):
		# All links run on the I_ref cycle, so nearest point on it is an accurate phase (up to coupling perturbations)
		if not np.all(np.isfinite(self.link_states)):
			raise FloatingPointError("link states diverged at t=%.1f" % self.link_time)

		v, n = self.link_states[:, 0], self.link_states[:, 1]
		d = ((v[:, None] - self.v_cycle[None, :]) / self.v_scale)**2 + ((n[:, None] - self.n_cycle[None, :]) / self.n_scale)**2
		return 2 * np.pi * np.argmin(d, axis=1) / len(self.v_cycle)

	def measured_phase(self):
		# Linear-in-time between spikes: uniform phase velocity, so no static offset in z
		phase = self.spike_phase()					 # NaN until a link has a finite ISI
		undefined = np.isnan(phase)
		if np.any(undefined):
			phase[undefined] = self.continuous_phase()[undefined]   # only during the first period or so
		return phase
