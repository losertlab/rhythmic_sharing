import copy
from spike_sharing import SpikingNetwork
from rhythmic_sharing import RhythmicNetwork
from dynamics_testing import attractor_dist
import scipy.sparse as sparse
import numpy as np

# import spike_sharing
# import importlib
# importlib.reload(spike_sharing)
# SpikingNetwork = spike_sharing.SpikingNetwork

# Defaults
ALPHAS = [1e-12, 1e-20, 1e-8, 1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

# Both models re-entrain and closed loop
def run_frozen(warm, x, t, W_out, n_reentrain, n_auto):
	w = copy.deepcopy(warm); w.output_weights = W_out
	for _ in range(n_reentrain):
		w.advance_nodes(x[:, t], save_history=False); t += 1
	out, preds = W_out @ w.node_states, np.empty((3, n_auto))
	for k in range(n_auto):
		preds[:, k] = out; w.advance_nodes(out, save_history=False); out = W_out @ w.node_states
	return preds, t

def ridge_sweep(net):
	W = {}
	for a in ALPHAS:
		net.regularization, net.lsqrs = a, False
		net.compute_weights(); W[a] = net.output_weights.copy()
	return W

def link_cost(net, n_steps):
	rate = net.spike_counts/n_steps
	out_deg = np.asarray(net.link_adj_matrix.sum(axis=0)).ravel()

	event_msgs = rate @ out_deg
	dense_msgs = net.link_adj_matrix.nnz

	return {
		"n_steps": n_steps,
		"rate_mean": np.mean(rate),
		"rate_median": np.median(rate),
		"frac_silent": np.mean(net.spike_counts == 0),
		"frac_sparse": np.mean(rate < 0.1 * np.median(rate)),

		"event_msgs": event_msgs,
		"dense_msgs": dense_msgs,
		"msg_ratio": event_msgs / dense_msgs
	}

# Spiking network harness
def spk_train(_, seed, c, train, warm_time, save_hist=False):
	net = SpikingNetwork(model_seed=seed, link_strength_change_ratio=c)
	net.train(train, warmup_time=warm_time)
	W = ridge_sweep(net)
	if not save_hist:
		net.reset_history()
	return net, W

def spk_settle(net, x, settle, reentrain, offset=None, save_hist=False):
	w = copy.deepcopy(net); w.reset_initial_states(offset); w.reset_history()   # also zeroes spike_counts
	for t in range(settle):
		w.advance(x[:, t], save_history=save_hist)
	w.frozen = True
	cost = link_cost(w, settle)
	return w, settle, reentrain, True, cost

# Rhythmic network harness
class RN(RhythmicNetwork):
	def gen_node_adj_matrix(self):					   # dense eigvals instead of ARPACK, same as SpikingNetwork
		u = sparse.random(self.num_nodes, self.num_nodes, density=self.average_degree_nodes / self.num_nodes,
						  random_state=self.model_seed)
		b = 2 * u - u.ceil()
		return self.spectral_radius / np.max(np.abs(np.linalg.eigvals(b.toarray()))) * b

def rs_train(base, seed, c, train, warm_time, save_hist=False):
	net = RN(**base, link_strength_change_ratio=c, model_seed=seed)

	# SpikingNet has better weight initialization
	spk = SpikingNetwork(model_seed=seed)
	net.input_weights = spk.input_weights.copy()
	assert np.allclose(net.node_adj_matrix.toarray(), spk.node_adj_matrix.toarray())
	
	net.train(train, warmup_time=warm_time)
	W = ridge_sweep(net)
	if not save_hist:
		net.reset_history()
	return net, W

def rs_settle(net, x, _1, _2, offset=None, save_hist=False):
	   cost = dict(event_msgs=np.nan, dense_msgs=float(net.link_adj_matrix.nnz), msg_ratio=np.nan,
				   rate_mean=np.nan, frac_silent=np.nan)
	   return None, None, None, True, cost

# Getter
def get_harness():	
	return {"spk": (spk_train, spk_settle), "rs": (rs_train, rs_settle)}
