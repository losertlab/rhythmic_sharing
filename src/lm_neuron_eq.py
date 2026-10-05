import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

# Helper functions
V_lo, V_hi = -48, 32

def clip_voltage(V):
	return (np.clip(V, V_lo, V_hi) - V_lo)/(V_hi - V_lo) # Bound found from Check -1

def N_SS(v, V_3=12, V_4=17.4):
	return 0.5 * (1 + np.tanh((v - V_3)/V_4))

def initial_states(n, seed=69):
	states = np.zeros((n, 2))
	rng = np.random.default_rng(seed=seed)
	states[:, 0] = rng.uniform(size=(n,)) * (V_hi - V_lo) + V_lo
	states[:, 1] = N_SS(states[:, 0])
	return states

# Model
def single_unit(
		v, n, I,
		g_L=2, g_Ca=4, g_K=8,
		V_L=-60, V_Ca=120, V_K=-84,
		V_1=-1.2, V_2=18, V_3=12, V_4=17.4,
		C=20, phi=1/15
	):

	M_SS = 0.5 * (1 + np.tanh((v - V_1)/V_2))
	inv_tau_N = np.cosh((v - V_3)/(2 * V_4))

	dv = I - (g_L * (v - V_L)) - (g_Ca * M_SS * (v - V_Ca)) - (g_K * n * (v - V_K))
	dn = phi * (N_SS(v, V_3=V_3, V_4=V_4) - n) * inv_tau_N

	return dv/C, dn

# Test functions
def firing_rate(I, T=20000, t_trans=2000, initial_vals=[-60.0, 0.0]):

	sol = integrate(I, T=T, initial_vals=initial_vals)

	ts = sol.t_events[0]
	ts = ts[ts > t_trans]
	
	if len(ts) < 2:
		return 0.0, sol.y[:, np.argmax(sol.t)]
	return 1000.0 / np.mean(np.diff(ts)), sol.y[:, np.argmax(sol.t)]

def integrate(I, T=20000, initial_vals=[-60.0, 0.0], terminate_on_cross=False, rhs=None, max_step=1.0):
	if not rhs:
		rhs = lambda t, y, I: single_unit(y[0], y[1], I)

	spike = lambda t, y, I: y[0]
	spike.direction = 1
	spike.terminal = terminate_on_cross

	sol = solve_ivp(
		rhs, (0, T), initial_vals, args=(I,),
		events=spike, rtol=1e-6, atol=1e-8, max_step=max_step
	)

	return sol

def time_to_spike(I, T=20000, initial_vals=[-60.0, 0.0]):
	sol = integrate(I, T=T, initial_vals=initial_vals, terminate_on_cross=True)
	return sol.t_events[0][0] if len(sol.t_events[0]) else np.nan

if __name__ == "__main__":
	I_c = 39.96

	# Check -1: Plot firing
	plt.plot(integrate(I_c + 1).y[0, :], label="+1")
	plt.plot(integrate(I_c + 5).y[0, :], label="+5")
	plt.plot(integrate(I_c + 10).y[0, :], label="+10")

	plt.xlabel('V')
	plt.ylabel('T (ms)')
	plt.legend()
	plt.tight_layout()
	plt.show()

	# Check parameters/equations
	# Check 0: the square of the firing rate is linear

	Is = np.linspace(I_c - 0.5, I_c + 0.5, 51)
	f = np.array([firing_rate(I)[0] for I in tqdm(Is)])**2

	plt.plot(Is, f, 'o-')
	plt.xlabel(r'$I$ ($\mu$A/cm$^2$)')
	plt.ylabel(r'Firing rate squared (Hz$^2$)')
	plt.tight_layout()
	plt.show()

	# Check 1: Hysteresis
	Is = np.linspace(I_c - 1, I_c + 1, 51)

	up_res = []
	initial = [-60.0, 0.0]
	for i in tqdm(Is):
		rate, inital = firing_rate(i, initial_vals=initial)
		up_res.append(rate)
	plt.plot(Is, up_res, 'o-', label="Upwards Sweep")

	down_res = []
	initial = [-60.0, 0.0]
	for i in tqdm(Is[::-1]):
		rate, inital = firing_rate(i, initial_vals=initial)
		down_res.append(rate)
	plt.plot(Is[::-1], down_res, 'o-', label="Downwards Sweep")

	plt.xlabel(r'$I$ ($\mu$A/cm$^2$)')
	plt.ylabel(r'Firing rate (Hz)')
	plt.legend()
	plt.tight_layout()
	plt.show()

	# Check 2: PRC
	sol = integrate(I_c + 0.1, T=20000)
	ts, ys = sol.t_events[0], sol.y_events[0]
	ts, ys = ts[ts > 5000], ys[ts > 5000] # After transient
	s0 = ys[np.argmin(ts)] # First spike

	s0_temp = integrate(I_c + 0.1, T=1, initial_vals=s0).y[:, -1] # Just get a little away from spike
	T0 = 1 + time_to_spike(I_c + 0.1, initial_vals=s0_temp)

	# Kick at phases
	for dV in [0.1, 0.2, 0.5]:
		res = []
		
		for i, ph in tqdm(enumerate(np.arange(100)/100)):
			t_kick = ph * T0
			s = integrate(I_c + 0.1, t_kick, initial_vals=s0)
			s = s.y[:, -1].copy()
			v_pre = s[0]
			s[0] += dV

			if v_pre < 0 <= s[0]: # If true, the kick crossed the threshold (0)
				T1 = t_kick
			else:
				T1 = t_kick + time_to_spike(I_c + 0.1, initial_vals=s)
			res.append((T0 - T1) / dV)

		plt.plot(np.arange(100)/100, res, 'o-', label=r"$\Delta$V=%.2f" % dV)

	plt.xlabel(r'Phase at kick, $\varphi$')
	plt.ylabel('Spike advance per mV (ms/mV)')
	plt.title("PRC, I = %.2f" % (I_c + 0.1))
	plt.legend()
	plt.tight_layout()
	plt.show()

	# Check 3: Sub-threshold excitability
	a_stars = []
	num_events = np.zeros((3, 100))
	latency = np.ones((3, 100)) * np.nan
	for i, d in enumerate([0.5, 2, 5]):
		I = I_c - d

		rest = integrate(I).y[:, -1].copy()
		
		for j, a in tqdm(enumerate(np.linspace(0, 150, 100))):
			s = integrate(I + a, T=2, initial_vals=rest).y[:, -1].copy()
			sol = integrate(I, T=2000, initial_vals=s)
			num_events[i, j] = len(sol.t_events[0])
			if num_events[i, j] > 0:
				latency[i, j] = np.min(sol.t_events[0]) + 2

		num_events = np.asarray(num_events)
		latency = np.asarray(latency)

		A_low = np.linspace(0, 150, 100)[num_events[i, :] == 0][-1]
		A_high = np.linspace(0, 150, 100)[num_events[i, :] == 1][0]

		while A_high - A_low > 1e-6:
			A_mid = (A_low + A_high) / 2
			print(A_mid, A_low, A_high, )
			s = integrate(I + A_mid, T=2, initial_vals=rest).y[:, -1].copy()
			sol = integrate(I, T=2000, initial_vals=s)
			if len(sol.t_events[0]) > 0:
				A_high = A_mid
			else:
				A_low = A_mid
		a_stars.append(A_high)

	plt.plot(np.linspace(0, 150, 100),
		num_events.T, "o-", label=[
		r"$\delta=-%.2f$" % d for d in [0.5, 2, 5]
	])
	plt.xlabel(r'Amplitude kick ($\mu$A/cm$^2$)')
	plt.ylabel('Num spikes')
	plt.legend()
	plt.tight_layout()
	plt.show()
	
	plt.plot(
		np.expand_dims(np.linspace(0, 150, 100), -1) - np.expand_dims(a_stars, 0),
		latency.T, "o-", label=[
		r"$\delta=-%.2f$" % d for d in [0.5, 2, 5]
	])
	plt.xlabel(r'$A - A^\ast$ ($\mu$A/cm$^2$)')
	plt.ylabel('Latency to first spike')
	plt.legend()
	plt.tight_layout()
	plt.show()

	plt.plot([0.5, 2, 5], a_stars)
	plt.xlabel(r'Distance below onset ($\mu$A/cm$^2$)')
	plt.ylabel(r'$A^\ast$ ($\mu$A/cm$^2$)')
	plt.tight_layout()
	plt.show()

	# Check 4: Time-varying drive and phase locking
	f0, _ = firing_rate(I_c + 0.5)
	T0 = 1000 / f0

	As = np.linspace(0, 0.5, 20)

	spikes_per_cycle = np.zeros((50, 20))
	phase_concentration = np.zeros((50, 20))
	locked = np.zeros((50, 20)).astype(bool)

	for i, r in enumerate(tqdm(np.linspace(0.5, 2, 50))):
		f_d = r * f0
		T_d = 1000 / f_d
		for j, a in enumerate(As):
			I = lambda t: (I_c + 0.5) + (a * np.sin((2 * np.pi * t)/T_d))
			rhs_temp = lambda t, y, _: single_unit(y[0], y[1], I(t))
			sol = integrate(I, T=(150 * T_d), rhs=rhs_temp, max_step=np.inf)
			ts = sol.t_events[0]
			ts = ts[ts > 50 * T_d]

			spikes_per_cycle[i, j] = len(ts) / 100
			psi_k = ((2 * np.pi * ts) / T_d) % (2 * np.pi)
			phase_concentration[i, j] = np.abs(np.mean(np.exp(1j * psi_k))).astype(float)
			locked[i, j] = (np.abs(spikes_per_cycle[i, j] - 1) < 0.01) and (phase_concentration[i, j] > 0.99)

	plt.imshow(locked.T, extent=[0.5, 2, 0, 0.5], origin="lower")
	plt.xlabel('Driven Frequency over Natural')
	plt.ylabel(r'Amplitude ($\mu$A/cm$^2$)')
	plt.tight_layout()
	plt.show()

	fig, ax = plt.subplots()
	im = ax.imshow(spikes_per_cycle.T, extent=[0.5, 2, 0, 0.5], origin="lower")
	fig.colorbar(im, ax=ax, label="Number of Spikes", orientation="horizontal", location="top")
	plt.xlabel('Driven Frequency over Natural')
	plt.ylabel(r'Amplitude ($\mu$A/cm$^2$)')
	plt.tight_layout()
	plt.show()