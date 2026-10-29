import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from scipy.stats import wasserstein_distance
import numpy as np

def sim(rhs, b, n, init, transient=2000):
    sol = solve_ivp(rhs, (0, n + transient), init, args=(b,), t_eval=np.arange(n + transient), rtol=1e-9, atol=1e-9)
    return sol.y[:, transient:]

def make_train(rhs, seg, b1=0.18, b2=0.29, n_seg=4, init=(0.1, 0.0, -0.1)): # Defaults from Thomas
    y, out = sim(rhs, b1, 1, init)[:, -1], []
    for k in range(n_seg):
        sol = solve_ivp(rhs, (0, seg), y, args=(b1 if k % 2 == 0 else b2,),
                        t_eval=np.arange(seg + 1), rtol=1e-9, atol=1e-9)
        out.append(sol.y[:, :seg]); y = sol.y[:, seg]
    return np.concatenate(out, axis=1)

def attractor_dist(a, b):
    if not np.all(np.isfinite(a)) or np.abs(a).max() > 10:
        return np.inf
    return np.mean([wasserstein_distance(a[d], b[d]) for d in range(3)])

def classify(pred, ref):
    d = {k: attractor_dist(pred, r) for k, r in ref.items()}
    best, second = sorted(d, key=d.get)[:2]
    return (best if d[best] < 1.0 and d[best] < 0.5 * d[second] else "none"), d

def valid_time(pred, truth, thresh=0.4):
    n = min(pred.shape[1], truth.shape[1])
    err = np.linalg.norm(pred[:, :n] - truth[:, :n], axis=0) / np.sqrt(np.mean(np.sum(truth ** 2, axis=0)))
    bad = np.flatnonzero(~np.isfinite(err) | (err > thresh))
    return bad[0] if len(bad) else n
