import numpy as np


def ccsd(f, g, N, Eccsd, Rccsdt1, Rccsdt2, tol=1e-12):
    o = slice(0, N)
    v = slice(N, None)
    e = np.diag(f)
    eo = e[o]
    ev = e[v]
    oovv = g[o, o, v, v]
    D1 = eo[:, None] - ev[None, :]
    D2 = eo[:, None, None, None] + eo[None, :, None, None]- ev[None, None, :, None]  - ev[None, None, None, :]
    t1_o = np.zeros((D1.shape))
    t2_o = oovv / D2
    E_o = Eccsd(f, g, t1_o, t2_o)

    for n_iter in range(1, 201):
        t1_n = t1_o + Rccsdt1(f, g, t1_o, t2_o) / D1
        t2_n = t2_o + Rccsdt2(f, g, t1_o, t2_o) / D2
        E_n = Eccsd(f, g, t1_n, t2_n)

        if np.abs(E_n - E_o) < tol:
            return E_n, n_iter, t1_n, t2_n
        t1_o = t1_n
        t2_o = t2_n
        E_o = E_n
    raise RuntimeError("ccsd did not converge")