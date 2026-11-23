import numpy as np


es = lambda *a: np.einsum(*a, optimize=True)


def P3(x):
    y = x - x.transpose(1, 0, 2, 3, 4, 5) - x.transpose(2, 1, 0, 3, 4, 5)
    return y - y.transpose(0, 1, 2, 4, 3, 5) - y.transpose(0, 1, 2, 5, 4, 3)


def ccsd_t(f, g, N, t1, t2, tri):
    o = slice(0, N)
    v = slice(N, None)
    e = np.diag(f)
    eo = e[o]
    ev = e[v]

    D3 = (eo[:, None, None, None, None, None] + eo[None, :, None, None, None, None] + eo[None, None, :, None, None, None] - ev[None, None, None, :, None, None] - ev[None, None, None, None, :, None] - ev[None, None, None, None, None, :])

    tc = tri(f, g, t1, t2) / D3
    td = P3(es("ia,jkbc->ijkabc", t1, g[o, o, v, v])) / D3
    E_T = np.sum(tc * D3 * (tc + td)) / 36
    return E_T
