import numpy as np


def residual(f, g, t2):
    N = t2.shape[0]
    o = slice(0, N)
    v = slice(N, None)
    r = 1.0 * np.einsum('jkab,ki->ijab', t2, f[o, o], optimize=True)
    r -= 1.0 * np.einsum('ikab,kj->ijab', t2, f[o, o], optimize=True)
    r += 1.0 * np.einsum('ijac,bc->ijab', t2, f[v, v], optimize=True)
    r -= 1.0 * np.einsum('ijbc,ac->ijab', t2, f[v, v], optimize=True)
    r += 1.0 * np.einsum('abij->ijab', g[v, v, o, o], optimize=True)
    r += 0.5 * np.einsum('klab,klij->ijab', t2, g[o, o, o, o], optimize=True)
    r += 1.0 * np.einsum('jkac,bkci->ijab', t2, g[v, o, v, o], optimize=True)
    r -= 1.0 * np.einsum('jkbc,akci->ijab', t2, g[v, o, v, o], optimize=True)
    r -= 1.0 * np.einsum('ikac,bkcj->ijab', t2, g[v, o, v, o], optimize=True)
    r += 1.0 * np.einsum('ikbc,akcj->ijab', t2, g[v, o, v, o], optimize=True)
    r += 0.5 * np.einsum('ijcd,abcd->ijab', t2, g[v, v, v, v], optimize=True)
    r += 0.5 * np.einsum('ijbc,klad,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    r -= 0.5 * np.einsum('ijac,klbd,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    r += 0.25 * np.einsum('ijcd,klab,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    r -= 0.5 * np.einsum('ikab,jlcd,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    r -= 1.0 * np.einsum('ikbc,jlad,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    r += 1.0 * np.einsum('ikac,jlbd,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    r -= 0.5 * np.einsum('ikcd,jlab,klcd->ijab', t2, t2, g[o, o, v, v], optimize=True)
    return r
