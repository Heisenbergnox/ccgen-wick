import numpy as np


def tri(f, g, t1, t2):
    N = t2.shape[0]
    o = slice(0, N)
    v = slice(N, None)
    r = 1.0 * np.einsum('klab,clij->ijkabc', t2, g[v, o, o, o], optimize=True)
    r -= 1.0 * np.einsum('klac,blij->ijkabc', t2, g[v, o, o, o], optimize=True)
    r += 1.0 * np.einsum('klbc,alij->ijkabc', t2, g[v, o, o, o], optimize=True)
    r -= 1.0 * np.einsum('jlab,clik->ijkabc', t2, g[v, o, o, o], optimize=True)
    r += 1.0 * np.einsum('jlac,blik->ijkabc', t2, g[v, o, o, o], optimize=True)
    r -= 1.0 * np.einsum('jlbc,alik->ijkabc', t2, g[v, o, o, o], optimize=True)
    r += 1.0 * np.einsum('jkad,bcdi->ijkabc', t2, g[v, v, v, o], optimize=True)
    r -= 1.0 * np.einsum('jkbd,acdi->ijkabc', t2, g[v, v, v, o], optimize=True)
    r += 1.0 * np.einsum('jkcd,abdi->ijkabc', t2, g[v, v, v, o], optimize=True)
    r += 1.0 * np.einsum('ilab,cljk->ijkabc', t2, g[v, o, o, o], optimize=True)
    r -= 1.0 * np.einsum('ilac,bljk->ijkabc', t2, g[v, o, o, o], optimize=True)
    r += 1.0 * np.einsum('ilbc,aljk->ijkabc', t2, g[v, o, o, o], optimize=True)
    r -= 1.0 * np.einsum('ikad,bcdj->ijkabc', t2, g[v, v, v, o], optimize=True)
    r += 1.0 * np.einsum('ikbd,acdj->ijkabc', t2, g[v, v, v, o], optimize=True)
    r -= 1.0 * np.einsum('ikcd,abdj->ijkabc', t2, g[v, v, v, o], optimize=True)
    r += 1.0 * np.einsum('ijad,bcdk->ijkabc', t2, g[v, v, v, o], optimize=True)
    r -= 1.0 * np.einsum('ijbd,acdk->ijkabc', t2, g[v, v, v, o], optimize=True)
    r += 1.0 * np.einsum('ijcd,abdk->ijkabc', t2, g[v, v, v, o], optimize=True)
    return r
