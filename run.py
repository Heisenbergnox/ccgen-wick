import numpy as np
from hf_to_cc import H2O, integrals, scf_diis, ao2mo, spin_orbitals
from ccd_residual import residual
from ccd import ccd
from ccsd_residual import Eccsd, Rccsdt1, Rccsdt2
from ccsd import ccsd
from ccsd_t_residual import tri
from ccsd_t import ccsd_t
mol, S, h, eri, E_nuc, N = integrals(H2O, "sto-3g")
E_hf, eps, C, n_scf = scf_diis(S, h, eri, E_nuc, N)
mo, h_mo = ao2mo(C, eri, h)
hso, g = spin_orbitals(mo, h_mo)
E_ccd, n_iter_ccd = ccd(residual, np.diag(np.repeat(eps, 2)), g, N)
E_ccsd, n_iter_ccsd, t1, t2 = ccsd(np.diag(np.repeat(eps, 2)), g, N, Eccsd, Rccsdt1, Rccsdt2)
E_t = ccsd_t(np.diag(np.repeat(eps, 2)), g, N, t1, t2, tri)
print(E_ccd, n_iter_ccd)
print(E_ccsd, n_iter_ccsd)
print(E_t)
