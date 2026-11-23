# ccgen-wick — generating coupled-cluster equations from second quantization

ccgen derives coupled-cluster equations the way you would on paper, contraction by contraction with Wick's theorem. It then merges like terms and writes the result as NumPy code that you can run and check. The CCSD equations have 328 terms after the first merging stage and 80 after all three. ccgen also generates the connected triples of the (T) correction, so the example runs CCSD(T).

**Coming next:**
- collecting terms related by i↔j or a↔b into P(ij), P(ab) (see Known limitations);
- CCSDT and EOM-CC;
- plugging the generated equations into the solvers of [hf_to_cc](https://github.com/Heisenbergnox/hf_to_cc);
- faster generated code, with shared intermediates instead of one `einsum` per term.

## What's inside

With $`\bar H = e^{-T} H_N\, e^{T}`$, the CCSD equations (T = T₁ + T₂) are

```math
E_\text{CCSD} = \langle \Phi_0 | \bar H | \Phi_0 \rangle = \sum_{ia} f_{ia}\, t_i^a + \frac{1}{4} \sum_{ijab} \langle ij \Vert ab \rangle\, t_{ij}^{ab} + \frac{1}{2} \sum_{ijab} \langle ij \Vert ab \rangle\, t_i^a t_j^b ,
```

```math
0 = \langle \Phi_i^a | \bar H | \Phi_0 \rangle , \qquad 0 = \langle \Phi_{ij}^{ab} | \bar H | \Phi_0 \rangle .
```

CCD (T = T₂) keeps only the second amplitude equation, and its energy drops the terms with t₁.

The (T) correction uses the converged t₁ and t₂ from CCSD. With $`D_{ijk}^{abc} = f_{ii} + f_{jj} + f_{kk} - f_{aa} - f_{bb} - f_{cc}`$:

```math
\text{connected:}\quad D_{ijk}^{abc}\, t_{ijk}^{abc}(c) = \langle \Phi_{ijk}^{abc} | (V_N T_2)_c | \Phi_0 \rangle , \qquad \text{disconnected:}\quad D_{ijk}^{abc}\, t_{ijk}^{abc}(d) = P(i/jk)\, P(a/bc)\; t_i^a \langle jk \Vert bc \rangle ,
```

```math
E_{(T)} = \frac{1}{36} \sum_{ijkabc} t_{ijk}^{abc}(c)\, D_{ijk}^{abc} \left[ t_{ijk}^{abc}(c) + t_{ijk}^{abc}(d) \right] .
```

ccgen generates $`t(c)`$. $`t(d)`$ is written by hand (see Known limitations).

- `ccgen.py`: the generator. `make_ops(bra, h, ts)` builds one part of an equation as a list of normal-ordered operators. `bra` is the excitation level of the projection (0, 1, 2 or 3), `h` is `"f"` or `"v"`, and `ts` lists the cluster operators. For example, `make_ops(1, "v", [1, 2])` is $`\langle \Phi_i^a | (V_N T_1 T_2)_c | \Phi_0 \rangle`$. For each part, ccgen
  1. finds the connected full contractions;
  2. takes each sign from the number of crossing lines;
  3. attaches the integral, the amplitudes and the coefficient: 1/(n!)² for each operator with n creators, and 1/k! for k identical cluster operators;
  4. merges like terms in three stages: the antisymmetry of ⟨pq‖rs⟩ and t₂ (`merge_antisym`), renaming the summed indices (`merge_dummy`), and a fixed order for identical cluster operators (`merge_t_order`);
  5. writes one `einsum` line per term, with `optimize=True`.
- `provided.py`: lists the parts of each equation, and `python provided.py` writes
  - `ccd_residual.py`: `residual(f, g, t2)`, 18 terms;
  - `ccsd_residual.py`: `Eccsd`, `Rccsdt1`, `Rccsdt2`, all `(f, g, t1, t2)`, with 3, 14 and 63 terms;
  - `ccsd_t_residual.py`: `tri(f, g, t1, t2)`, the 18 terms of $`D_{ijk}^{abc}\, t_{ijk}^{abc}(c)`$.
- `ccd.py`, `ccsd.py`: `ccd(R, f, g, N)` and `ccsd(f, g, N, Eccsd, Rccsdt1, Rccsdt2)` iterate the generated equations, starting from MP2 amplitudes and t₁ = 0. `ccsd` also returns t₁ and t₂.
- `ccsd_t.py`: `ccsd_t(f, g, N, t1, t2, tri)` returns $`E_{(T)}`$.
- `run.py`: the example in Usage.

### What goes in

**Index letters** (spin orbitals):
- i–n are occupied, a–f are virtual, and p, q, r, s are the Hamiltonian's indices before contraction.
- The projection's indices (i, j, a, b) stay free.
- The cluster operators use k–n and c–f, which are summed over. In the triples projection, k and c are free too, so the cluster operators use l–n and d–f there.

**Input format:** an operator is a tuple `(index, "+" or "-", space, group)`.
- The space is `"o"`, `"v"` or `"g"` (general).
- The group is 0 for the projection, 1 for the Hamiltonian, and 2, 3, … for the cluster operators.

For example, `make_ops(2, "v", [2])`, which is $`\langle \Phi_{ij}^{ab} | (V_N T_2)_c | \Phi_0 \rangle`$, gives

```
[('i', '+', 'o', 0), ('j', '+', 'o', 0), ('b', '-', 'v', 0), ('a', '-', 'v', 0),
 ('p', '+', 'g', 1), ('q', '+', 'g', 1), ('s', '-', 'g', 1), ('r', '-', 'g', 1),
 ('c', '+', 'v', 2), ('d', '+', 'v', 2), ('l', '-', 'o', 2), ('k', '-', 'o', 2)]
```

A part is named by its operators: VT₁T₂ is $`(V_N T_1 T_2)_c`$. `provided.py` passes every combination of `"f"` or `"v"` with the cluster operators. Combinations without a connected full contraction give no terms, so nothing is picked by hand. The parts that give terms:

| Equation | Parts (terms) | Total | First merging stage only |
|---|---|---|---|
| CCD T₂ | V 1, FT₂ 4, VT₂ 6, VT₂² 7 | 18 | 125 |
| CCSD energy | FT₁ 1, VT₂ 1, VT₁² 1 | 3 | 3 |
| CCSD T₁ | F 1, FT₁ 2, FT₂ 1, FT₁² 1, VT₁ 1, VT₂ 2, VT₁² 2, VT₁T₂ 3, VT₁³ 1 | 14 | 32 |
| CCSD T₂ | V 1, FT₂ 4, FT₁T₂ 4, VT₁ 4, VT₂ 6, VT₁² 6, VT₁T₂ 16, VT₂² 7, VT₁³ 4, VT₁²T₂ 10, VT₁⁴ 1 | 63 | 293 |
| (T) connected triples | VT₂ 18 | 18 | 36 |

The generated CCSD energy is the formula above, term by term:

```python
def Eccsd(f, g, t1, t2):
    N = t2.shape[0]
    o = slice(0, N)
    v = slice(N, None)
    r = 1.0 * np.einsum('kc,kc->', t1, f[o, v], optimize=True)
    r += 0.25 * np.einsum('klcd,klcd->', t2, g[o, o, v, v], optimize=True)
    r += 0.5 * np.einsum('kc,ld,klcd->', t1, t1, g[o, o, v, v], optimize=True)
    return r
```

## Results

Water in STO-3G, correlation energies: CCD gives −0.0506409699 Eh (29 iterations), CCSD −0.0509029526 Eh (33 iterations), and (T) −0.0000686513 Eh. All three together take under a second.

I compared the results with PySCF (converged to 1e-12) for 14 molecules in STO-3G, from H₂ to CH₃OH, and for water in 6-31G and cc-pVDZ:

| | Largest difference (Eh) | Most runs (Eh) |
|---|---|---|
| CCD, against `pyscf.cc.ccd.CCD` | 3e-9 (CO) | ≲ 1e-10 |
| CCSD | 6e-10 (CO) | ~ 1e-11 |
| (T), against `ccsd_t()` (H₂ has no triples) | 7e-10 (CO) | ≲ 1e-11 |

The CCSD difference for CO comes from the SCF orbitals: on PySCF's own orbitals, CO agrees to 4e-12 Eh. For (T), see Known issue. In cc-pVDZ (24 basis functions), water takes 1.6 s for CCSD and 3 s for (T).

## Known limitations and issues

**Limitations**
- **Permutation operators:** terms related by i↔j or a↔b are not yet collected into P(ij), P(ab). So CCD comes out as 18 `einsum` lines, where the textbook writes ten terms. The results are not affected.
- **Disconnected terms:** $`t(d)`$ is not generated. The generator keeps only terms in which every cluster operator is contracted with the Hamiltonian, so `ccsd_t.py` writes $`t(d)`$ by hand.

**Known issue**
- For CO in STO-3G, (T) differs from PySCF by 5e-10 Eh even on PySCF's own orbitals, where CCSD agrees to 4e-12 Eh. Converging CCSD more tightly does not change it. The cause is not known yet.

## Usage

The example takes its integrals from [hf_to_cc](https://github.com/Heisenbergnox/hf_to_cc). Clone both repositories side by side and install the dependencies:

```
git clone https://github.com/Heisenbergnox/hf_to_cc
git clone https://github.com/Heisenbergnox/ccgen-wick
pip install numpy scipy pyscf
cd ccgen-wick
```

The generated files are already in the repository. `python provided.py` writes all three again. It needs only the standard library and takes about 0.1 s.

`PYTHONPATH=../hf_to_cc python run.py` prints CCD and CCSD (energy in Eh, number of iterations), then (T): −0.0506409699 and 29, −0.0509029526 and 33, −0.0000686513.

```python
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
```

**Conventions:**
- Spin orbitals, with the `N` occupied ones first.
- `g[p, q, r, s]` is $`\langle pq \Vert rs \rangle`$, `t1[i, a]` is $`t_i^a`$, and `t2[i, j, a, b]` is $`t_{ij}^{ab}`$.
- `f` is the Fock matrix. For canonical HF orbitals it is `np.diag(np.repeat(eps, 2))`, but the generated equations also accept a full Fock matrix.
- `ccsd_t` assumes canonical orbitals and takes the orbital energies from the diagonal of `f`.

## License

MIT, see [LICENSE](LICENSE).
