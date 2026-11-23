from ccgen import make_ops, get_residual

ccd_parts = [make_ops(2, i, j)for i in ("f", "v") for j in ([], [2], [2, 2] )]
ccsd_e_parts = [make_ops(0, i, j)for i in ("f", "v") for j in ([], [1], [2], [1, 1])]
ccsd_t1_parts = [make_ops(1, i, j)for i in ("f", "v") for j in ([], [1], [2], [1, 1], [1, 2], [1, 1, 1])]
ccsd_t2_parts = [make_ops(2, i, j)for i in ("f", "v") for j in ([], [1], [2], [1, 1], [1, 2], [1, 1, 1], [2, 2], [1, 1, 1, 1], [1, 1, 2])]
ccsd_t_parts = [make_ops(3, i, j)for i in ("v",) for j in ([2],)]

def write_function(fh, name, args, parts):
    terms = get_residual(parts).split(" + ")
    fh.write(f"\n\ndef {name}({args}):\n")
    fh.write("    N = t2.shape[0]\n    o = slice(0, N)\n    v = slice(N, None)\n")
    fh.write(f"    r = {terms[0]}\n")
    for x in terms[1:]:
        if x.startswith("-"):
            fh.write(f"    r -= {x[1:]}\n")
        else:
            fh.write(f"    r += {x}\n")
    fh.write("    return r\n")


def write_ccd_residual(path="ccd_residual.py"):
    with open(path, "w") as fh:
        fh.write("import numpy as np\n")
        write_function(fh, "residual", "f, g, t2", ccd_parts)


def write_ccsd_parts(path="ccsd_residual.py"):
    with open(path, "w") as fh:
        fh.write("import numpy as np\n")
        write_function(fh, "Eccsd", "f, g, t1, t2", ccsd_e_parts)
        write_function(fh, "Rccsdt1", "f, g, t1, t2", ccsd_t1_parts)
        write_function(fh, "Rccsdt2", "f, g, t1, t2", ccsd_t2_parts)


def write_ccsd_t_parts(path="ccsd_t_residual.py"):
    with open(path, "w") as fh:
        fh.write("import numpy as np\n")
        write_function(fh, "tri", "f, g, t1, t2", ccsd_t_parts)


if __name__ == "__main__":
    write_ccd_residual()
    write_ccsd_parts()
    write_ccsd_t_parts()
