import math
from fractions import Fraction
from collections import defaultdict
from itertools import combinations, permutations
from math import factorial


def make_ops(bra, H, Ts):

    ops = []

    if bra == 1:
        bra1 = ("i", "+", "o", 0), ("a", "-", "v", 0)
        ops += bra1
    if bra == 2:
        bra2 = ("i", "+", "o", 0), ("j", "+", "o", 0), ("b", "-", "v", 0), ("a", "-", "v", 0)
        ops += bra2
    if bra == 3:
        bra3 = ("i", "+", "o", 0), ("j", "+", "o", 0), ("k", "+", "o", 0), ("c", "-", "v", 0), ("b", "-", "v", 0), ("a", "-", "v", 0)
        ops += bra3

    if H == "f":
        Hf = ("p", "+", "g", 1), ("q", "-", "g", 1)
        ops += Hf
    else:
        Hv = ("p", "+", "g", 1), ("q", "+", "g", 1), ("s", "-", "g", 1), ("r", "-", "g", 1)
        ops += Hv

    n_t2 = 0
    oop = ["k", "l", "m", "n"]
    vop = ["c", "d", "e", "f"]
    for num, exc in enumerate(Ts):
        if bra == 3:
            if exc == 1:
                exc1 = ((vop[num + n_t2 + 1], "+", "v", num + 2), (oop[num + n_t2 + 1], "-", "o", num + 2))
                ops += exc1
            if exc == 2:
                exc2 = ((vop[num + n_t2 + 1], "+", "v", num + 2), (vop[num + n_t2 + 2], "+", "v", num + 2), (oop[num + n_t2 + 1], "-", "o", num + 2), (oop[num + n_t2 + 2], "-", "o", num + 2))
                ops += exc2
                n_t2 += 1
        else:
            if exc == 1:
                exc1 = ((vop[num + n_t2], "+", "v", num + 2), (oop[num + n_t2], "-", "o", num + 2))
                ops += exc1
            if exc == 2:
                exc2 = ((vop[num + n_t2], "+", "v", num + 2), (vop[num + n_t2 + 1], "+", "v", num + 2), (oop[num + n_t2 + 1], "-", "o", num + 2), (oop[num + n_t2], "-", "o", num + 2))
                ops += exc2
                n_t2 += 1

    return ops


def contraction(ops):
    result = []
    cre = [(i, op) for i, op in enumerate(ops) if op[1] == "+"]
    ana = [(i, op) for i, op in enumerate(ops) if op[1] == "-"]
    t_type = [op[3] for op in ops if op[3] >= 2]
    for anop in permutations(ana):
        pairing = list(zip(cre, anop))
        ok = True
        for (i, a), (j, b) in pairing:
            if a[3] != b[3] and i < j and a[2] in ("o", "g") and b[2] in ("o", "g"):
                continue
            elif a[3] != b[3] and i > j and a[2] in ("v", "g") and b[2] in ("v", "g"):
                continue
            else:
                ok = False
                break
        if ok:
            linked = []
            for pair in pairing:
                H = [(j, op) for j, op in pair if op[3] == 1]
                T = [(j, op) for j, op in pair if op[3] >= 2]
                if H and T:
                    linked.append(T[0][1][3])

            if set(linked) == set(t_type):
                result.append([tuple(sorted((i, j))) for (i, a), (j, b) in pairing])
    return result


def cross_counting(pairs):
    n = 0
    sign = 1
    for (i, j), (k, l) in combinations(pairs, 2):
        if i < k < j < l or k < i < l < j:
            n += 1
    if n % 2 == 1:
        sign *= -1
    return sign


def rename_antisym(ops, pairs):
    rename = {}
    for i, j in pairs:
        x, y = ops[i], ops[j]
        if x[3] == 0:
            rename[y[0]] = x[0]
        elif y[3] == 0:
            rename[x[0]] = y[0]
        elif x[2] == "g":
            rename[x[0]] = y[0]
        elif y[2] == "g":
            rename[y[0]] = x[0]
    return rename


def get_factor_and_c(ops):
    factor = []
    cre_h = [op[0] for op in ops if op[1] == "+" and op[3] == 1]
    ana_h = [op[0] for op in ops if op[1] == "-" and op[3] == 1]

    if len(cre_h) == 2:
        name = "g"
    else:
        name = "f"
    c_H = factorial(len(cre_h)) ** 2
    idx = tuple(cre_h + ana_h[::-1])
    factor.append((name, idx))

    coe_H = Fraction(1, (c_H))

    coet_list = []
    for i in range(2, 7):
        cre_t = [op[0] for op in ops if op[1] == "+" and op[3] == i]
        ana_t = [op[0] for op in ops if op[1] == "-" and op[3] == i]

        if len(cre_t) == 0:
            continue

        if len(cre_t) == 2:
            name = "t_2"
        if len(cre_t) == 1:
            name = "t_1"
        c_t = factorial(len(cre_t)) ** 2
        idx = tuple(ana_t[::-1] + cre_t)
        factor.append((name, idx))

        coe_t = Fraction(1, (c_t))
        coet_list.append(coe_t)

    coe = coe_H * math.prod(coet_list)
    T2_n = 0
    T1_n = 0
    for j in coet_list:
        if j == Fraction(1, 4):
            T2_n += 1
        if j == Fraction(1, 1):
            T1_n += 1
    if T2_n != 0:
        coe *= Fraction(1, (factorial(T2_n)))
    if T1_n != 0:
        coe *= Fraction(1, (factorial(T1_n)))

    return factor, coe


def get_terms(ops):
    formula, coe = get_factor_and_c(ops)
    result = contraction(ops)
    terms = []
    for pairs in result:
        sign = cross_counting(pairs)
        rename = rename_antisym(ops, pairs)
        new = []
        for name, idx in formula:
            new.append((name, tuple(rename.get(ch, ch) for ch in idx)))
        terms.append((coe * sign, new))
    return terms


def get_standard(term):
    c, factors = term
    std = []
    for name, idx in factors:
        if len(idx) == 2:
            std.append((name, idx))
        else:
            h1, h2, h3, h4 = idx[:4]
            if h1 > h2:
                h1, h2 = h2, h1
                c *= -1
            if h3 > h4:
                h3, h4 = h4, h3
                c *= -1
            idx = (h1, h2, h3, h4)
            std.append((name, idx))
    return (c, tuple(std))


def merge_antisym(terms):
    f_terms = defaultdict(Fraction)
    for term in terms:
        std = get_standard(term)
        f_terms[std[1]] += std[0]
    return f_terms


def rename_dummy(factor, ops):
    n_bra = 0
    for op in ops:
        if op[3] == 0:
            n_bra += 1

    if n_bra == 6:
        o_list = "lmn"
        v_list = "def"
    else:
        o_list = "klmn"
        v_list = "cdef"

    rename_dummy_table = {}
    o_used = 0
    v_used = 0
    for name, idx in factor:
        for ch in idx:
            if ch in o_list and ch not in rename_dummy_table:
                rename_dummy_table[ch] = o_list[o_used]
                o_used += 1
            if ch in v_list and ch not in rename_dummy_table:
                rename_dummy_table[ch] = v_list[v_used]
                v_used += 1
    return rename_dummy_table


def merge_dummy(f_terms, ops):
    f_terms2 = defaultdict(Fraction)
    for factor, c in f_terms.items():
        rename_table = rename_dummy(factor, ops)
        term2 = []
        for name, idx in factor:
            term2.append((name, tuple(rename_table.get(ch, ch) for ch in idx)))
        std = get_standard((c, term2))
        f_terms2[std[1]] += std[0]
    return f_terms2


def merge_t_order(f_terms2, ops):
    rename_t = {}
    f_terms3 = defaultdict(Fraction)
    for factor, c in f_terms2.items():
        t_list = []
        for name, idx in factor:
            if name != "f" and name != "g":
                for ch in idx:
                    if ch in "klmn":
                        rename_t[ch] = "o"
                    if ch in "cdef":
                        rename_t[ch] = "v"
                new_shape = tuple(rename_t.get(ch, ch) for ch in idx)
                t_list.append((new_shape, (name, idx)))
        t_list.sort()
        ordered = [t for shape, t in t_list] + [factor[0]]

        rename_table = rename_dummy(ordered, ops)
        term3 = []
        for name, idx in ordered:
            term3.append((name, tuple(rename_table.get(ch, ch) for ch in idx)))
        std = get_standard((c, term3))
        f_terms3[std[1]] += std[0]
    return f_terms3


def generate_einsum(f_terms, ops):
    space_of = {}
    lines = []

    for op in ops:
        space_of[op[0]] = op[2]
    bra_cre = [op[0] for op in ops if op[1] == "+" and op[3] == 0]
    bra_ana = [op[0] for op in ops if op[1] == "-" and op[3] == 0]
    out = "".join(bra_cre + bra_ana[::-1])

    for factors, c in f_terms.items():
        subs = []
        arrs = []
        for name, idx in factors:
            sub = "".join(idx)
            space = ", ".join(space_of[ch] for ch in idx)

            if name == "t_1":
                arr = "t1"
            if name == "t_2":
                arr = "t2"
            if name == "g" or name == "f":
                arr = f"{name}[{space}]"
            subs.append(sub)
            arrs.append(arr)

        subs_str = ",".join(subs)
        arrs_str = ", ".join(arrs)
        lines.append(f"{float(c)} * np.einsum('{subs_str}->{out}', {arrs_str}, optimize=True)")
    return lines


def get_residual(equations):
    lines = []
    for ops in equations:
        terms = get_terms(ops)
        f_terms2 = merge_dummy(merge_antisym(terms), ops)
        f_terms3 = merge_t_order(f_terms2, ops)
        lines += generate_einsum(f_terms3, ops)
    return " + ".join(lines)
