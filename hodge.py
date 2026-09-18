"""
hodge.py

Hodge number computation: h^{p,q}(X) = dim H^q(X, Omega^p_X).

For a cominuscule homogeneous variety, Omega^p_X is literally computable
(cotangent_bundle, wedge powers, BWB). For projective bundles the cotangent
sequence need not split, and the relative sequences must also be considered.
For a non-cominuscule G/P its cotangent bundle is filtered; its Levi associated
graded gives the exact K-class and hence exact Euler characteristics, while
individual cohomology groups retain the uncertainty of the filtration spectral
sequence. For a ZeroLocus
X = Z(s) ⊂ ambient (s a section of E, rank r = codim X), Omega^p_X is NOT a
literal bundle we can construct symbolically — the conormal sequence

    0 → N* → Omega^1_ambient|_X → Omega^1_X → 0       (N* = E*|_X)

is generally non-split. Instead we resolve Omega^p_X by the classical Koszul
complex for exterior powers of a quotient bundle (terms built from SYMMETRIC
powers of the sub N*, and EXTERIOR powers of the middle term):

    0 → Sym^p(N*) → Sym^{p-1}(N*)⊗Omega^1_M|_X → ... → N*⊗Omega^{p-1}_M|_X
      → Omega^p_M|_X → Omega^p_X → 0

(checked against the tautological sequence on G(2,4): ranks 3,8,6,1 for p=2,
alternating sum 0, and the p=1 case reduces to the conormal sequence itself.)

Every term Sym^{p-i}(N*) ⊗ Omega^i_M is the restriction to X of an AMBIENT
bundle, computable via the existing Koszul/BundleOnZeroLocus machinery. We
peel H^*(X, Omega^p_X) off the end of this (p+1)-term exact resolution by
breaking it into p short exact sequences via syzygy sheaves.

The Hodge-number path keeps integer lower and upper bounds for actual
cohomology. Koszul and homogeneous-filtration E1 dimensions are upper bounds,
not final dimensions: differentials may cancel classes in consecutive total
degrees. Grothendieck vanishing forces cohomology outside [0, dim(X)] to zero,
but those E1 terms still contribute to the exact Euler characteristic.

Bounds propagate through the conormal and relative long exact sequences.
Across the Hodge diamond we identify variables related by
  - Hodge symmetry: h^{p,q} = h^{q,p};
  - Serre duality: h^{p,q} = h^{d-p,d-q};
and impose hard Lefschetz injectivity
  h^{p,q} <= h^{p+1,q+1} for p+q < d,
together with all exact equations
  chi(Omega^p_X) = sum_q (-1)^q h^{p,q}.
Exact rational row reduction and integer interval propagation can determine
several unknowns together, including vanishing forced by a symmetry partner.
Only singleton bounds are returned as Hodge numbers. If uncertainty remains,
hodge_numbers raises ValueError with the unresolved ranges. This procedure
is sufficient, not complete: it need not detect every forced map rank.

These statements assume a nonempty smooth regular zero locus in the supported
complex projective ambient. They do not require E1 degeneration. The older
zero_locus_omega_partial/solve_*_dims helpers retain their heuristic behavior
for compatibility and are not used to certify the Hodge diamond.
"""

from __future__ import annotations
from fractions import Fraction
from math import ceil, floor
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# 1. Dimension-dictionary utilities
# ============================================================

def dim_dict(cohomology_list: list) -> Dict[int, int]:
    """Collapse a .cohomology-style list of dicts into {degree: total_dim}."""
    result: Dict[int, int] = {}
    for entry in cohomology_list:
        d = entry['degree']
        result[d] = result.get(d, 0) + entry['multiplicity'] * entry['dimension']
    return {d: v for d, v in result.items() if v != 0}


def euler_char(dims: Dict[int, int]) -> int:
    # Negative total degrees occur on a Koszul E1 page. Python's (-1)**q
    # becomes a float when q < 0, losing exactness for large dimensions.
    return sum((1 if q % 2 == 0 else -1) * v for q, v in dims.items())


# Bounds always refer to actual cohomology, never to a chosen differential.
_Bounds = Dict[int, Tuple[int, int]]


def _tighten_integer_bounds(bounds: dict, equations: list) -> None:
    """Intersect integer intervals with exact linear equations, in place.

    Each equation is (coefficient_dict, rhs). Rational row reduction also
    exposes consequences involving several equations; interval propagation
    then uses nonnegativity and integrality, even in underdetermined systems.
    This is a sufficient constraint solver, not a complete integer solver.
    """
    keys = list(bounds)
    rows = [[Fraction(coeffs.get(key, 0)) for key in keys] + [Fraction(rhs)]
            for coeffs, rhs in equations]
    pivot = 0
    for column in range(len(keys)):
        row = next((i for i in range(pivot, len(rows)) if rows[i][column]), None)
        if row is None:
            continue
        rows[pivot], rows[row] = rows[row], rows[pivot]
        scale = rows[pivot][column]
        rows[pivot] = [value / scale for value in rows[pivot]]
        for i in range(len(rows)):
            if i != pivot and rows[i][column]:
                scale = rows[i][column]
                rows[i] = [a - scale * b for a, b in zip(rows[i], rows[pivot])]
        pivot += 1
    constraints = list(equations) + [
        ({key: row[i] for i, key in enumerate(keys) if row[i]}, row[-1])
        for row in rows
    ]
    if any(lo < 0 or hi < lo for lo, hi in bounds.values()):
        raise ValueError("Inconsistent nonnegative cohomology bounds")
    progress = True
    while progress:
        progress = False
        for coeffs, rhs in constraints:
            contributions = {
                key: (min(a * bounds[key][0], a * bounds[key][1]),
                      max(a * bounds[key][0], a * bounds[key][1]))
                for key, a in coeffs.items() if a
            }
            total_lo = sum(lo for lo, _ in contributions.values())
            total_hi = sum(hi for _, hi in contributions.values())
            if not total_lo <= rhs <= total_hi:
                raise ValueError("Cohomology bounds contradict exact Euler/symmetry constraints")
            for key, (term_lo, term_hi) in contributions.items():
                a = Fraction(coeffs[key])
                ends = ((rhs - total_hi + term_hi) / a,
                        (rhs - total_lo + term_lo) / a)
                lo, hi = bounds[key]
                narrowed = (max(lo, ceil(min(ends))), min(hi, floor(max(ends))))
                if narrowed[0] > narrowed[1]:
                    raise ValueError("No integral cohomology dimension satisfies the constraints")
                if narrowed != bounds[key]:
                    bounds[key] = narrowed
                    progress = True


def _tighten_cohomology_bounds(bounds: _Bounds, chi: int) -> _Bounds:
    _tighten_integer_bounds(bounds, [({q: 1 if q % 2 == 0 else -1 for q in bounds}, chi)])
    return bounds


def _cohomology_bounds_from_e1(dims: Dict[int, int], max_degree: int) -> _Bounds:
    """Bound the abutment of Koszul/filtration spectral sequences.

    Every differential raises total degree by one. Thus the loss in degree q
    is at most the original dimensions in degrees q-1 and q+1, and an E1
    zero is a genuine vanishing. Retain ALL total degrees for chi, including
    those outside [0, dim X] whose eventual cohomology must vanish.
    See https://stacks.math.columbia.edu/tag/012K for the degree convention.
    """
    if any(value < 0 for value in dims.values()):
        raise ValueError("E1 dimensions must be nonnegative")
    bounds = {
        q: (max(0, dims.get(q, 0) - dims.get(q - 1, 0) - dims.get(q + 1, 0)),
            dims.get(q, 0))
        for q in range(max_degree + 1)
    }
    return _tighten_cohomology_bounds(bounds, euler_char(dims))


def _ses_cohomology_bounds(left, right, missing: str):
    """Bounds from 0 -> A -> B -> C -> 0, including shifted LES terms.

    Arguments are the two known (bounds, chi) pairs in A,B,C order; missing
    is 'sub', 'mid' or 'quot'. For example C_q <= B_q + A_{q+1}, whereas
    C_q >= max(0, B_q-A_q, A_{q+1}-B_{q+1}). No map rank is guessed.
    """
    first, chi_first = left
    second, chi_second = right

    def lo(b, q):
        return b.get(q, (0, 0))[0]

    def hi(b, q):
        return b.get(q, (0, 0))[1]

    bounds = {}
    for q in first:
        if missing == 'quot':
            a, b = first, second
            lower = max(0, lo(b, q) - hi(a, q), lo(a, q + 1) - hi(b, q + 1))
            upper = hi(b, q) + hi(a, q + 1)
        elif missing == 'sub':
            b, c = first, second
            lower = max(0, lo(b, q) - hi(c, q), lo(c, q - 1) - hi(b, q - 1))
            upper = hi(b, q) + hi(c, q - 1)
        elif missing == 'mid':
            a, c = first, second
            lower = max(0, lo(a, q) - hi(c, q - 1), lo(c, q) - hi(a, q + 1))
            upper = hi(a, q) + hi(c, q)
        else:
            raise ValueError(f"Unknown SES position: {missing}")
        bounds[q] = (lower, upper)
    chi = (chi_second - chi_first if missing == 'quot' else
           chi_first - chi_second if missing == 'sub' else chi_first + chi_second)
    return _tighten_cohomology_bounds(bounds, chi), chi


def solve_quotient_dims(
    dims_sub: Dict[int, int], guess_sub: Set[int],
    dims_mid: Dict[int, int], max_degree: int,
    log: List[dict], context: str, p: int,
    guess_mid: Optional[Set[int]] = None,
) -> Tuple[Dict[int, int], Set[int]]:
    """
    Given a short exact sequence 0 → A → B → C → 0 on a variety of dimension
    max_degree, and the cohomology of A (sub — `dims_sub`/`guess_sub`) and B
    (middle — `dims_mid`/`guess_mid`, defaulting to fully rigorous), with
    guess-sets marking which degrees are themselves only guesses (not
    rigorous), determine C's (quotient) dictionary. Returns
    (dims_quot, guess_quot) — dims_quot has a value for every degree in
    [0, max_degree], and guess_quot marks which of those are guesses.

    A degree that is a guess in A or B must not be treated as rigorously
    known when deciding whether C's value at that degree is rigorous — that
    would silently launder uncertainty into fact. If that's the only
    obstruction we fall back to the naive guess (assume the connecting map
    vanishes) and log the ambiguous LES — together with the cohomology of A
    and B at this step — to `log` for the user to inspect.

    The long exact sequence is:
        ... → H^q(A) → H^q(B) → H^q(C) → H^{q+1}(A) → H^{q+1}(B) → ...
    By Grothendieck vanishing, H^q(-) = 0 outside [0, max_degree] for every
    sheaf on X, so those degrees are never ambiguous.
    """
    guess_mid = guess_mid or set()

    def a_rigorously_zero(q: int) -> bool:
        if q < 0 or q > max_degree:
            return True  # Grothendieck vanishing
        if q in guess_sub:
            return False
        return dims_sub.get(q, 0) == 0

    def a_val(q: int) -> int:
        if q < 0 or q > max_degree:
            return 0
        return dims_sub.get(q, 0)

    def b(q: int) -> int:
        return dims_mid.get(q, 0) if 0 <= q <= max_degree else 0

    def b_rigorous(q: int) -> bool:
        return 0 <= q <= max_degree and q not in guess_mid

    dims_quot: Dict[int, int] = {}
    guess_quot: Set[int] = set()
    for q in range(0, max_degree + 1):
        rigorous = a_rigorously_zero(q + 1) and a_rigorously_zero(q) and b_rigorous(q)
        val = b(q) - a_val(q)
        if not rigorous:
            guess_quot.add(q)
            log.append({
                'context': context,
                'p': p,
                'degree': q,
                'dims_sub': dict(dims_sub), 'guess_sub': set(guess_sub),
                'dims_mid': dict(dims_mid),
                'guess_value': val,
            })
        dims_quot[q] = val
    return dims_quot, guess_quot


def solve_sub_dims(
    dims_mid: Dict[int, int], guess_mid: Set[int],
    dims_quot: Dict[int, int], guess_quot: Set[int],
    max_degree: int, log: List[dict], context: str, p: int,
) -> Tuple[Dict[int, int], Set[int]]:
    """
    Dual of solve_quotient_dims: given a SES 0 → A → B → C → 0 with B (middle)
    and C (quotient) known (each possibly only partially — guess sets as
    elsewhere), determine A (sub).

    Two independent sufficient conditions pin down dim H^q(A) rigorously:
      (i)  the connecting maps H^{q-1}(C)->H^q(A) and H^q(C)->H^{q+1}(A) are
           forced to vanish whenever their SOURCE (in H^*(C)) is rigorously
           0 — giving dim H^q(A) = dim H^q(B) - dim H^q(C);
      (ii) if H^{q-1}(B) = H^q(B) = 0 (rigorously), the maps into/out of
           those zero spaces are trivially rank 0, which forces a SHIFTED
           isomorphism dim H^q(A) = dim H^{q-1}(C) instead — this is the
           case that matters when B is entirely acyclic (e.g. a relative
           twist in the acyclic range of the projective bundle formula):
           the long exact sequence then degenerates into isomorphisms
           H^q(A) ≅ H^{q-1}(C) for every q, not just "connecting map is
           zero", and (i) alone misses this.
    """
    def c_rigorously_zero(q: int) -> bool:
        if q < 0 or q > max_degree:
            return True
        if q in guess_quot:
            return False
        return dims_quot.get(q, 0) == 0

    def g_rigorously_zero(q: int) -> bool:
        if q < 0 or q > max_degree:
            return True
        if q in guess_mid:
            return False
        return dims_mid.get(q, 0) == 0

    def g(q: int) -> int:
        return dims_mid.get(q, 0) if 0 <= q <= max_degree else 0

    def h(q: int) -> int:
        return dims_quot.get(q, 0) if 0 <= q <= max_degree else 0

    def mid_rigorous(q: int) -> bool:
        return 0 <= q <= max_degree and q not in guess_mid

    def quot_rigorous(q: int) -> bool:
        return q < 0 or (0 <= q <= max_degree and q not in guess_quot)

    dims_sub: Dict[int, int] = {}
    guess_sub: Set[int] = set()
    for q in range(0, max_degree + 1):
        via_quot_zero = c_rigorously_zero(q - 1) and c_rigorously_zero(q) and mid_rigorous(q)
        via_mid_zero = g_rigorously_zero(q - 1) and g_rigorously_zero(q) and quot_rigorous(q - 1)
        if via_quot_zero:
            val = g(q) - h(q)
            rigorous = True
        elif via_mid_zero:
            val = h(q - 1)
            rigorous = True
        else:
            val = g(q) - h(q)
            rigorous = False
        if not rigorous:
            guess_sub.add(q)
            log.append({
                'context': context, 'p': p, 'degree': q,
                'dims_sub': dict(dims_quot), 'guess_sub': set(guess_quot),
                'dims_mid': dict(dims_mid),
                'guess_value': val,
            })
        dims_sub[q] = val
    return dims_sub, guess_sub


def solve_middle_dims(
    dims_sub: Dict[int, int], guess_sub: Set[int],
    dims_quot: Dict[int, int], guess_quot: Set[int],
    max_degree: int, log: List[dict], context: str, p: int,
) -> Tuple[Dict[int, int], Set[int]]:
    """
    Given a SES 0 → A → B → C → 0 with A (sub) and C (quotient) known
    (each possibly only partially), determine B (middle).

    dim H^q(B) = dim H^q(A) + dim H^q(C), rigorously, whenever the two
    connecting maps touching H^q(A) are forced to vanish. Since BOTH A and
    C are known here (unlike solve_quotient_dims/solve_sub_dims, where the
    unknown term is itself one end of the map), a connecting map
        delta: H^{q'}(C) → H^{q'+1}(A)
    is forced to vanish if EITHER its source H^{q'}(C) OR its target
    H^{q'+1}(A) is rigorously known to be 0 — not just the target.
    """
    def a_rigorously_zero(q: int) -> bool:
        if q < 0 or q > max_degree:
            return True
        if q in guess_sub:
            return False
        return dims_sub.get(q, 0) == 0

    def c_rigorously_zero(q: int) -> bool:
        if q < 0 or q > max_degree:
            return True
        if q in guess_quot:
            return False
        return dims_quot.get(q, 0) == 0

    def f(q: int) -> int:
        return dims_sub.get(q, 0) if 0 <= q <= max_degree else 0

    def h(q: int) -> int:
        return dims_quot.get(q, 0) if 0 <= q <= max_degree else 0

    def sub_rigorous(q: int) -> bool:
        return 0 <= q <= max_degree and q not in guess_sub

    def quot_rigorous(q: int) -> bool:
        return 0 <= q <= max_degree and q not in guess_quot

    dims_mid: Dict[int, int] = {}
    guess_mid: Set[int] = set()
    for q in range(0, max_degree + 1):
        delta_qm1_zero = c_rigorously_zero(q - 1) or a_rigorously_zero(q)
        delta_q_zero = c_rigorously_zero(q) or a_rigorously_zero(q + 1)
        rigorous = delta_qm1_zero and delta_q_zero and sub_rigorous(q) and quot_rigorous(q)
        val = f(q) + h(q)
        if not rigorous:
            guess_mid.add(q)
            log.append({
                'context': context, 'p': p, 'degree': q,
                'dims_sub': dict(dims_sub), 'guess_sub': set(guess_sub),
                'dims_mid': dict(dims_quot),
                'guess_value': val,
            })
        dims_mid[q] = val
    return dims_mid, guess_mid


# ============================================================
# 2. H^*(X, multiplier ⊗ Omega^k_M |_X) for an arbitrary tower variety M
# ============================================================
#
# Omega^k_M is, in general, not a literal bundle: for M a Grassmannian it
# is (cotangent_bundle.wedge_power(k)), but for M = ProjectiveBundle(base, F)
# it sits in two non-split extensions:
#
#   (relative)  0 → Omega^1_rel → pi*(F*)⊗O(-1) → O → 0      (dual Euler seq)
#   (absolute)  0 → pi*(Omega^1_base) → Omega^1_M → Omega^1_rel → 0
#
# Tensoring is exact, so we never need Omega^k_M itself as a bundle — only
# the EXACT RESOLUTIONS it sits in, with every term tensored by `multiplier`
# (e.g. Sym^{p-i}(N*) from the outer conormal resolution) BEFORE restricting
# to X. This section builds those resolutions and peels them with the same
# rigor/guess tracking as section 1.
#
# (1) Omega^j_rel resolution, via the dual Euler sequence with quotient O
#     (rank 1): writing B = pi*(F*)⊗O(-1),
#         0 → Omega^j_rel → ∧^j(B) → ∧^{j-1}(B) → ... → B → O → 0
#     (checked: j=1 is the Euler sequence itself; rank-alternating-sum is 0
#     for j=2 at any rank — this is the dual of the "Koszul complex for
#     exterior powers of a quotient" used for the conormal resolution, with
#     quotient = trivial line bundle). Omega^j_rel is the SUB (leftmost) of
#     this complex, so it's peeled via solve_sub_dims, working from the
#     known tail (O) backward.
#
# (2) Omega^k_M, via the filtration of the absolute/relative SES: graded
#     pieces pi*(Omega^j_base) ⊗ Omega^{k-j}_rel for j = 0,...,k (zero
#     whenever j > dim(base) or k-j > rank(F)-1). Built up via
#     solve_middle_dims, starting from the deepest piece
#     pi*(Omega^k_base) (j=k, known once base's own Omega^k_base is known —
#     recursively, via this same machinery if base is itself a tower) and
#     working outward to Omega^k_M = the full middle term.


def _homogeneous_cotangent_power(variety, degree: int):
    """Return ``(Lambda^degree Omega, is_filtered_model)`` for a G/P.

    On a cominuscule variety the returned bundle is genuine. On a general
    parabolic it is the Levi associated graded. Exterior powers commute with
    passing to the associated graded in K-theory, so this model is exact for
    Euler characteristics but not necessarily for individual cohomology groups.
    """
    if degree < 0 or degree > variety.dimension:
        return None, False
    try:
        cotangent = variety.cotangent_bundle
        filtered = False
    except NotImplementedError:
        cotangent = variety.cotangent_associated_graded
        filtered = True
    return cotangent.wedge_power(degree), filtered and degree > 0


def _mark_filtered_guesses(
    dims: Dict[int, int], guesses: Set[int], max_degree: int,
    log: List[dict], context: str, p: int,
) -> Tuple[Dict[int, int], Set[int]]:
    """Taint E1-page dimensions whose filtration differentials are unknown."""
    result = set(guesses)
    result.update(range(max_degree + 1))
    log.append({
        'kind': 'filtered',
        'context': context,
        'p': p,
        'dims_mid': dict(dims),
    })
    return dims, result

def _relative_B(M):
    """B = pi*(F*) ⊗ O(-1) on M = ProjectiveBundle(base, F)."""
    from varieties import RelativeTwist
    L = RelativeTwist(M)
    return M.F.dual * (L ** -1)


def _omega_rel_dims(M, j: int, multiplier, X, d: int, log: List[dict]) -> Tuple[Dict[int, int], Set[int]]:
    """H^*(X, multiplier ⊗ Omega^j_rel |_X) via the dual-Euler-sequence resolution."""
    if j == 0:
        bundle = multiplier
        return dim_dict(bundle.restrict_support(X).cohomology), set()

    B = _relative_B(M)
    # Term i (1-indexed, i=1,...,j+1) = multiplier ⊗ ∧^{j+1-i}(B);
    # X_1 = multiplier⊗∧^j(B), ..., X_{j+1} = multiplier⊗∧^0(B) = multiplier.
    X_terms = [dim_dict((multiplier * B.wedge_power(j + 1 - i)).restrict_support(X).cohomology)
               for i in range(1, j + 2)]

    Z, guess_Z = X_terms[-1], set()  # Z_{j} = X_{j+1} = multiplier (rigorous, direct Koszul cohomology)
    for i in range(j, 0, -1):
        context = (
            f"Omega^{j}_rel resolution (M=ProjectiveBundle), step {j - i + 1}: "
            f"0 → Z_{i-1} → multiplier⊗∧^{j+1-i}(B)|_X → Z_{i} → 0"
            + (" → Omega^j_rel" if i == 1 else "")
        )
        Z, guess_Z = solve_sub_dims(X_terms[i - 1], set(), Z, guess_Z, d, log, context, j)
    return Z, guess_Z


def _omega_rel_chi(M, j: int, multiplier, X) -> int:
    """chi(X, multiplier ⊗ Omega^j_rel|_X), exact via the alternating sum."""
    if j == 0:
        return euler_char(dim_dict(multiplier.restrict_support(X).cohomology))
    B = _relative_B(M)
    total = 0
    for l in range(j + 1):
        term = euler_char(dim_dict((multiplier * B.wedge_power(j - l)).restrict_support(X).cohomology))
        total += (-1) ** l * term
    return total


def _product_omega_pieces(M, k: int):
    """
    The graded pieces of Omega^k of a ProductVariety M = X_1 × … × X_m:
        Omega^k_M = ⊕_{k_1+…+k_m = k}  Omega^{k_1}_{X_1} ⊠ … ⊠ Omega^{k_m}_{X_m}.
    Yields ``(bundle, filtered)`` pairs, one per admissible composition. The
    product decomposition itself is split; ``filtered`` records whether a
    non-cominuscule factor was represented by its Levi associated graded.
    """
    from grassmannians import FlagVariety
    from plethysms import compositions_of
    from varieties import _external_product

    factors = M.factors
    for comp in compositions_of(k, len(factors)):
        if any(comp[i] > factors[i].dimension for i in range(len(factors))):
            continue
        pieces = []
        filtered = False
        for i, ki in enumerate(comp):
            Xi = factors[i]
            if not isinstance(Xi, FlagVariety):
                raise NotImplementedError(
                    "Hodge numbers of a ZeroLocus inside a product are currently "
                    "supported only when every product factor is a homogeneous G/P."
                )
            piece, piece_filtered = _homogeneous_cotangent_power(Xi, ki)
            pieces.append(piece)
            filtered = filtered or piece_filtered
        yield _external_product(M, pieces), filtered


def _omega_M_tensor_dims(M, k: int, multiplier, X, d: int, log: List[dict]) -> Tuple[Dict[int, int], Set[int]]:
    """H^*(X, multiplier tensor Omega^k_M|_X) for a supported ambient."""
    from grassmannians import FlagVariety
    from varieties import ProjectiveBundle, ProductVariety

    if isinstance(M, FlagVariety):
        omega_k, filtered = _homogeneous_cotangent_power(M, k)
        bundle = multiplier * omega_k
        dims = dim_dict(bundle.restrict_support(X).cohomology)
        if filtered:
            return _mark_filtered_guesses(
                dims, set(), d, log,
                f"Levi filtration for Omega^{k}_{M} on non-cominuscule {M}", k,
            )
        return dims, set()

    if isinstance(M, ProductVariety):
        # Omega^k_M is a genuine direct sum of external products; sum the
        # cohomology of each graded piece (all rigorous — no syzygy resolution).
        total: Dict[int, int] = {}
        filtered = False
        for piece, piece_filtered in _product_omega_pieces(M, k):
            dd = dim_dict((multiplier * piece).restrict_support(X).cohomology)
            for q, v in dd.items():
                total[q] = total.get(q, 0) + v
            filtered = filtered or piece_filtered
        dims = {q: v for q, v in total.items() if v != 0}
        if filtered:
            return _mark_filtered_guesses(
                dims, set(), d, log,
                f"Levi filtration for Omega^{k} on a product with non-cominuscule factors", k,
            )
        return dims, set()

    if isinstance(M, ProjectiveBundle):
        base = M.base
        r = M.relative_rank
        base_dim = base.dimension

        j_range = [j for j in range(k + 1) if j <= base_dim and (k - j) <= r - 1]
        if not j_range:
            return {}, set()

        # Deepest piece (largest j): pi*(Omega^{j_max}_base) ⊗ Omega^{k-j_max}_rel.
        j_max = j_range[-1]
        if isinstance(base, FlagVariety):
            omega_base_jmax, base_filtered = _homogeneous_cotangent_power(base, j_max)
        else:
            omega_base_jmax, base_filtered = None, False
        if omega_base_jmax is None:
            raise NotImplementedError(
                "ZeroLocus inside a ProjectiveBundle tower more than one level deep "
                "is not yet supported. The immediate base must be a homogeneous G/P."
            )
        deepest_mult = multiplier * omega_base_jmax
        dims, guesses = _omega_rel_dims(M, k - j_max, deepest_mult, X, d, log)
        if base_filtered:
            dims, guesses = _mark_filtered_guesses(
                dims, guesses, d, log,
                f"Levi filtration for the pulled-back Omega^{j_max} of {base}", k,
            )

        for j in reversed(j_range[:-1]):
            omega_base_j, base_filtered = _homogeneous_cotangent_power(base, j)
            piece_mult = multiplier * omega_base_j
            piece_dims, piece_guesses = _omega_rel_dims(M, k - j, piece_mult, X, d, log)
            if base_filtered:
                piece_dims, piece_guesses = _mark_filtered_guesses(
                    piece_dims, piece_guesses, d, log,
                    f"Levi filtration for the pulled-back Omega^{j} of {base}", k,
                )
            context = (
                f"Omega^{k}_M filtration (M=ProjectiveBundle), combining piece j={j}: "
                f"0 → [deeper pieces] → Omega^{k}_M-partial → "
                f"pi*(Omega^{j}_base)⊗Omega^{k-j}_rel|_X → 0"
            )
            dims, guesses = solve_middle_dims(
                dims, guesses, piece_dims, piece_guesses, d, log, context, k
            )
        return dims, guesses

    raise NotImplementedError(f"_omega_M_tensor_dims not implemented for {type(M)}")


def _omega_M_tensor_chi(M, k: int, multiplier, X) -> int:
    """chi(X, multiplier ⊗ Omega^k_M|_X), exact, mirroring _omega_M_tensor_dims."""
    from grassmannians import FlagVariety
    from varieties import ProjectiveBundle, ProductVariety

    if isinstance(M, FlagVariety):
        omega_k, _ = _homogeneous_cotangent_power(M, k)
        bundle = multiplier * omega_k
        return euler_char(dim_dict(bundle.restrict_support(X).cohomology))

    if isinstance(M, ProductVariety):
        total = 0
        for piece, _ in _product_omega_pieces(M, k):
            total += euler_char(dim_dict((multiplier * piece).restrict_support(X).cohomology))
        return total

    if isinstance(M, ProjectiveBundle):
        base = M.base
        r = M.relative_rank
        base_dim = base.dimension
        total = 0
        for j in range(k + 1):
            if j > base_dim or (k - j) > r - 1:
                continue
            if not isinstance(base, FlagVariety):
                raise NotImplementedError(
                    "ZeroLocus inside a ProjectiveBundle tower more than one level deep "
                    "is not yet supported. The immediate base must be a homogeneous G/P."
                )
            omega_base_j, _ = _homogeneous_cotangent_power(base, j)
            piece_mult = multiplier * omega_base_j
            total += _omega_rel_chi(M, k - j, piece_mult, X)
        return total

    raise NotImplementedError(f"_omega_M_tensor_chi not implemented for {type(M)}")


# ============================================================
# 3. Omega^p_X cohomology for a ZeroLocus, via the conormal resolution
# ============================================================

def _resolution_term_dims(X, p: int, i: int, log: List[dict]) -> Tuple[Dict[int, int], Set[int]]:
    """T_i = Sym^{p-i}(N*) ⊗ Omega^i_M restricted to X."""
    M = X.ambient
    Ndual = X.bundle.dual
    multiplier = Ndual.sym_power(p - i)
    return _omega_M_tensor_dims(M, i, multiplier, X, X.dimension, log)


def zero_locus_omega_chi(X, p: int) -> int:
    """
    chi(Omega^p_X) = sum_q (-1)^q h^{p,q}, computed exactly (no ambiguity)
    via the alternating sum over the exact resolution:
        chi(Omega^p_X) = (-1)^p * sum_{i=0}^p (-1)^i chi(T_i)
    """
    if p == 0:
        return euler_char(dim_dict(X.trivial_bundle.cohomology))
    M = X.ambient
    Ndual = X.bundle.dual
    total = 0
    for i in range(p + 1):
        multiplier = Ndual.sym_power(p - i)
        total += (-1) ** i * _omega_M_tensor_chi(M, i, multiplier, X)
    return (-1) ** p * total


def zero_locus_omega_partial(X, p: int, log: List[dict]) -> Tuple[Dict[int, int], Set[int]]:
    """
    Legacy heuristic H^*(X, Omega^p_X) via local conormal LES calculations.
    Returns (dims, guess_degrees); ambiguous steps are appended to `log`.
    The guess set does not account for Koszul differentials, so even entries
    outside it are not certified. hodge_numbers uses interval bounds instead.
    """
    d = X.dimension
    if p == 0:
        return dim_dict(X.trivial_bundle.cohomology), set()

    dims, guesses = _resolution_term_dims(X, p, 0, log)
    for i in range(1, p + 1):
        T_i_dims, T_i_guesses = _resolution_term_dims(X, p, i, log)
        context = (
            f"Omega^{p}_X resolution, step {i}: "
            f"0 → K_{i-1} → Sym^{p-i}(N*)⊗Omega^{i}_M|_X → K_{i} → 0"
            + (f" → Omega^{p}_X" if i == p else "")
        )
        dims, guesses = solve_quotient_dims(
            dims, guesses, T_i_dims, d, log, context, p, guess_mid=T_i_guesses
        )
    return dims, guesses


def _restricted_cohomology_bounds(bundle, X):
    dims = dim_dict(bundle.restrict_support(X).cohomology)
    return _cohomology_bounds_from_e1(dims, X.dimension), euler_char(dims)


def _omega_rel_cohomology_bounds(M, j: int, multiplier, X):
    """Bound the relative cotangent resolution from its right-hand end."""
    result = _restricted_cohomology_bounds(multiplier, X)
    if j:
        B = _relative_B(M)
        for degree in range(1, j + 1):
            middle = _restricted_cohomology_bounds(multiplier * B.wedge_power(degree), X)
            result = _ses_cohomology_bounds(middle, result, 'sub')
    return result


def _omega_M_tensor_bounds(M, k: int, multiplier, X):
    """Certified bounds and exact chi for multiplier tensor Omega^k_M|_X.

    Levi associated gradeds and Koszul terms provide spectral-sequence
    bounds. Relative resolutions and cotangent filtrations provide LES
    bounds. None of these require a degeneration or splitting assumption.
    """
    from grassmannians import FlagVariety
    from varieties import ProjectiveBundle, ProductVariety

    if isinstance(M, FlagVariety):
        omega, _ = _homogeneous_cotangent_power(M, k)
        return _restricted_cohomology_bounds(multiplier * omega, X)

    if isinstance(M, ProductVariety):
        dims: Dict[int, int] = {}
        for piece, _ in _product_omega_pieces(M, k):
            for q, value in dim_dict((multiplier * piece).restrict_support(X).cohomology).items():
                dims[q] = dims.get(q, 0) + value
        return _cohomology_bounds_from_e1(dims, X.dimension), euler_char(dims)

    if isinstance(M, ProjectiveBundle):
        base = M.base
        if not isinstance(base, FlagVariety):
            raise NotImplementedError(
                "ZeroLocus inside a ProjectiveBundle tower more than one level deep "
                "is not yet supported. The immediate base must be a homogeneous G/P."
            )
        result = ({q: (0, 0) for q in range(X.dimension + 1)}, 0)
        for j in reversed(range(k + 1)):
            if j > base.dimension or k - j > M.relative_rank - 1:
                continue
            omega, _ = _homogeneous_cotangent_power(base, j)
            piece = _omega_rel_cohomology_bounds(M, k - j, multiplier * omega, X)
            result = _ses_cohomology_bounds(result, piece, 'mid')
        return result

    raise NotImplementedError(f"_omega_M_tensor_bounds not implemented for {type(M)}")


def _zero_locus_omega_bounds(X, p: int):
    """Propagate bounds through the conormal resolution, retaining exact chi."""
    if p == 0:
        dims = dim_dict(X.trivial_bundle.cohomology)
        return _cohomology_bounds_from_e1(dims, X.dimension), euler_char(dims)
    Ndual = X.bundle.dual
    result = _omega_M_tensor_bounds(X.ambient, 0, Ndual.sym_power(p), X)
    for i in range(1, p + 1):
        middle = _omega_M_tensor_bounds(X.ambient, i, Ndual.sym_power(p - i), X)
        result = _ses_cohomology_bounds(result, middle, 'quot')
    return result


def _tighten_hodge_bounds(bounds: dict, chi_p: dict, d: int) -> None:
    """Combine Hodge/Serre orbits, Euler rows and hard Lefschetz inequalities.

    In particular, a proved h^{q,p}=0 forces h^{p,q}=0 even when its local
    LES was ambiguous. Identifying symmetry partners before row reduction
    also retains relations between multiple unresolved entries in a row.

    Cup product with an ample class injects H^{p,q} into H^{p+1,q+1}
    when p+q < d. In particular h^{p,p} >= h^{0,0} on every diagonal,
    using Serre duality above the middle. This is intrinsic hard Lefschetz,
    not the optional weak Lefschetz hypothesis relating X to its ambient.
    """
    def orbit(p, q):
        return min((p, q), (q, p), (d - p, d - q), (d - q, d - p))

    shared = {}
    for (p, q), (lo, hi) in bounds.items():
        key = orbit(p, q)
        old_lo, old_hi = shared.get(key, (lo, hi))
        shared[key] = (max(lo, old_lo), min(hi, old_hi))
    equations = []
    for p in range(d + 1):
        coeffs = {}
        for q in range(d + 1):
            key = orbit(p, q)
            coeffs[key] = coeffs.get(key, 0) + (1 if q % 2 == 0 else -1)
        equations.append(({key: a for key, a in coeffs.items() if a}, chi_p[p]))
    injections = sorted({
        (orbit(p, q), orbit(p + 1, q + 1))
        for p in range(d) for q in range(d)
        if p + q < d and orbit(p, q) != orbit(p + 1, q + 1)
    })
    while True:
        previous = dict(shared)
        _tighten_integer_bounds(shared, equations)
        for source, target in injections:
            source_lo, source_hi = shared[source]
            target_lo, target_hi = shared[target]
            if source_lo > target_hi:
                raise ValueError("Cohomology bounds contradict hard Lefschetz injectivity")
            # source <= target transfers lower bounds forwards and upper
            # bounds backwards. Revisit Euler rows after any improvement.
            shared[source] = (source_lo, min(source_hi, target_hi))
            shared[target] = (max(target_lo, source_lo), target_hi)
        if shared == previous:
            break
    for p, q in bounds:
        bounds[(p, q)] = shared[orbit(p, q)]


def _obvious_ample_complete_intersection(X) -> bool:
    """Recognize sums of ample homogeneous line bundles on a bare G/P."""
    from grassmannians import FlagVariety, HomogeneousIrreducible

    if X.dimension <= 0 or not isinstance(X.ambient, FlagVariety):
        return False
    for monomial in X.bundle._monomials:
        if monomial.shift or monomial.multiplicity <= 0 or len(monomial.factors) != 1:
            return False
        generator = monomial.factors[0]
        if not isinstance(generator, HomogeneousIrreducible) or generator.rank != 1:
            return False
        if any(generator.weight[node - 1] <= 0 for node in X.ambient.crossed_nodes):
            return False
    return bool(X.bundle._monomials)


def _lefschetz_zero_locus_hodge_numbers(X, verbose: bool) -> Dict[Tuple[int, int], int]:
    """Reconstruct a Hodge diamond from weak Lefschetz and exact chi(Omega^p).

    This applies when restriction from the ambient is an isomorphism in total
    cohomological degrees below ``dim(X)`` (for example, a smooth complete
    intersection of ample divisors, or more generally a zero locus covered by
    the appropriate Lefschetz theorem for ample vector bundles).
    """
    d = X.dimension
    ambient_hodge = hodge_numbers(X.ambient, verbose=verbose)
    values: Dict[Tuple[int, int], int] = {}
    chi_p = {p: zero_locus_omega_chi(X, p) for p in range(d + 1)}

    for p in range(d + 1):
        for q in range(d + 1):
            if p + q < d:
                values[(p, q)] = ambient_hodge.get((p, q), 0)
            elif p + q > d:
                values[(p, q)] = ambient_hodge.get((d - p, d - q), 0)

    for p in range(d + 1):
        middle_q = d - p
        known = sum(
            (-1) ** q * values[(p, q)]
            for q in range(d + 1)
            if q != middle_q
        )
        middle = (-1) ** middle_q * (chi_p[p] - known)
        if middle < 0:
            raise ValueError(
                "Lefschetz reconstruction produced a negative Hodge number at "
                f"({p}, {middle_q}); check the positivity/Lefschetz assumption."
            )
        values[(p, middle_q)] = middle

    for p in range(d + 1):
        for q in range(d + 1):
            value = values[(p, q)]
            if value != values[(q, p)] or value != values[(d - p, d - q)]:
                raise ValueError(
                    "Lefschetz reconstruction violates Hodge symmetry or Serre duality; "
                    "check the positivity/Lefschetz assumption."
                )
        expected = chi_p[p]
        actual = sum((-1) ** q * values[(p, q)] for q in range(d + 1))
        if actual != expected:
            raise ArithmeticError("Lefschetz Hodge reconstruction failed its Euler check")

    return {pq: value for pq, value in values.items() if value}


# ============================================================
# 4. Hodge diamond, certified by duality/symmetry/Euler constraints
# ============================================================

def _format_les_log(entry: dict) -> str:
    if entry.get('kind') == 'filtered':
        return (
            f"  {entry['context']}\n"
            f"    associated-graded E1 dimensions = {entry['dims_mid']}\n"
            "    => individual dimensions depend on the homogeneous-filtration differential"
        )
    lines = [f"  {entry['context']}"]
    lines.append(f"    H^*(K_sub)  = {entry['dims_sub']}  (guessed at degrees {sorted(entry['guess_sub'])})")
    lines.append(f"    H^*(T_mid)  = {entry['dims_mid']}")
    lines.append(
        f"    => ambiguous at degree q={entry['degree']}; "
        f"guessing dim H^q(quotient) = {entry['guess_value']} (assuming the connecting map vanishes)"
    )
    return "\n".join(lines)


def _zero_locus_hodge_numbers(X, verbose: bool) -> Dict[Tuple[int, int], int]:
    d = X.dimension
    if d < 0:
        raise ValueError("Hodge numbers require a nonempty smooth zero locus of nonnegative dimension")
    bounds = {}
    chi_p = {}
    for p in range(d + 1):
        row, chi_p[p] = _zero_locus_omega_bounds(X, p)
        bounds.update({(p, q): interval for q, interval in row.items()})

    # Constants survive on a nonempty X. Do not assume X is connected:
    # zero-dimensional or disconnected loci can have h^{0,0} > 1.
    lo, hi = bounds[(0, 0)]
    bounds[(0, 0)] = (max(1, lo), hi)
    initially_unknown = sum(lo != hi for lo, hi in bounds.values())
    _tighten_hodge_bounds(bounds, chi_p, d)
    unresolved = {pq: interval for pq, interval in bounds.items() if interval[0] != interval[1]}
    if unresolved:
        detail = "; ".join(f"h^{pq} in [{lo}, {hi}]" for pq, (lo, hi) in unresolved.items())
        raise ValueError(
            "Hodge numbers could not be determined uniquely from the cohomology bounds, "
            "Hodge symmetry, Serre duality, hard Lefschetz, and exact Euler characteristics. "
            f"Unresolved bounds: {detail}. Additional information about the differentials "
            "or a justified Lefschetz hypothesis is needed; zero_locus_omega_chi(X, p) "
            "still gives exact Euler characteristics."
        )
    values = {pq: lo for pq, (lo, _) in bounds.items()}
    # Guard against bookkeeping errors in orbit expansion and row signs.
    for (p, q), value in values.items():
        if value < 0 or value != values[q, p] or value != values[d - p, d - q]:
            raise ArithmeticError("Hodge bound reconstruction failed its symmetry check")
        if p + q < d and value > values[p + 1, q + 1]:
            raise ArithmeticError("Hodge bound reconstruction failed its hard Lefschetz check")
    for p in range(d + 1):
        if euler_char({q: values[p, q] for q in range(d + 1)}) != chi_p[p]:
            raise ArithmeticError("Hodge bound reconstruction failed its Euler check")
    if verbose and initially_unknown:
        print(f"Resolved {initially_unknown} uncertain Hodge entries using "
              "cohomology bounds, Hodge symmetry, Serre duality, hard Lefschetz, "
              "and exact Euler characteristics.")
    return {pq: value for pq, value in values.items() if value}


def hodge_numbers(
    X, verbose: bool = False, *, assume_lefschetz: bool = False,
) -> Dict[Tuple[int, int], int]:
    """
    {(p, q): h^{p,q}} for p, q = 0, ..., dim(X).

    Dispatches on the variety type:
      - FlagVariety: exact diagonal Hodge numbers from its Schubert cells.
      - ProjectiveBundle: Leray-Hirsch additive formula
            h^{p,q}(P(F)) = sum_{i=0}^{r-1} h^{p-i,q-i}(base),  r = rank(F)
        (rigorous: the Leray spectral sequence for a projective bundle
        degenerates).
      - ZeroLocus: cohomology bounds from Koszul and conormal resolutions,
        narrowed using Hodge symmetry, Serre duality, hard Lefschetz,
        nonnegativity, and all exact Euler characteristics. Returns only
        determined values; raises ValueError with unresolved bounds if these
        constraints do not suffice.

    Set ``assume_lefschetz=True`` for a smooth zero locus known to satisfy
    weak Lefschetz with respect to its ambient variety. Then all non-middle
    cohomology is inherited from the ambient and the middle anti-diagonal is
    reconstructed exactly from ``chi(Omega^p)``. Sums of ample homogeneous
    line bundles on a bare ``G/P`` are recognized automatically.

    ``verbose=True`` reports how many uncertain entries were resolved by
    global constraints. Smoothness and regularity of the zero locus are
    assumed; the algorithm does not verify them from a defining section.
    """
    from grassmannians import FlagVariety
    from varieties import ProjectiveBundle, ZeroLocus, ProductVariety

    d = X.dimension
    result: Dict[Tuple[int, int], int] = {}

    if isinstance(X, FlagVariety):
        return {
            (p, p): multiplicity
            for p, multiplicity in enumerate(X.schubert_betti_numbers)
            if multiplicity
        }

    if isinstance(X, ProductVariety):
        # Künneth for Hodge numbers: h^{p,q}(A×B) = Σ h^{a,b}(A)·h^{c,d}(B).
        result_pq: Dict[Tuple[int, int], int] = {(0, 0): 1}
        for factor in X.factors:
            fh = hodge_numbers(factor, verbose=verbose)
            combined: Dict[Tuple[int, int], int] = {}
            for (p1, q1), v1 in result_pq.items():
                for (p2, q2), v2 in fh.items():
                    key = (p1 + p2, q1 + q2)
                    combined[key] = combined.get(key, 0) + v1 * v2
            result_pq = combined
        return {pq: v for pq, v in result_pq.items() if v}

    if isinstance(X, ProjectiveBundle):
        r = X.relative_rank
        base_hodge = hodge_numbers(X.base, verbose=verbose)
        for (bp, bq), v in base_hodge.items():
            for i in range(r):
                result[(bp + i, bq + i)] = result.get((bp + i, bq + i), 0) + v
        return result

    if isinstance(X, ZeroLocus):
        if assume_lefschetz or _obvious_ample_complete_intersection(X):
            return _lefschetz_zero_locus_hodge_numbers(X, verbose)
        return _zero_locus_hodge_numbers(X, verbose)

    raise NotImplementedError(f"hodge_numbers not implemented for {type(X)}")


def hodge_diamond(X, verbose: bool = False, *, assume_lefschetz: bool = False) -> str:
    """Pretty-print the Hodge diamond of X as a string (and print it)."""
    d = X.dimension
    h = hodge_numbers(X, verbose=verbose, assume_lefschetz=assume_lefschetz)

    cell_w = max((len(str(v)) for v in h.values()), default=1)
    cell_w = max(cell_w, 1)

    lines = []
    for total in range(2 * d + 1):
        row_entries = []
        for p in range(d + 1):
            q = total - p
            if 0 <= q <= d:
                row_entries.append((p, q))
        row_entries.sort(key=lambda pq: pq[0])
        cells = [str(h.get((p, q), 0)).rjust(cell_w) for p, q in row_entries]
        line = "  ".join(cells)
        indent = " " * ((d - len(row_entries) + 1) * (cell_w + 2))
        lines.append(indent + line)

    text = "\n".join(lines)
    print(text)
    return text
