"""
grassmannians.py

Core computational layer: rational homogeneous varieties G/P, homogeneous vector bundles, and the
Borel–Weil–Bott theorem.

Classes defined here:

  FlagVariety            — an arbitrary finite-type G/P from Cartan data and crossed nodes.
  Grassmannian           — G(k,n) = A_{n-1}/P_k (shorthand notation for An-type Grassmannians).
  HomogeneousIrreducible — a homogeneous irreducible bundle associated with a Levi highest weight in ambient fundamental coordinates.
  HomogeneousExpression  — a formal direct sum of Monomials of such generators.

Module-level functions:

  chi(F, G)   — Euler characteristic χ(F^∨ ⊗ G); accepts BundleOnZeroLocus.
  RHom(F, G)  — full cohomology list of F^∨ ⊗ G; accepts BundleOnZeroLocus.

Internal helpers:

  _cohomology_of_monomial  — recursive projective-bundle-formula engine.
  _merge_relative_twists   — normalises O(a)⊗O(b) → O(a+b) within a factor tuple.
  _repeated_line_bundle_power — fast Sym^k / ∧^k for repeated copies of a line bundle (plethysms can be prohibitively slow on large ranks).
  _unwrap_bundles          — unwraps BundleOnZeroLocus tags and applies Koszul resolutions.
"""

from __future__ import annotations
from math import comb
from operator import index
from abstract import AbstractGenerator, AbstractExpression, Monomial, _coerce, _gather_cohomology
from bott import borel_weil_bott
from plethysms import compositions_of
from root_systems import LeviRepresentationRing, RootSystem
from varieties import Variety, VectorBundle # type: ignore


# ============================================================
# 0. TOWER COHOMOLOGY HELPERS
# ============================================================

def _repeated_line_bundle_power(gen: 'HomogeneousIrreducible', n: int, k: int, wedge: bool) -> 'HomogeneousExpression':
    """
    Sym^k(L^{⊕n}) or ∧^k(L^{⊕n}) for a line bundle L (gen, rank 1) appearing
    n > 1 times in a direct sum. Writing the n copies as C^n ⊗ L (trivial
    rank-n bundle tensored with L):
        Sym^k(C^n⊗L) = C(k+n-1, n-1) copies of L^{⊗k}
        ∧^k(C^n⊗L)   = C(n, k) copies of L^{⊗k}
    No LR multiplication needed — L^{⊗k} is just gen.sym_power(k) (the
    line-bundle fast path).
    """
    if wedge:
        if k > n:
            return HomogeneousExpression([])
        mult = comb(n, k)
    else:
        mult = comb(k + n - 1, n - 1)
    if mult == 0:
        return HomogeneousExpression([])
    return mult * gen.sym_power(k)

def _lr_multiply_hi_factors(factors: list):
    """
    Decompose a tensor product of HomogeneousIrreducible factors (all on the
    same G/P, as enforced by _lr_product) down to a single normal form:
    a dict {HomogeneousIrreducible: multiplicity}.  Returns None for an empty
    list (no HI factors present — the caller's job, not this function's, to
    decide what that means for the surrounding monomial).

    Iterative (not via HomogeneousExpression.__mul__, to avoid recursing back
    into this same reduction), so this is the single place eager character-ring
    normalization happens, regardless of how many HI factors have piled up or
    what else (RelativeTwist, ExternalProduct) sits alongside them.
    """
    if not factors:
        return None
    current = {factors[0]: 1}
    for f in factors[1:]:
        nxt: dict = {}
        for gen, mult in current.items():
            for gen2, lr_mult in gen._lr_product(f):
                nxt[gen2] = nxt.get(gen2, 0) + mult * lr_mult
        current = nxt
    return current


def _merge_relative_twists(factors: tuple) -> tuple:
    """
    In a factor tuple, combine all RelativeTwist generators that share the
    same ProjectiveBundle (their powers add). Keep zero twists to record the
    projective variety on which the resulting trivial bundle lives.
    Non-RelativeTwist factors are kept in their original order at the front.
    """
    from varieties import RelativeTwist
    hi = [f for f in factors if not isinstance(f, RelativeTwist)]
    rt_totals: dict = {}   # id(proj) -> [proj, total_power]
    for f in factors:
        if isinstance(f, RelativeTwist):
            pid = id(f.proj)
            if pid not in rt_totals:
                rt_totals[pid] = [f.proj, 0]
            rt_totals[pid][1] += f.power
    # A zero twist records the projective variety, even after O(1)*O(-1).
    rt = [RelativeTwist(proj, power) for proj, power in rt_totals.values()]
    return tuple(hi) + tuple(rt)


def _outermost_projection(projections):
    """Find the top of one tower, allowing omitted intermediate twists."""
    from varieties import ProjectiveBundle

    required = {id(p) for p in projections}
    for candidate in projections:
        ancestors = set()
        current = candidate
        while isinstance(current, ProjectiveBundle):
            ancestors.add(id(current))
            current = current.base
        if required <= ancestors:
            return candidate
    raise ValueError("Relative twists must belong to a single projective-bundle tower")


def _cohomology_of_monomial(mono: Monomial) -> list:
    """
    Compute the cohomology contribution of a single Monomial whose factors
    are HomogeneousIrreducible and/or RelativeTwist generators.

    Algorithm (recursive):
      1. Separate factors into HI (HomogeneousIrreducible) and RT (RelativeTwist).
      2. Base case — no RT factors: LR-multiply the HI factors to get a
         HomogeneousExpression, then apply BWB via AbstractExpression.cohomology.
      3. Recursive case — find the outermost ProjectiveBundle π (the one whose
         total space is not the base of any other π' in the RT factors), collect
         its total twist m, and apply the projective bundle formula:
           m ≥ 0   →  E ⊗ Sym^m(F^∨)              (R^0, no degree shift)
           -r<m<0  →  0                           (acyclic)
           m ≤ -r  →  E ⊗ Sym^{-m-r}(F)⊗det(F)  (R^{r-1}, degree +r-1)
         where F = π.F (a HomogeneousExpression on π.base).
         Each term of the correction HomogeneousExpression yields a new Monomial
         (with the remaining RT factors), and we recurse.
    """
    from varieties import RelativeTwist, ProjectiveBundle, ExternalProduct

    # ------------------------------------------------------------------
    # External-product factors (bundles on a ProductVariety): Künneth.
    # ------------------------------------------------------------------
    # Several ExternalProduct factors in one Monomial encode a tensor product
    # (F⊠…)⊗(G⊠…) = (F⊗G)⊠… — combine them slot-by-slot, compute each slot's
    # cohomology by recursing into this same engine, then take the Künneth
    # product of the per-slot cohomology lists.
    ep_factors = [f for f in mono.factors if isinstance(f, ExternalProduct)]
    if ep_factors:
        product = ep_factors[0].product
        if any(ep.product is not product for ep in ep_factors):
            raise ValueError("Cannot tensor external products on different product varieties")
        if any(isinstance(f, RelativeTwist) for f in mono.factors):
            raise NotImplementedError(
                "Cohomology of projective bundles over products is not implemented; "
                "relative twists cannot be discarded in the Kunneth formula."
            )
        if any(not isinstance(f, ExternalProduct) for f in mono.factors):
            raise ValueError("Use ProductVariety.box() or pullback() for every product factor")
        m = len(product.factors)
        # Concatenate the factor tuples slot-by-slot across all EP factors.
        slot_factors = [tuple() for _ in range(m)]
        for ep in ep_factors:
            for i, slot in enumerate(ep.slots):
                slot_factors[i] = slot_factors[i] + slot

        # Cohomology of each factor variety's slot (recursion handles BWB,
        # projective-bundle towers, even nested products).
        per_slot = [
            _cohomology_of_monomial(Monomial(1, 0, slot_factors[i]))
            for i in range(m)
        ]

        # Künneth: combine across slots.  The product representation is recorded
        # as the list of per-slot fundamental-weight vectors.
        combined = [{'weight': [], 'dimension': 1, 'degree': 0, 'multiplicity': 1}]
        for slot_cohom in per_slot:
            combined = [
                {
                    'weight':      a['weight'] + [b['weight']],
                    'dimension':   a['dimension'] * b['dimension'],
                    'degree':      a['degree'] + b['degree'],
                    'multiplicity': a['multiplicity'] * b['multiplicity'],
                }
                for a in combined
                for b in slot_cohom
            ]

        # Apply this monomial's own multiplicity and shift.
        for e in combined:
            e['degree'] += mono.shift
            e['multiplicity'] *= mono.multiplicity
        return combined

    rt_factors = [f for f in mono.factors if isinstance(f, RelativeTwist)]
    hi_factors = [f for f in mono.factors if isinstance(f, HomogeneousIrreducible)]

    # ------------------------------------------------------------------
    # Base case: only HomogeneousIrreducible factors
    # ------------------------------------------------------------------
    if not rt_factors:
        if not hi_factors:
            # Empty factor tuple = trivial bundle O_X; H^0 = k, rest vanish.
            if mono.multiplicity == 0:
                return []
            return [{'weight': [], 'dimension': 1,
                     'degree': mono.shift, 'multiplicity': mono.multiplicity}]
        # LR-multiply all HI factors into a flat HomogeneousExpression
        expr = hi_factors[0]._as_expr()
        for f in hi_factors[1:]:
            expr = expr * f._as_expr()
        # Apply this monomial's multiplicity and shift
        expr = mono.multiplicity * expr
        expr = expr[mono.shift]
        # Call the base AbstractExpression cohomology (bypasses our override,
        # which would loop; single-HI monomials go straight to BWB)
        return AbstractExpression.cohomology.fget(expr)

    # ------------------------------------------------------------------
    # Recursive case: process the outermost projection
    # ------------------------------------------------------------------
    # The outermost ProjectiveBundle is the one whose identity does NOT appear
    # as the base of any other ProjectiveBundle in rt_factors.
    projs = [rt.proj for rt in rt_factors]
    outermost = _outermost_projection(projs)

    # Total twist at this level (sum of all RelativeTwist powers for outermost)
    m = sum(rt.power for rt in rt_factors if rt.proj is outermost)

    F = outermost.F          # HomogeneousExpression on outermost.base
    r = outermost.relative_rank

    remaining_rt = [rt for rt in rt_factors if rt.proj is not outermost]

    # Delegate the projective bundle formula to ProjectiveBundle.pushforward,
    # then recurse on each resulting monomial.
    pushed = outermost.pushforward(
        AbstractExpression([mono])   # wrap as a one-term expression
    )
    results = []
    for new_mono in pushed._monomials:
        results.extend(_cohomology_of_monomial(new_mono))
    return results


def _schur_of_irreducible(gen: 'HomogeneousIrreducible', lam) -> 'HomogeneousExpression':
    """Apply an arbitrary Schur functor using the exact Levi character ring."""
    return gen.schur_power(lam)


def _schur_of_factor_slot(factor, slot: tuple, mu) -> 'HomogeneousExpression':
    """
    S_μ of the bundle that a single product-slot represents, as a
    HomogeneousExpression on the corresponding factor variety.

    A slot is the factor tuple of one product factor.  Supported (the cases the
    external-product Schur calculus actually generates):
      - a single HomogeneousIrreducible      → _schur_of_irreducible
      - the empty (trivial) bundle O          → O if μ has ≤1 row, else 0
    Multi-factor slots (e.g. an honest tensor on one factor, or a slot living on
    a projective-bundle factor) are not yet handled and raise NotImplementedError.
    """
    from varieties import RelativeTwist, ExternalProduct

    mu = tuple(x for x in mu if x > 0)
    hi = [f for f in slot if isinstance(f, HomogeneousIrreducible)]
    other = [f for f in slot if not isinstance(f, HomogeneousIrreducible)]

    if not other and len(hi) == 1:
        return _schur_of_irreducible(hi[0], mu)

    if not slot:
        # Trivial line bundle O on this factor: S_μ(O) = O for μ ≤ 1 row, else 0.
        if len(mu) <= 1:
            return factor.trivial_bundle._as_expr()
        return HomogeneousExpression([])

    raise NotImplementedError(
        "Schur functor of an external product is currently implemented only for "
        "slots that are a single irreducible bundle (or trivial) on each factor; "
        f"got slot {slot!r}."
    )


def _schur_of_external_product(lam, product, slots) -> 'HomogeneousExpression':
    """
    S_λ(A₁ ⊠ … ⊠ A_m) on a ProductVariety, via the multi-Kronecker rule
        S_λ(A₁ ⊠ … ⊠ A_m) = ⊕_{μ⁽¹⁾,…,μ⁽ᵐ⁾ ⊢ n} g^λ_{μ⁽¹⁾,…,μ⁽ᵐ⁾}
                                · S_{μ⁽¹⁾}(A₁) ⊠ … ⊠ S_{μ⁽ᵐ⁾}(A_m).
    Each per-factor S_{μ}(A_i) is computed on its own factor (a sum of
    irreducibles), then the external products are assembled by _external_product.
    """
    from itertools import product as iproduct
    from plethysms import partitions_of, multi_kronecker
    from varieties import _external_product

    lam = tuple(x for x in lam if x > 0)
    n = sum(lam)
    if n == 0:
        return product.trivial_bundle

    parts = partitions_of(n)
    # For each factor, the non-zero per-partition Schur pieces S_μ(A_i).
    per_factor = []
    for i, slot in enumerate(slots):
        pieces = {}
        for mu in parts:
            s = _schur_of_factor_slot(product.factors[i], slot, mu)
            if s._monomials:
                pieces[mu] = s
        per_factor.append(pieces)

    result = HomogeneousExpression([])
    for mu_tuple in iproduct(*[list(pf.keys()) for pf in per_factor]):
        coeff = multi_kronecker(lam, mu_tuple)
        if coeff == 0:
            continue
        boxed = _external_product(product, [per_factor[i][mu_tuple[i]] for i in range(len(slots))])
        result = result + coeff * boxed
    return result


def _det_of_external_product(product, slots) -> 'HomogeneousExpression':
    """
    Determinant of an external product, in closed form (no Kronecker needed):
        det(A₁ ⊠ … ⊠ A_m) = ⊠_i (det A_i)^{R / r_i},   R = Π_j r_j,  r_i = rank A_i,
    from det(V ⊗ W) = (det V)^{dim W} ⊗ (det W)^{dim V}.
    """
    from varieties import _external_product, _generator_rank

    ranks = []
    det_slots_exprs = []
    for slot in slots:
        r_i = 1
        slot_expr = None
        for g in slot:
            r_i *= _generator_rank(g)
            slot_expr = g._as_expr() if slot_expr is None else slot_expr * g._as_expr()
        if slot_expr is None:                       # empty slot = trivial line bundle O
            slot_expr = None                        # det O = O; handled below
        ranks.append(r_i)
        det_slots_exprs.append(slot_expr.det if slot_expr is not None else None)

    R = 1
    for r_i in ranks:
        R *= r_i

    det_pieces = []
    for i, slot in enumerate(slots):
        factor = product.factors[i]
        d = det_slots_exprs[i]
        if d is None:                               # det of trivial bundle = O
            det_pieces.append(factor.trivial_bundle._as_expr())
            continue
        power = R // ranks[i]
        det_pieces.append(d ** power)
    return _external_product(product, det_pieces)


def _unwrap_bundles(bundle1, bundle2):
    """Prepare an intrinsic pairing, applying a locus's Koszul complex once.

    Two bundles on Z are paired on Z. A plain ambient operand is implicitly
    restricted to that same locus. For ambient Ext between pushforwards,
    callers must explicitly pass A.pushforward() and B.pushforward().
    """
    from varieties import BundleOnZeroLocus, _validate_bundle_on_variety

    z1 = bundle1.variety if isinstance(bundle1, BundleOnZeroLocus) else None
    z2 = bundle2.variety if isinstance(bundle2, BundleOnZeroLocus) else None
    if z1 is None and z2 is None:
        return bundle1, bundle2
    if z1 is not None and z2 is not None and z1 is not z2:
        raise ValueError("RHom/chi require bundles on the same zero locus; "
                         "use explicit pushforwards for an ambient pairing")
    locus = z1 if z1 is not None else z2
    b1 = bundle1.ambient_lift if z1 is not None else bundle1
    b2 = bundle2.ambient_lift if z2 is not None else bundle2
    _validate_bundle_on_variety(b1, locus.ambient)
    _validate_bundle_on_variety(b2, locus.ambient)
    return b1, b2 * locus.koszul_sequence


def chi(bundle1, bundle2) -> int:
    """
    Euler characteristic χ(bundle1, bundle2) = Σ_i (-1)^i dim Ext^i(bundle1, bundle2).

    Bundles may be plain ambient expressions or BundleOnZeroLocus objects.
    If either lives on Z, this is the intrinsic pairing on Z; plain ambient
    operands are restricted implicitly. For the ambient Ext pairing between
    i_*F and i_*G, pass F.pushforward() and G.pushforward() explicitly.
    """
    total = 0
    for entry in RHom(bundle1, bundle2):
        sign = 1 if entry['degree'] % 2 == 0 else -1
        total += sign * entry['multiplicity'] * entry['dimension']
    return total


def RHom(bundle1, bundle2) -> list:
    """
    Compute RHom(bundle1, bundle2) = H*(bundle1^∨ ⊗ bundle2).

    BundleOnZeroLocus arguments are paired intrinsically on their common
    locus, applying one Koszul resolution (see _unwrap_bundles). A plain
    ambient operand is restricted implicitly. With Koszul resolutions or formal
    cones this returns total-degree E1 data, not the cohomology of the
    differential. Only its Euler characteristic is automatically exact.
    """
    b1, b2 = _unwrap_bundles(bundle1, bundle2)
    return (b1.dual * b2).cohomology


# ============================================================
# 1. RATIONAL HOMOGENEOUS VARIETIES G/P
# ============================================================

class FlagVariety(Variety):
    """A rational homogeneous variety G/P for a finite simple group G.

    ``crossed_nodes`` are the Bourbaki-numbered simple roots omitted from the
    Levi of P.  They form the natural basis of Pic(G/P); no maximal-parabolic or
    Picard-rank-one assumption is made.
    """

    def __init__(self, cartan_type, crossed_nodes):
        self.root_system = RootSystem.from_type(cartan_type)
        self.crossed_nodes = tuple(sorted(set(index(node) for node in crossed_nodes)))
        if not self.crossed_nodes:
            raise ValueError("a proper parabolic must have at least one crossed node")
        if any(node < 1 or node > self.root_system.rank for node in self.crossed_nodes):
            raise ValueError("crossed node is outside the Dynkin diagram")
        crossed = set(self.crossed_nodes)
        self.levi_nodes = tuple(
            node for node in range(1, self.root_system.rank + 1) if node not in crossed
        )
        self.representation_ring = LeviRepresentationRing(self.root_system, self.levi_nodes)

    @property
    def rank(self) -> int:
        return self.root_system.rank

    @property
    def cartan_type(self) -> str:
        return f"{self.root_system.family}{self.root_system.rank}"

    @property
    def dimension(self) -> int:
        return len(self.root_system.positive_roots) - len(
            self.root_system.positive_roots_for(self.levi_nodes)
        )

    @property
    def picard_rank(self) -> int:
        return len(self.crossed_nodes)

    @property
    def is_cominuscule(self) -> bool:
        """Whether ``G/P`` is cominuscule (hence has semisimple cotangent fiber)."""
        if len(self.crossed_nodes) != 1:
            return False
        node = self.crossed_nodes[0] - 1
        return max(root[node] for root in self.root_system.positive_roots) == 1

    @property
    def schubert_betti_numbers(self) -> tuple[int, ...]:
        """Coefficients of ``sum_{w in W/W_L} q^length(w)``.

        Since ``G/P`` has an affine Schubert-cell decomposition, the coefficient
        of ``q^p`` is both its ``2p``-th Betti number and ``h^{p,p}``.
        """
        coefficients = self.root_system.parabolic_poincare_polynomial(self.levi_nodes)
        if len(coefficients) != self.dimension + 1:
            raise ArithmeticError("Schubert polynomial has the wrong degree for G/P")
        return coefficients

    def bundle(self, weight) -> 'HomogeneousIrreducible':
        return HomogeneousIrreducible(self, weight)

    def line_bundle(self, degrees) -> 'HomogeneousIrreducible':
        """Line bundle with the given coefficients on the crossed nodes."""
        if isinstance(degrees, int):
            if self.picard_rank != 1:
                raise ValueError("give one degree for each crossed node")
            degrees = [degrees]
        degrees = list(degrees)
        if len(degrees) != self.picard_rank:
            raise ValueError(f"expected {self.picard_rank} line-bundle degrees")
        weight = [0] * self.rank
        for node, degree in zip(self.crossed_nodes, degrees):
            weight[node - 1] = index(degree)
        return HomogeneousIrreducible(self, weight)

    @property
    def trivial_bundle(self) -> 'HomogeneousIrreducible':
        return HomogeneousIrreducible(self, [0] * self.rank)

    @property
    def canonical_weight(self) -> list[int]:
        """Weight of omega_{G/P}, the negative sum of non-Levi positive roots."""
        levi_roots = set(self.root_system.positive_roots_for(self.levi_nodes))
        total = [0] * self.rank
        for root in self.root_system.positive_roots:
            if root in levi_roots:
                continue
            for i, value in enumerate(root):
                total[i] += value
        dynkin = self.root_system.root_to_weight(total)
        if any(value.denominator != 1 for value in dynkin):
            raise ArithmeticError("canonical weight is non-integral")
        return [-int(value) for value in dynkin]

    @property
    def canonical_bundle(self) -> 'HomogeneousExpression':
        return HomogeneousIrreducible(self, self.canonical_weight)._as_expr()

    def _associated_graded_from_weights(self, weights) -> 'HomogeneousExpression':
        character = {}
        for weight in weights:
            character[weight] = character.get(weight, 0) + 1
        result = HomogeneousExpression([])
        for highest, multiplicity in self.representation_ring.decompose_character(character).items():
            result = result + multiplicity * HomogeneousIrreducible(self, highest)
        return result

    @property
    def tangent_associated_graded(self) -> 'HomogeneousExpression':
        """Completely reducible Levi associated graded of T_{G/P}."""
        levi_roots = set(self.root_system.positive_roots_for(self.levi_nodes))
        weights = [
            tuple(int(value) for value in self.root_system.root_to_weight(root))
            for root in self.root_system.positive_roots if root not in levi_roots
        ]
        return self._associated_graded_from_weights(weights)

    @property
    def cotangent_associated_graded(self) -> 'HomogeneousExpression':
        """Completely reducible Levi associated graded of Omega^1_{G/P}."""
        levi_roots = set(self.root_system.positive_roots_for(self.levi_nodes))
        weights = [
            tuple(-int(value) for value in self.root_system.root_to_weight(root))
            for root in self.root_system.positive_roots if root not in levi_roots
        ]
        return self._associated_graded_from_weights(weights)

    @property
    def cotangent_bundle(self) -> 'HomogeneousExpression':
        # The P-module g/p is completely reducible for cominuscule P.  For the
        # ordinary Grassmannian subclass this property is overridden by an
        # explicit irreducible formula.  Elsewhere, returning the associated
        # graded silently would give wrong individual cohomology groups.
        if self.is_cominuscule:
            return self.cotangent_associated_graded
        raise NotImplementedError(
            "Omega^1 on a general G/P need not be completely reducible as a P-module; "
            "use cotangent_associated_graded explicitly (and its filtration spectral "
            "sequence), or a cominuscule-specific subclass."
        )

    def __repr__(self) -> str:
        nodes = ",".join(map(str, self.crossed_nodes))
        return f"{self.cartan_type}/P_{{{nodes}}}"


class Grassmannian(FlagVariety):
    """The ordinary Grassmannian G(k,n) = A_{n-1}/P_k."""

    def __init__(self, k: int, n: int):
        k, n = index(k), index(n)
        if not (1 <= k < n):
            raise ValueError("Grassmannian requires 1 <= k < n")
        self.k = k
        self.n = n
        super().__init__(f"A{n - 1}", [k])

    def weight_from_partitions(self, first_partition, second_partition) -> list[int]:
        """Compatibility conversion from the former S_lambda(U*) x S_mu(Q*) API."""
        first = list(first_partition) + [0] * (self.k - len(first_partition))
        second = list(second_partition) + [0] * (self.n - self.k - len(second_partition))
        if len(first) != self.k or len(second) != self.n - self.k:
            raise ValueError("partition is longer than the corresponding tautological rank")
        gl_weight = first + second
        return [gl_weight[i] - gl_weight[i + 1] for i in range(self.n - 1)]

    @property
    def Udual(self) -> 'HomogeneousIrreducible':
        return HomogeneousIrreducible(self, [1] + [0] * (self.n - 2))

    @property
    def Qdual(self) -> 'HomogeneousIrreducible':
        weight = [0] * (self.n - 1)
        weight[self.k - 1] = -1
        if self.k < self.n - 1:
            weight[self.k] = 1
        return HomogeneousIrreducible(self, weight)

    def O(self, degree: int = 1) -> 'HomogeneousIrreducible':
        return self.line_bundle(degree)

    @property
    def cotangent_bundle(self) -> 'HomogeneousExpression':
        return self.Udual.dual * self.Qdual

    def __repr__(self) -> str:
        return f"G({self.k},{self.n})"


# ============================================================
# 2. HOMOGENEOUS EXPRESSION
# ============================================================

class HomogeneousExpression(AbstractExpression, VectorBundle):
    """An AbstractExpression whose generators are HomogeneousIrreducible bundles
    on a G/P. Overrides __mul__ to decompose tensor products in the Levi character ring
    eagerly, so expressions are always in normal form (flat direct sums).
    Also implements VectorBundle so it can participate in the varieties.py tower."""

    # --- Variety identity (required by VectorBundle ABC) ---

    @property
    def variety(self) -> Variety:
        """
        Return the variety on which this bundle genuinely lives — not just
        the Grassmannian of its HomogeneousIrreducible factors, which may be
        pulled back from further down a projective-bundle tower.

        A monomial carrying a RelativeTwist only makes sense on the
        (outermost) ProjectiveBundle that twist belongs to, regardless of any
        HI factors also present (those are pullbacks from its base); a
        monomial carrying an ExternalProduct lives on that ProductVariety;
        otherwise (plain HI factors only) it's the Grassmannian itself.
        Considers twists in all summands, then the first non-empty monomial;
        undefined for the zero bundle.
        """
        from varieties import RelativeTwist, ProjectiveBundle, ExternalProduct

        rt_all = [f for m in self._monomials for f in m.factors
                  if isinstance(f, RelativeTwist)]
        if rt_all:
            return _outermost_projection([rt.proj for rt in rt_all])
        for m in self._monomials:
            if not m.factors:
                continue
            ep = next((f for f in m.factors if isinstance(f, ExternalProduct)), None)
            if ep is not None:
                return ep.product
            hi = next((f for f in m.factors if isinstance(f, HomogeneousIrreducible)), None)
            if hi is not None:
                return hi.G
        raise ValueError("variety is undefined for the zero bundle")

    # --- arithmetic: override to maintain type and apply LR ---

    def __add__(self, other) -> HomogeneousExpression:
        from varieties import BundleOnZeroLocus
        if isinstance(other, BundleOnZeroLocus):
            return NotImplemented
        other = _coerce(other)
        return HomogeneousExpression(self._monomials + other._monomials)

    def __radd__(self, other) -> HomogeneousExpression:
        return _coerce(other).__add__(self) # type: ignore

    def __mul__(self, other) -> HomogeneousExpression:
        from varieties import BundleOnZeroLocus
        if isinstance(other, BundleOnZeroLocus):
            return NotImplemented
        if isinstance(other, int):
            if other == 0:
                return HomogeneousExpression([])
            return HomogeneousExpression([m.scaled(other) for m in self._monomials])
        other = _coerce(other)
        result = []
        for m1 in self._monomials:
            for m2 in other._monomials:
                all_factors = m1.factors + m2.factors
                hi_factors  = [f for f in all_factors if isinstance(f, HomogeneousIrreducible)]
                # Symbolic (non-HI) factors: merge RelativeTwist factors with
                # the same proj so O(a)⊗O(b) = O(a+b) stays normal; anything
                # else (e.g. ExternalProduct) passes through unchanged.
                rest = _merge_relative_twists(
                    tuple(f for f in all_factors if not isinstance(f, HomogeneousIrreducible))
                )
                hi_dict = _lr_multiply_hi_factors(hi_factors)
                if hi_dict is None:
                    result.append(Monomial(
                        m1.multiplicity * m2.multiplicity,
                        m1.shift + m2.shift,
                        rest,
                    ))
                else:
                    for gen, lr_mult in hi_dict.items():
                        result.append(Monomial(
                            m1.multiplicity * m2.multiplicity * lr_mult,
                            m1.shift + m2.shift,
                            (gen,) + rest,
                        ))
        return HomogeneousExpression(result)

    def __rmul__(self, other) -> HomogeneousExpression:
        if isinstance(other, int):
            return self.__mul__(other)
        return _coerce(other).__mul__(self) # type: ignore

    def __getitem__(self, k: int) -> HomogeneousExpression:
        if not isinstance(k, int):
            raise TypeError(f"Homological shift must be an integer, got {type(k).__name__}")
        return HomogeneousExpression([m.shifted(k) for m in self._monomials])

    # --- geometric operations ---

    @property
    def dual(self) -> HomogeneousExpression:
        from varieties import RelativeTwist, ExternalProduct
        result = []
        for m in self._monomials:
            dual_factors = []
            for f in m.factors:
                if isinstance(f, HomogeneousIrreducible):
                    dual_factors.append(f.dual)
                elif isinstance(f, RelativeTwist):
                    dual_factors.append(RelativeTwist(f.proj, -f.power))
                elif isinstance(f, ExternalProduct):
                    dual_factors.append(f.dual)
                else:
                    raise NotImplementedError(f"dual not implemented for {type(f).__name__}")
            result.append(Monomial(m.multiplicity, -m.shift, tuple(dual_factors)))
        return HomogeneousExpression(result)

    def __pow__(self, n: int) -> HomogeneousExpression:
        """B**n = Sym^n(B) for any bundle (tensor power for line bundles)."""
        if not isinstance(n, int):
            raise ValueError("exponent must be an integer")
        from varieties import RelativeTwist
        # Fast path: single RT monomial (line bundle O(m)) — scale the twist.
        # Negative exponents are fine here: O(m)**(-1) = O(-m).
        if (len(self._monomials) == 1
                and len(self._monomials[0].factors) == 1
                and isinstance(self._monomials[0].factors[0], RelativeTwist)
                and self._monomials[0].multiplicity == 1
                and self._monomials[0].shift == 0):
            rt = self._monomials[0].factors[0]
            return rt.proj.O(rt.power * n)
        if n < 0:
            raise ValueError("negative exponent only supported for line bundles O(m)")
        return self.sym_power(n)

    @property
    def rank(self) -> int:
        # E₁⊗E₂⊗···⊗Eₖ ⊗ L₁⊗L₂⊗…: rank multiplies over the tensor product;
        # each RT factor is a line bundle (rank 1), so only the HI factors
        # (any number — they needn't be LR-merged first) contribute.
        from varieties import ExternalProduct
        total = 0
        for m in self._monomials:
            rank_m = 1
            for f in m.factors:
                if isinstance(f, HomogeneousIrreducible):
                    rank_m *= f.rank
                elif isinstance(f, ExternalProduct):
                    rank_m *= f.rank
            total += m.multiplicity * rank_m
        return total

    @property
    def det(self) -> 'HomogeneousExpression':
        """
        Determinant line bundle ∧^{rank}(self).

        Computed monomial-by-monomial via multiplicativity of det over direct
        sums (det(A⊕B) = det(A)⊗det(B), and a monomial of multiplicity c is
        c copies of the same bundle, contributing det(...)^c). Within a
        single monomial, any HI factors are character-ring-reduced
        plethysm) down to single-HI pieces, each handled by the closed-form
        HomogeneousIrreducible.det. This avoids ever computing ∧^{rank}(self)
        through a top exterior-power expansion when the defining highest
        weights and ranks are large.
        """
        from varieties import RelativeTwist

        result = None
        for mono in self._monomials:
            if mono.shift or mono.multiplicity < 0:
                raise ValueError("det requires a plain bundle with non-negative multiplicities")
            d = self._det_of_monomial(mono)
            if mono.multiplicity != 1:
                d = d ** mono.multiplicity
            result = d if result is None else result * d
        return result

    def _det_of_monomial(self, mono) -> 'HomogeneousExpression':
        """det of a single monomial's factor tuple (ignoring its multiplicity)."""
        from varieties import RelativeTwist, ExternalProduct
        from abstract import Monomial

        ep_factors = [f for f in mono.factors if isinstance(f, ExternalProduct)]
        if ep_factors:
            # External product (possibly tensored with relative-twist line bundles):
            #   det(E ⊗ L) = det(E) ⊗ L^{rank E},  with det(E) in closed form.
            if len(ep_factors) != 1 or any(
                not isinstance(f, (ExternalProduct, RelativeTwist)) for f in mono.factors
            ):
                raise NotImplementedError(
                    "det of an external product tensored with non-line-bundle factors "
                    "is not supported."
                )
            ep = ep_factors[0]
            result = _det_of_external_product(ep.product, ep.slots)
            rank_E = ep.rank
            rt_by_proj: dict = {}
            for rt in mono.factors:
                if isinstance(rt, RelativeTwist):
                    pid = id(rt.proj)
                    rt_by_proj.setdefault(pid, [rt.proj, 0])[1] += rt.power
            for proj, total_power in rt_by_proj.values():
                result = result * proj.O(total_power * rank_E)
            return result

        hi_factors = [f for f in mono.factors if isinstance(f, HomogeneousIrreducible)]
        rt_factors = [f for f in mono.factors if isinstance(f, RelativeTwist)]

        if len(hi_factors) >= 2:
            # LR-reduce the HI factors into a sum of single-HI terms, then
            # recurse: det(direct sum) = tensor product of dets.
            expr = hi_factors[0]._as_expr()
            for f in hi_factors[1:]:
                expr = expr * f._as_expr()
            result = None
            for sub_mono in expr._monomials:
                combined = Monomial(1, 0, sub_mono.factors + tuple(rt_factors))
                d = self._det_of_monomial(combined)
                if sub_mono.multiplicity != 1:
                    d = d ** sub_mono.multiplicity
                result = d if result is None else result * d
            return result

        if hi_factors:
            hi = hi_factors[0]
            result = hi.det          # fast, closed-form
            r = hi.rank
        else:
            # Pure tensor of line bundles (or trivial bundle): already rank 1.
            result = HomogeneousExpression([Monomial(1, 0, ())])
            r = 1

        rt_by_proj: dict = {}
        for rt in rt_factors:
            pid = id(rt.proj)
            if pid not in rt_by_proj:
                rt_by_proj[pid] = [rt.proj, 0]
            rt_by_proj[pid][1] += rt.power
        for proj, total_power in rt_by_proj.values():
            result = result * proj.O(total_power * r)

        return result

    def sym_power(self, m: int) -> HomogeneousExpression:
        """
        Sym^m of this direct sum, using the splitting formula:
            Sym^m(E₁ ⊕ ··· ⊕ Eₖ) = Σ_{n₁+···+nₖ=m} Sym^{n₁}(E₁) ⊗ ··· ⊗ Sym^{nₖ}(Eₖ)
        Monomials with multiplicity c are expanded into c copies.
        Raises ValueError if any monomial carries a non-zero homological shift.
        """
        return self._power_of_direct_sum(m, wedge=False)

    def wedge_power(self, m: int) -> HomogeneousExpression:
        """
        ∧^m of this direct sum, using the splitting formula:
            ∧^m(E₁ ⊕ ··· ⊕ Eₖ) = Σ_{n₁+···+nₖ=m} ∧^{n₁}(E₁) ⊗ ··· ⊗ ∧^{nₖ}(Eₖ)
        Monomials with multiplicity c are expanded into c copies.
        Raises ValueError if any monomial carries a non-zero homological shift.
        """
        return self._power_of_direct_sum(m, wedge=True)

    def _power_of_direct_sum(self, m: int, wedge: bool) -> HomogeneousExpression:
        from varieties import RelativeTwist, ExternalProduct, ProjectiveBundle

        if not isinstance(m, int) or m < 0:
            raise ValueError("power must be a non-negative integer")
        if not self._monomials:
            raise ValueError("cannot take a power of the zero bundle")

        for mono in self._monomials:
            if mono.multiplicity < 0:
                raise ValueError("sym_power / wedge_power require non-negative multiplicities")
            if mono.shift != 0:
                raise ValueError(
                    "sym_power / wedge_power are only defined for plain bundles "
                    f"(all shifts must be 0), but found shift = {mono.shift}"
                )

        if m == 0:
            return HomogeneousExpression(_coerce(self.variety.trivial_bundle)._monomials)

        # ----------------------------------------------------------------
        # External products: route to the general Schur-functor engine.
        #   ∧^m(E) = S_{(1^m)}(E),  Sym^m(E) = S_{(m)}(E).
        # Currently supported for a single external-product term (the case the
        # Koszul resolution and conormal multipliers produce); a relative twist
        # L on the term contributes L^m via S_λ(E⊗L) = S_λ(E)⊗L^{|λ|}.
        # ----------------------------------------------------------------
        if any(isinstance(f, ExternalProduct) for mono in self._monomials for f in mono.factors):
            if len(self._monomials) != 1 or self._monomials[0].multiplicity != 1:
                raise NotImplementedError(
                    "sym_power / wedge_power of a sum of external products is not yet "
                    "supported; take the power of a single external-product bundle."
                )
            mono = self._monomials[0]
            ep_factors = [f for f in mono.factors if isinstance(f, ExternalProduct)]
            rt_factors = [f for f in mono.factors if isinstance(f, RelativeTwist)]
            hi_factors = [f for f in mono.factors if isinstance(f, HomogeneousIrreducible)]
            if len(ep_factors) != 1 or hi_factors:
                raise NotImplementedError(
                    "sym/wedge of an external product tensored with other factors "
                    "is only supported for relative-twist (line bundle) factors."
                )
            ep = ep_factors[0]
            lam = (1,) * m if wedge else (m,)
            result = _schur_of_external_product(lam, ep.product, ep.slots)
            # Tensor with L^m for each relative twist on the term.
            for rt in rt_factors:
                result = result * rt.proj.O(rt.power * m)
            return result

        # ----------------------------------------------------------------
        # Special case: single monomial that is a tensor product E ⊗ L₁ ⊗ …
        # where E is an HI bundle and the Lᵢ are RT line bundles.
        #
        #   ∧^m(E ⊗ L) = ∧^m(E) ⊗ L^m
        #   Sym^m(E ⊗ L) = Sym^m(E) ⊗ L^m
        #
        # This is correct because tensoring with a line bundle commutes with
        # all Schur functors: S_λ(E ⊗ L) = S_λ(E) ⊗ L^{|λ|}.
        # ----------------------------------------------------------------
        if len(self._monomials) == 1:
            mono = self._monomials[0]
            hi_factors = [f for f in mono.factors if isinstance(f, HomogeneousIrreducible)]
            rt_factors = [f for f in mono.factors if isinstance(f, RelativeTwist)]

            if rt_factors and len(hi_factors) > 1:
                # Multiple HI factors tensored together (e.g. Q* ⊗ U*) plus an
                # RT twist: decompose the HI factors first in the Levi character ring,
                # not plethysm) into a sum of single-HI monomials, each
                # carrying the same RT twist, then fall through to the
                # general direct-sum splitting formula below.
                expr = hi_factors[0]._as_expr()
                for f in hi_factors[1:]:
                    expr = expr * f._as_expr()
                reduced_monomials = [
                    Monomial(mono.multiplicity * sub.multiplicity, 0, sub.factors + tuple(rt_factors))
                    for sub in expr._monomials
                ]
                return HomogeneousExpression(reduced_monomials)._power_of_direct_sum(m, wedge)

            if rt_factors and mono.multiplicity == 1:
                # Collect total RT power per ProjectiveBundle level
                rt_by_proj: dict = {}
                for rt in rt_factors:
                    pid = id(rt.proj)
                    if pid not in rt_by_proj:
                        rt_by_proj[pid] = [rt.proj, 0]
                    rt_by_proj[pid][1] += rt.power

                if len(hi_factors) == 0:
                    # A tensor of relative twists is a line bundle, even
                    # when the twists belong to different levels of a tower.
                    if wedge and m > 1:
                        return HomogeneousExpression([])
                    result = None
                    for proj, p in rt_by_proj.values():
                        term = proj.O(p * m)
                        result = term if result is None else result * term
                    return result

                # Single HI ⊗ RT line bundles
                hi = hi_factors[0]
                hi_result = hi.wedge_power(m) if wedge else hi.sym_power(m)
                result = mono.multiplicity * hi_result
                for proj, total_power in rt_by_proj.values():
                    result = result * proj.O(total_power * m)
                return result

        # ----------------------------------------------------------------
        # General case: direct sum (splitting formula).
        # Each monomial is assumed to have a single HI generator, plus
        # OPTIONALLY some RelativeTwist factors (e.g. a pullback-from-base
        # bundle tensored with a relative O(m) on a ProjectiveBundle). Each
        # summand therefore really represents E ⊗ L (E the HI part, L the
        # combined RT twist), and its m-th wedge/sym power must use
        # ∧^m(E⊗L) = ∧^m(E)⊗L^m / Sym^m(E⊗L) = Sym^m(E)⊗L^m — the RT factors
        # are NOT optional decoration to drop, they change the answer.
        # ----------------------------------------------------------------
        # Find the base Grassmannian from any HI factor
        G = next((
            f.G
            for mono in self._monomials
            for f in mono.factors
            if isinstance(f, HomogeneousIrreducible)
        ), None)
        if G is None:
            base = self.variety
            while isinstance(base, ProjectiveBundle):
                base = base.base
            if not isinstance(base, FlagVariety):
                raise NotImplementedError(
                    "could not identify the homogeneous base of the line-bundle sum"
                )
            G = base

        def apply_rt_twist(bundle, rt_factors: tuple, ni: int):
            if not rt_factors or ni == 0:
                return bundle
            rt_by_proj: dict = {}
            for rt in rt_factors:
                pid = id(rt.proj)
                if pid not in rt_by_proj:
                    rt_by_proj[pid] = [rt.proj, 0]
                rt_by_proj[pid][1] += rt.power
            result = bundle
            for proj, total_power in rt_by_proj.values():
                result = result * proj.O(total_power * ni)
            return result

        # Group summands by distinct (generator, RT-twist) pair (combining
        # multiplicities), instead of naively expanding a multiplicity-c
        # monomial into c literal duplicate entries. For a REPEATED LINE
        # BUNDLE (rank 1, count c > 1) that expansion is a real combinatorial
        # trap: it feeds c indistinguishable copies into
        # compositions_of(m, total_count) with full pairwise LR
        # multiplication at every composition, when Sym^k/∧^k of c copies of
        # the same line bundle L has an elementary closed form (L is rank 1,
        # so c copies = C^c ⊗ L, and Sym^k(C^c⊗L) = C(k+c-1,c-1) copies of
        # L^k, ∧^k(C^c⊗L) = C(c,k) copies of L^k — no LR multiplication
        # needed at all). Repeated generators of rank > 1 still use the
        # general (slower) expansion.
        counts: dict = {}
        order: list = []
        for mono in self._monomials:
            hi_factors = [f for f in mono.factors if isinstance(f, HomogeneousIrreducible)]
            rt_factors = tuple(f for f in mono.factors if isinstance(f, RelativeTwist))
            if not hi_factors:
                # A pure relative line bundle is the trivial homogeneous line
                # tensored with its RelativeTwist. Keeping that trivial factor
                # lets the same grouped splitting formula handle sums such as
                # O_P(1) + pi*O_X(1).
                hi_factors = [G.trivial_bundle]
            for hi in hi_factors:
                key = (hi, rt_factors)
                if key not in counts:
                    counts[key] = 0
                    order.append(key)
                counts[key] += mono.multiplicity

        line_bundle_groups: list = []   # (gen, rt_factors, count)
        plain_summands: list = []       # (gen, rt_factors)
        for key in order:
            hi, rt_factors = key
            c = counts[key]
            if c > 1 and hi.rank == 1:
                line_bundle_groups.append((hi, rt_factors, c))
            else:
                plain_summands.extend([key] * c)

        n_groups = len(plain_summands) + len(line_bundle_groups)

        result = HomogeneousExpression([])
        for comp in compositions_of(m, n_groups):
            idx = 0
            term_parts = []
            for hi, rt_factors in plain_summands:
                ni = comp[idx]; idx += 1
                base = hi.wedge_power(ni) if wedge else hi.sym_power(ni)
                term_parts.append(apply_rt_twist(base, rt_factors, ni))
            for hi, rt_factors, c in line_bundle_groups:
                ni = comp[idx]; idx += 1
                base = _repeated_line_bundle_power(hi, c, ni, wedge)
                term_parts.append(apply_rt_twist(base, rt_factors, ni))
            term = term_parts[0]
            for factor in term_parts[1:]:
                term = term * factor
            result = result + term
        return result

    # --- cohomology (handles mixed HI + RelativeTwist factor tuples) ---

    @property
    def cohomology(self) -> list:
        entries = []
        for mono in self._monomials:
            entries.extend(_cohomology_of_monomial(mono))
        return _gather_cohomology(entries)

    # --- pushforward along a tower of projective bundles ---

    def pushforward(self, target: Variety) -> HomogeneousExpression:
        """
        Compute Rπ_*(self) pushed all the way down to *target*, iterating
        through every intermediate ProjectiveBundle in the tower.

        Linearity (over sums and scalar multiples) is automatic because
        ProjectiveBundle.pushforward works monomial-by-monomial.

        Raises ValueError if *target* is not an ancestor of the current bundle's
        variety in the projective-bundle tower.
        """
        from varieties import RelativeTwist, ProjectiveBundle

        result: HomogeneousExpression = self

        if not self._monomials:
            return self
        current = self.variety
        if current is target:
            return self
        while isinstance(current, ProjectiveBundle) and current is not target:
            current = current.base
        if current is not target:
            raise ValueError(f"target variety {target!r} is not an ancestor of {self.variety!r}")

        while True:
            # Collect all RelativeTwist generators across every monomial.
            rt_all = [
                f
                for mono in result._monomials
                for f in mono.factors
                if isinstance(f, RelativeTwist)
            ]
            if not rt_all:
                break  # no more projective levels — we're already on a Grassmannian

            # Find the outermost projection: the one whose total space is not
            # the base of any other projection present in the expression.
            projs = list({id(rt.proj): rt.proj for rt in rt_all}.values())
            outermost = _outermost_projection(projs)
            # Remaining twists can all be below the requested target.
            current = target
            while isinstance(current, ProjectiveBundle):
                if current is outermost:
                    return result
                current = current.base

            result = outermost.pushforward(result)

            if outermost.base is target:
                break  # reached the requested level

        return result

    def restrict_support(self, zero_locus) -> 'BundleOnZeroLocus':
        """Restrict F to Z; call the result's pushforward() for ambient i_*(F|_Z)."""
        from varieties import BundleOnZeroLocus
        return BundleOnZeroLocus(self, zero_locus)

    # --- cohomology alias ---

    @property
    def bott(self):
        return self.cohomology

    # --- display ---

    def __repr__(self) -> str:
        if not self._monomials:
            return "0"
        parts = []
        for m in self._monomials:
            if len(m.factors) == 1:
                body = repr(m.factors[0])
            else:
                body = "(" + " ⊗ ".join(repr(f) for f in m.factors) + ")"
            if m.shift != 0:
                body = f"{body}[{m.shift}]"
            if m.multiplicity != 1:
                body = f"{m.multiplicity}·{body}"
            parts.append(body)
        return " ⊕ ".join(parts)


# ============================================================
# 3. IRREDUCIBLE HOMOGENEOUS BUNDLES
# ============================================================

class HomogeneousIrreducible(AbstractGenerator):
    """The bundle E_weight associated to an irreducible Levi representation.

    ``weight`` is the coefficient vector in the basis of fundamental weights
    of the ambient group G.  It may be arbitrary on crossed nodes and must be
    non-negative on uncrossed (Levi) nodes.

    For one transition release, ordinary Grassmannians also accept the former
    ``HomogeneousIrreducible(G, lambda, mu)`` form; it is converted immediately
    and never stored as a pair of partitions.
    """

    def __init__(self, variety: FlagVariety, weight, second_partition=None):
        super().__init__()
        if not isinstance(variety, FlagVariety):
            raise TypeError("HomogeneousIrreducible requires a FlagVariety G/P")
        if second_partition is not None:
            if not isinstance(variety, Grassmannian):
                raise TypeError("the legacy two-partition form is only defined on Grassmannians")
            weight = variety.weight_from_partitions(weight, second_partition)
        self.G = variety
        self.weight = variety.representation_ring.validate_highest_weight(weight)

    # --- identity ---

    def __eq__(self, other) -> bool:
        if not isinstance(other, HomogeneousIrreducible):
            return NotImplemented
        return self.G is other.G and self.weight == other.weight

    def __hash__(self) -> int:
        return hash((id(self.G), self.weight))

    @property
    def variety(self) -> FlagVariety:
        return self.G

    # --- delegate arithmetic to HomogeneousExpression ---

    def _as_expr(self) -> HomogeneousExpression:
        return HomogeneousExpression([Monomial(1, 0, (self,))])

    def __add__(self, other) -> HomogeneousExpression:
        return self._as_expr() + other

    def __radd__(self, other) -> HomogeneousExpression:
        return _coerce(other) + self._as_expr() # type: ignore

    def __mul__(self, other) -> HomogeneousExpression:
        if isinstance(other, int):
            return other * self._as_expr()
        return self._as_expr() * other

    def __rmul__(self, other) -> HomogeneousExpression:
        if isinstance(other, int):
            return other * self._as_expr()
        return _coerce(other) * self._as_expr() # type: ignore

    def __getitem__(self, k: int) -> HomogeneousExpression:
        return self._as_expr()[k]

    def __pow__(self, n: int) -> HomogeneousExpression:
        return self._as_expr() ** n

    # --- tensor product decomposition ---

    def _lr_product(self, other: 'HomogeneousIrreducible') -> list:
        """Decompose a tensor product in the representation ring of the Levi."""
        if self.G is not other.G:
            raise ValueError(
                f"Cannot tensor-multiply bundles on different varieties: "
                f"{self.G!r} vs {other.G!r}. "
                "For bundles on factors of a ProductVariety use ProductVariety.box()."
            )
        decomposition = self.G.representation_ring.tensor_product(self.weight, other.weight)
        return [
            (HomogeneousIrreducible(self.G, weight), multiplicity)
            for weight, multiplicity in decomposition.items()
        ]

    def normalize(self) -> 'HomogeneousIrreducible':
        """Compatibility no-op: fundamental-weight coordinates are already canonical."""
        return self

    # --- dual, rank, and determinant ---

    @property
    def dual(self) -> 'HomogeneousIrreducible':
        return HomogeneousIrreducible(
            self.G, self.G.representation_ring.dual_highest_weight(self.weight)
        )

    @property
    def rank(self) -> int:
        return self.G.representation_ring.dimension(self.weight)

    @property
    def det(self) -> HomogeneousExpression:
        determinant = self.G.representation_ring.determinant_weight(self.weight)
        return HomogeneousIrreducible(self.G, determinant)._as_expr()

    # --- Schur functors from exact Levi characters ---

    def _expression_from_decomposition(self, decomposition) -> HomogeneousExpression:
        result = HomogeneousExpression([])
        for weight, multiplicity in decomposition.items():
            result = result + multiplicity * HomogeneousIrreducible(self.G, weight)
        return result

    def sym_power(self, m: int) -> HomogeneousExpression:
        return self._expression_from_decomposition(
            self.G.representation_ring.symmetric_power(self.weight, m)
        )

    def wedge_power(self, m: int) -> HomogeneousExpression:
        return self._expression_from_decomposition(
            self.G.representation_ring.exterior_power(self.weight, m)
        )

    def schur_power(self, partition) -> HomogeneousExpression:
        return self._expression_from_decomposition(
            self.G.representation_ring.schur_power(self.weight, partition)
        )

    @property
    def bott(self):
        return self._as_expr().cohomology

    # --- Borel--Weil--Bott ---

    def _generator_cohomology(self) -> list:
        result = borel_weil_bott(self.G.root_system, self.weight)
        return [] if result is None else [result]

    def restrict_support(self, zero_locus) -> 'BundleOnZeroLocus':
        """Restrict F to Z; call the result's pushforward() for ambient i_*(F|_Z)."""
        from varieties import BundleOnZeroLocus
        return BundleOnZeroLocus(self._as_expr(), zero_locus)

    def __repr__(self) -> str:
        return f"E{list(self.weight)} on {self.G!r}"


# ProjectiveBundle, BundleOnProjectiveBundle, and their pushforward calculus
# now live in varieties.py.
