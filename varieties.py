"""
varieties.py

Abstract base classes for varieties and vector bundles, plus the projective
bundle tower infrastructure and product varieties.

Classes defined here:

  Variety (ABC)            — abstract smooth algebraic variety
  VectorBundle (ABC)       — abstract vector bundle on a Variety
  ProjectiveBundle         — P(F) → base (Grothendieck convention)
  RelativeTwist            — O_{P(F)}(m), the tautological line bundle
  ZeroLocus                — zero locus Z(s) ⊂ ambient of a general section
  BundleOnZeroLocus        — a bundle on Z, represented by an explicit ambient lift
  ProductVariety           — X₁ × X₂
  ExternalProduct          — F₁ ⊠ F₂ = π₁*(F₁) ⊗ π₂*(F₂) on X₁ × X₂

Class hierarchy:

    Variety (ABC)
    ├── FlagVariety              (in grassmannians.py)
    │   └── Grassmannian         (in grassmannians.py)
    └── ProjectiveBundle         (here)

    VectorBundle (ABC)
    └── HomogeneousExpression    (in grassmannians.py)

Generator types (both are AbstractGenerator subclasses):
    HomogeneousIrreducible       (in grassmannians.py) — a Levi highest weight on G/P
    RelativeTwist                (here)                 — O_{P(F)}(m) on a projective bundle

A bundle on a variety in the tower is a HomogeneousExpression whose Monomial
factors are HomogeneousIrreducible and/or RelativeTwist generators.  Multi-factor
monomials are always cohomologically computable by the projective bundle formula;
see _cohomology_of_monomial in grassmannians.py.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from operator import index
from abstract import AbstractGenerator, CohomologyNotImplementedError


# ============================================================
# 1. ABSTRACT VARIETY
# ============================================================

class Variety(ABC):
    """A smooth algebraic variety."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def canonical_bundle(self) -> 'VectorBundle':
        """The canonical line bundle ω_X = det(Ω^1_X)."""
        ...

    @property
    @abstractmethod
    def trivial_bundle(self) -> 'VectorBundle':
        """The trivial line bundle O_X."""
        ...


# ============================================================
# 2. ABSTRACT VECTOR BUNDLE
# ============================================================

class VectorBundle(ABC):
    """A vector bundle on a Variety."""

    @property
    @abstractmethod
    def variety(self) -> Variety:
        ...

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @property
    @abstractmethod
    def dual(self) -> 'VectorBundle':
        ...

    @abstractmethod
    def sym_power(self, n: int) -> 'VectorBundle':
        ...

    @abstractmethod
    def wedge_power(self, n: int) -> 'VectorBundle':
        ...

    @property
    def det(self) -> 'VectorBundle':
        """Determinant line bundle ∧^{rank}(self)."""
        return self.wedge_power(self.rank)

    @property
    @abstractmethod
    def cohomology(self) -> list:
        ...


# ============================================================
# 3. PROJECTIVE BUNDLE  P(F) → base
# ============================================================

class ProjectiveBundle(Variety):
    """
    Projectivization P(F) of a vector bundle F on a base variety.

    Convention: P(F) parametrizes lines in F, i.e. Proj(Sym F^∨).
    O(-1) is the tautological subbundle and π_*(O(m)) = Sym^m(F^∨) for m ≥ 0.

    The relative dualizing sheaf is
        ω_{P(F)/base} = O(-r) ⊗ π*(det F^∨),  r = rank F.
    """

    def __init__(self, base: Variety, bundle: VectorBundle):
        self.base = base
        self.F    = bundle     # the bundle being projectivized

    @property
    def relative_rank(self) -> int:
        return self.F.rank

    @property
    def relative_dimension(self) -> int:
        return self.F.rank - 1

    @property
    def dimension(self) -> int:
        return self.base.dimension + self.relative_dimension

    @property
    def trivial_bundle(self):
        """Trivial bundle on P(F), implemented as O(0)."""
        return self.O(0)

    def O(self, m: int = 1):
        """
        The line bundle O_{P(F)}(m) returned as a HomogeneousExpression
        containing a RelativeTwist generator.
        """
        # Lazy import to avoid circular dependency (grassmannians imports varieties).
        from grassmannians import HomogeneousExpression
        from abstract import Monomial

        m = index(m)
        # Keep even O(0) on this projective variety instead of replacing it
        # with the structure sheaf of a lower level of the tower.
        return HomogeneousExpression([Monomial(1, 0, (RelativeTwist(self, m),))])

    def pullback(self, bundle: VectorBundle) -> VectorBundle:
        """
        Pull back a bundle from the base to P(F).
        In our representation a HomogeneousExpression is valid at any level of
        the tower, so this is just the identity — no wrapping needed.
        """
        return bundle

    def pushforward(self, bundle) -> 'HomogeneousExpression':
        """
        Compute Rπ_*(bundle) where π: P(F) → base.

        Applies the projective bundle formula to each monomial of *bundle*,
        looking at the total RelativeTwist power m for *this* projection:

            m ≥ 0      →  R⁰π_*(O(m)) = Sym^m(F^∨)            (Hartshorne conv.)
            -r < m < 0 →  acyclic
            m ≤ -r     →  R^{r-1}π_*(O(m)) = Sym^{-m-r}(F) ⊗ det(F)

        Returns a HomogeneousExpression on self.base.  Homological shifts in
        the result encode which derived pushforward R^i the term belongs to.
        Monomials whose RelativeTwist power for this projection is 0 are
        passed through unchanged (they live on the base already, or belong to
        a different projection in the tower).
        """
        from grassmannians import HomogeneousExpression
        from abstract import Monomial

        r = self.relative_rank
        F = self.F
        result_monomials = []

        for mono in bundle._monomials:
            rt_here  = [f for f in mono.factors if isinstance(f, RelativeTwist) and f.proj is self]
            keep     = tuple(f for f in mono.factors if not (isinstance(f, RelativeTwist) and f.proj is self))

            if not rt_here:
                # No twist for this projection — pass through unchanged.
                result_monomials.append(mono)
                continue

            m = sum(rt.power for rt in rt_here)

            if -r < m < 0:
                continue  # acyclic

            if m >= 0:
                correction_expr = F.dual.sym_power(m)
                deg_shift = 0
            else:          # m ≤ -r
                n = -m - r
                correction_expr = F.sym_power(n) * F.det
                deg_shift = r - 1

            for corr in correction_expr._monomials:
                result_monomials.append(Monomial(
                    mono.multiplicity * corr.multiplicity,
                    mono.shift + deg_shift + corr.shift,
                    keep + corr.factors,
                ))

        return HomogeneousExpression(result_monomials)

    @property
    def canonical_bundle(self):
        """
        Absolute canonical bundle:
            ω_{P(F)} = π*(ω_base) ⊗ det(F^∨) ⊗ O(-r)
        where r = rank(F).
        """
        r = self.relative_rank
        return self.base.canonical_bundle * self.F.dual.det * self.O(-r)

    def relative_canonical(self):
        """
        Relative dualizing sheaf for the convention of lines in F:
            ω_{P(F)/base} = π*(det F^∨) ⊗ O(-r)
        """
        r = self.relative_rank
        return self.F.dual.det * self.O(-r)

    def __repr__(self) -> str:
        return f"P({self.F!r})"


# ============================================================
# 4. RELATIVE TWIST  O_{P(F)}(m)
# ============================================================

class RelativeTwist(AbstractGenerator):
    """
    The line bundle O_{P(F)}(m) for a specific projection π: P(F) → base.

    Stored as a single AbstractGenerator so it can live inside the factors
    tuple of a Monomial alongside HomogeneousIrreducible generators.  Two
    RelativeTwist factors for the same proj are merged (powers add) by
    HomogeneousExpression.__mul__; see grassmannians.py.

    _generator_cohomology is intentionally left unimplemented: a RelativeTwist
    must always appear in a multi-factor monomial alongside a Grassmannian piece,
    and cohomology is computed by _cohomology_of_monomial in grassmannians.py.
    """

    def __init__(self, proj: ProjectiveBundle, power: int = 1):
        super().__init__()
        self.proj  = proj    # the ProjectiveBundle P(F) on which this lives
        self.power = index(power)   # the integer m in O(m)

    # --- identity (use object identity for proj) ---

    def __eq__(self, other) -> bool:
        return (isinstance(other, RelativeTwist)
                and self.proj is other.proj
                and self.power == other.power)

    def __hash__(self) -> int:
        return hash((id(self.proj), self.power))

    # --- cohomology stub ---

    def _generator_cohomology(self):
        raise CohomologyNotImplementedError(
            "RelativeTwist cannot be computed alone. "
            "It must appear in a tensor product with HomogeneousIrreducible factors."
        )

    # --- dual and tensor power ---

    @property
    def dual(self) -> 'RelativeTwist':
        """O(m)^∨ = O(-m)."""
        return RelativeTwist(self.proj, -self.power)

    def __pow__(self, n: int) -> 'HomogeneousExpression':
        """O(m)**n = O(m·n)."""
        if not isinstance(n, int):
            raise ValueError("exponent must be an integer")
        return self.proj.O(self.power * n)

    # --- delegate arithmetic to HomogeneousExpression (mirrors HomogeneousIrreducible) ---

    def _as_expr(self):
        from grassmannians import HomogeneousExpression
        from abstract import Monomial
        return HomogeneousExpression([Monomial(1, 0, (self,))])

    def __mul__(self, other):
        return self._as_expr() * other

    def __rmul__(self, other):
        if isinstance(other, int):
            return other * self._as_expr()
        from abstract import _coerce
        return _coerce(other) * self._as_expr()

    def __add__(self, other):
        return self._as_expr() + other

    def __radd__(self, other):
        from abstract import _coerce
        return _coerce(other) + self._as_expr()

    def __getitem__(self, k: int):
        return self._as_expr()[k]

    def pushforward(self, target):
        return self._as_expr().pushforward(target)

    @property
    def cohomology(self) -> list:
        # AbstractExpression.cohomology would call _generator_cohomology and fail.
        # Instead, delegate to the tower algorithm in grassmannians.py.
        from grassmannians import _cohomology_of_monomial
        from abstract import _gather_cohomology
        entries = []
        for mono in self._monomials:
            entries.extend(_cohomology_of_monomial(mono))
        return _gather_cohomology(entries)

    def __repr__(self) -> str:
        return f"O({self.power:+d}) on {self.proj!r}"


# ============================================================
# 5. ZERO LOCUS  Z(s) ⊂ X  of a general section of E
# ============================================================

def _validate_bundle_on_variety(bundle, variety) -> None:
    """Check an ambient lift, allowing implicit pullback along one tower.

    Validate every generator rather than only the first summand's variety.
    A restriction object must actually live on the requested ambient; it
    cannot silently become a bundle on a different locus or on its parent.
    """
    if isinstance(bundle, BundleOnZeroLocus):
        if bundle.variety is not variety:
            raise ValueError("Bundle and ambient variety have different zero-locus supports")
        return
    if isinstance(variety, ZeroLocus):
        raise ValueError("Restrict the bundle to the ambient zero locus first")
    monomials = getattr(bundle, '_monomials', None)
    if monomials is None:
        raise TypeError(f"Expected a bundle, got {type(bundle).__name__}")
    ancestors = []
    current = variety
    while True:
        ancestors.append(current)
        if not isinstance(current, ProjectiveBundle):
            break
        current = current.base
    for monomial in monomials:
        for factor in monomial.factors:
            owner = (factor.proj if isinstance(factor, RelativeTwist) else
                     factor.product if isinstance(factor, ExternalProduct) else factor.variety)
            if all(owner is not ancestor for ancestor in ancestors):
                raise ValueError("Bundle does not live on the requested ambient variety or its tower base")


class BundleOnZeroLocus(VectorBundle):
    """
    A bundle on a ZeroLocus, represented by a chosen lift to its ambient.

    Created via ZeroLocus.restrict_support(bundle) or bundle.restrict_support(Z).
    Arithmetic, duals and bundle powers remain on the same locus. chi() and
    RHom() pair these objects intrinsically on that locus, using one Koszul
    resolution. ``ambient_lift`` exposes the chosen lift; ``pushforward()``
    explicitly produces the ambient Koszul expression for i_*F.
    """

    def __init__(self, bundle, zero_locus: 'ZeroLocus'):
        if isinstance(bundle, BundleOnZeroLocus) and bundle.variety is zero_locus:
            bundle = bundle.ambient_lift  # restriction to the same locus is idempotent
        _validate_bundle_on_variety(bundle, zero_locus.ambient)
        self.bundle = bundle
        self.zero_locus = zero_locus

    @property
    def variety(self) -> 'ZeroLocus':
        return self.zero_locus

    @property
    def ambient_lift(self):
        """The chosen bundle on the immediate ambient; this is not i_*F."""
        return self.bundle

    @property
    def rank(self) -> int:
        return self.ambient_lift.rank

    @property
    def det(self) -> 'BundleOnZeroLocus':
        if not getattr(self.ambient_lift, '_monomials', None) and self.rank == 0:
            return self.variety.trivial_bundle
        return BundleOnZeroLocus(self.ambient_lift.det, self.variety)

    def sym_power(self, n: int) -> 'BundleOnZeroLocus':
        n = index(n)
        if n < 0:
            raise ValueError("power must be non-negative")
        if getattr(self.ambient_lift, '_monomials', None) == []:
            return self.variety.trivial_bundle if n == 0 else self
        return BundleOnZeroLocus(self.ambient_lift.sym_power(n), self.variety)

    def wedge_power(self, n: int) -> 'BundleOnZeroLocus':
        n = index(n)
        if n < 0:
            raise ValueError("power must be non-negative")
        if getattr(self.ambient_lift, '_monomials', None) == []:
            return self.variety.trivial_bundle if n == 0 else self
        return BundleOnZeroLocus(self.ambient_lift.wedge_power(n), self.variety)

    def _operand_lift(self, other):
        if isinstance(other, BundleOnZeroLocus) and other.variety is self.variety:
            return other.ambient_lift
        _validate_bundle_on_variety(other, self.variety.ambient)
        return other

    @property
    def dual(self) -> 'BundleOnZeroLocus':
        return BundleOnZeroLocus(self.bundle.dual, self.zero_locus)

    def __add__(self, other: 'BundleOnZeroLocus') -> 'BundleOnZeroLocus':
        if isinstance(other, int) and other == 0:
            return self
        return BundleOnZeroLocus(self.bundle + self._operand_lift(other), self.zero_locus)

    def __radd__(self, other) -> 'BundleOnZeroLocus':
        return self.__add__(other)

    def __mul__(self, other) -> 'BundleOnZeroLocus':
        lift = other if isinstance(other, int) else self._operand_lift(other)
        return BundleOnZeroLocus(self.bundle * lift, self.zero_locus)

    def __rmul__(self, other) -> 'BundleOnZeroLocus':
        return self.__mul__(other)

    def __pow__(self, n: int) -> 'BundleOnZeroLocus':
        if isinstance(n, int) and n < 0 and self.rank == 1:
            return self.dual.sym_power(-n)
        if isinstance(n, int) and n >= 0:
            return self.sym_power(n)
        return BundleOnZeroLocus(self.bundle ** n, self.zero_locus)

    def __getitem__(self, k: int) -> 'BundleOnZeroLocus':
        return BundleOnZeroLocus(self.bundle[k], self.zero_locus)

    @property
    def cohomology(self) -> list:
        """Total-degree Koszul E1 data; true cohomology needs degeneration."""
        return self.pushforward().cohomology

    def restrict_support(self, zero_locus: 'ZeroLocus') -> 'BundleOnZeroLocus':
        """Restrict further; restriction to the current locus is a no-op."""
        if zero_locus is self.variety:
            return self
        return BundleOnZeroLocus(self, zero_locus)

    def pushforward(self, target=None):
        """The formal Koszul expression for i_*F on the immediate ambient."""
        if target is not None and target is not self.variety.ambient:
            raise ValueError("A zero-locus pushforward targets its immediate ambient variety")
        return self.ambient_lift * self.zero_locus.koszul_sequence

    def __repr__(self) -> str:
        return f"({self.bundle!r})|_Z"


class ZeroLocus(Variety):
    """
    The zero locus Z of a general section of a vector bundle E on a variety X.

    Geometry:
        dim Z = dim X − rank E          (expected dimension, assumed non-empty)
        ω_Z   = (ω_X ⊗ det E)|_Z       (adjunction)

    The Koszul complex resolves O_Z on X:
        O_Z  ≅  [ ∧^0(E^∨)  →  ∧^1(E^∨)  →  …  →  ∧^r(E^∨) ]
    stored here as the formal sum
        koszul_sequence = Σ_{i=0}^{r}  ∧^i(E^∨)[-i]
    where [-i] is the total-degree shift so that chi-type computations
    automatically pick up the sign (−1)^i from the degree parity.
    """

    def __init__(self, ambient: Variety, bundle: 'VectorBundle'):
        self.ambient = ambient   # X
        self.bundle  = bundle    # E

    @property
    def dimension(self) -> int:
        return self.ambient.dimension - self.bundle.rank

    @property
    def canonical_bundle(self):
        """The canonical line bundle ω_Z = (ω_X ⊗ det E)|_Z, living on Z."""
        return self.restrict_support(self.ambient.canonical_bundle * self.bundle.det)

    @property
    def koszul_sequence(self):
        """
        Koszul complex: Σ_{i=0}^{r} ∧^i(E^∨)[-i]  as a HomogeneousExpression on X.

        The shift is -i because the Koszul complex places ∧^i(E^∨) at cohomological
        position -i (the complex runs ∧^r(E^∨) → … → E^∨ → O_X, resolving O_Z
        in degree 0).  With this convention:

          E1 terms contributing to H^d(Z, F|_Z) have total degree d.
          Actual cohomology requires accounting for the Koszul differentials.

        For chi the sign is (-1)^{j-i} = (-1)^{j+i} (mod 2), so the alternating
        sum is unchanged relative to a [+i] convention.
        """
        r = self.bundle.rank
        result = None
        for i in range(r + 1):
            term = self.bundle.dual.wedge_power(i)[-i]
            result = term if result is None else result + term
        return result

    @property
    def ideal_sheaf(self):
        """
        Locally free resolution of the ideal sheaf I_Z ⊂ O_X on the ambient X,
        obtained by truncating the Koszul resolution of O_Z: drop the i=0 term
        (which is O_X itself — the sheaf the ideal sheaf's resolution replaces)
        and shift everything by +1 so that E^∨ sits in cohomological degree 0:

            I_Z  ≅  [ ∧^r(E^∨) → … → ∧^2(E^∨) → E^∨ ],   r = rank E
            ideal_sheaf = Σ_{i=1}^{r} ∧^i(E^∨)[-i+1]

        With this shift, chi(F, ideal_sheaf) == chi(F, O_X) - chi(F, O_Z) for
        any F on the ambient (i.e. it correctly represents O_X → O_Z's kernel
        in the Grothendieck group), verified against koszul_sequence directly.
        """
        r = self.bundle.rank
        result = None
        for i in range(1, r + 1):
            term = self.bundle.dual.wedge_power(i)[-i + 1]
            result = term if result is None else result + term
        return result

    def restrict_support(self, bundle) -> BundleOnZeroLocus:
        """
        Restrict an ambient bundle to Z. The result lives on Z; use its
        pushforward() explicitly when an ambient Koszul expression is wanted.
        """
        return BundleOnZeroLocus(bundle, self)

    @property
    def trivial_bundle(self):
        """The structure sheaf O_Z, living on Z."""
        return self.restrict_support(self.ambient.trivial_bundle)

    def __repr__(self) -> str:
        return f"ZeroLocus({self.ambient!r}, {self.bundle!r})"


# ============================================================
# 6. PRODUCT VARIETY  X₁ × X₂ × … × X_m  and EXTERNAL PRODUCTS
# ============================================================

def _generator_rank(g) -> int:
    """Rank of a single generator appearing in a factor tuple."""
    if isinstance(g, ExternalProduct):
        return g.rank
    # HomogeneousIrreducible exposes .rank; RelativeTwist is a line bundle.
    return getattr(g, "rank", 1)


class ExternalProduct(AbstractGenerator):
    """
    A single external-product generator  F₁ ⊠ F₂ ⊠ … ⊠ F_m  on a ProductVariety
    X₁ × … × X_m, where F_i = π_i^*(slot_i) is pulled back from the i-th factor.

    Each *slot* is a tuple of generators (the factors of a Monomial living on
    that factor variety) — e.g. a slot may be (HomogeneousIrreducible,) or a
    tensor (HomogeneousIrreducible, RelativeTwist), or even a nested
    ExternalProduct for an iterated product.

    Multiplicity and cohomological shift are NOT stored here; like every other
    generator they live in the surrounding Monomial.  Two ExternalProduct
    factors in the same Monomial represent a tensor product and are combined
    slot-by-slot only at cohomology time (Künneth), never eagerly — so the
    generic semiring arithmetic needs no special cases.

    Cohomology is delegated to _cohomology_of_monomial in grassmannians.py,
    exactly like RelativeTwist; _generator_cohomology is intentionally a stub.
    """

    def __init__(self, product: 'ProductVariety', slots: tuple):
        super().__init__()
        self.product = product
        self.slots   = tuple(tuple(slot) for slot in slots)   # one factor-tuple per factor variety

    # --- identity ---

    def __eq__(self, other) -> bool:
        return (isinstance(other, ExternalProduct)
                and self.product is other.product
                and self.slots == other.slots)

    def __hash__(self) -> int:
        return hash((id(self.product), self.slots))

    # --- cohomology stub (handled by _cohomology_of_monomial) ---

    def _generator_cohomology(self):
        raise CohomologyNotImplementedError(
            "ExternalProduct cohomology is computed by _cohomology_of_monomial "
            "(Künneth over the product factors), not at the generator level."
        )

    # --- dual and rank ---

    @property
    def dual(self) -> 'ExternalProduct':
        """(F₁⊠…⊠F_m)^∨ = F₁^∨ ⊠ … ⊠ F_m^∨ — dualise every factor in every slot."""
        dual_slots = tuple(tuple(g.dual for g in slot) for slot in self.slots)
        return ExternalProduct(self.product, dual_slots)

    @property
    def rank(self) -> int:
        """rank(F₁⊠…⊠F_m) = Π_i rank(F_i)."""
        total = 1
        for slot in self.slots:
            for g in slot:
                total *= _generator_rank(g)
        return total

    # --- delegate arithmetic to HomogeneousExpression (mirrors RelativeTwist) ---

    def _as_expr(self):
        from grassmannians import HomogeneousExpression
        from abstract import Monomial
        return HomogeneousExpression([Monomial(1, 0, (self,))])

    def __mul__(self, other):
        return self._as_expr() * other

    def __rmul__(self, other):
        if isinstance(other, int):
            return other * self._as_expr()
        from abstract import _coerce
        return _coerce(other) * self._as_expr()

    def __add__(self, other):
        return self._as_expr() + other

    def __radd__(self, other):
        from abstract import _coerce
        return _coerce(other) + self._as_expr()

    def __getitem__(self, k: int):
        return self._as_expr()[k]

    @property
    def cohomology(self) -> list:
        from grassmannians import _cohomology_of_monomial
        from abstract import _gather_cohomology
        entries = []
        for mono in self._monomials:
            entries.extend(_cohomology_of_monomial(mono))
        return _gather_cohomology(entries)

    def __repr__(self) -> str:
        def slot_repr(slot):
            if not slot:
                return "O"
            return " ⊗ ".join(repr(g) for g in slot)
        return " ⊠ ".join(slot_repr(s) for s in self.slots)


class ProductVariety(Variety):
    """
    A product of varieties  X₁ × X₂ × … × X_m.

    Bundles on the product are direct sums of external products F₁ ⊠ … ⊠ F_m
    (pullbacks from the factors, tensored together).  Build them with `box`:

        XY = ProductVariety(X, Y)
        E  = XY.box(F, G)        # F ⊠ G  (F on X, G on Y)
        pE = XY.pullback(0, F)   # π₁^*(F) = F ⊠ O_Y

    Cohomology is computed by the Künneth formula (exact over ℂ — no Tor):
        H^k(F₁⊠…⊠F_m) = ⊕_{i₁+…+i_m=k} H^{i₁}(F₁) ⊗ … ⊗ H^{i_m}(F_m).
    """

    def __init__(self, *factors: Variety):
        if len(factors) < 2:
            raise ValueError("a ProductVariety needs at least two factor varieties")
        self.factors = tuple(factors)

    @property
    def dimension(self) -> int:
        return sum(f.dimension for f in self.factors)

    def box(self, *bundles) -> 'HomogeneousExpression':
        """
        External product  bundles[0] ⊠ bundles[1] ⊠ … (one bundle per factor).

        Each argument is a bundle on the corresponding factor variety; the
        result is a HomogeneousExpression on the product, with direct sums on
        any factor distributed out into a sum of ExternalProduct monomials.
        """
        if len(bundles) != len(self.factors):
            raise ValueError(
                f"box expects one bundle per factor ({len(self.factors)}), got {len(bundles)}"
            )
        return _external_product(self, bundles)

    def pullback(self, index: int, bundle) -> 'HomogeneousExpression':
        """π_index^*(bundle) = O ⊠ … ⊠ bundle ⊠ … ⊠ O (bundle in slot `index`)."""
        slots = [f.trivial_bundle for f in self.factors]
        slots[index] = bundle
        return _external_product(self, slots)

    @property
    def canonical_bundle(self):
        """ω_{X₁×…×X_m} = ω_{X₁} ⊠ … ⊠ ω_{X_m}."""
        return _external_product(self, [f.canonical_bundle for f in self.factors])

    @property
    def trivial_bundle(self):
        """O_{X₁×…×X_m} = O_{X₁} ⊠ … ⊠ O_{X_m}."""
        return _external_product(self, [f.trivial_bundle for f in self.factors])

    def __repr__(self) -> str:
        return " × ".join(repr(f) for f in self.factors)


def _external_product(product: ProductVariety, bundles) -> 'HomogeneousExpression':
    """
    Build the HomogeneousExpression for bundles[0] ⊠ … ⊠ bundles[-1] on
    `product`, distributing direct sums on each factor.  Each factor bundle is
    coerced to a HomogeneousExpression; its monomials supply the per-slot
    factor tuples (and contribute their multiplicity/shift to the product
    monomial, since ⊠ is bilinear and the shift is additive).
    """
    from grassmannians import HomogeneousExpression
    from abstract import Monomial

    exprs = []
    for b in bundles:
        if isinstance(b, HomogeneousExpression):
            exprs.append(b)
        elif hasattr(b, "_as_expr"):
            exprs.append(b._as_expr())
        else:
            raise TypeError(f"cannot form an external product with {type(b).__name__}")

    # Distribute: one monomial per choice of summand on each factor.
    partial = [(1, 0, [])]   # (multiplicity, shift, list of per-slot factor tuples)
    for e in exprs:
        nxt = []
        for mult, shift, slots in partial:
            for mono in e._monomials:
                nxt.append((mult * mono.multiplicity, shift + mono.shift, slots + [mono.factors]))
        partial = nxt

    monomials = [
        Monomial(mult, shift, (ExternalProduct(product, tuple(slots)),))
        for mult, shift, slots in partial
    ]
    return HomogeneousExpression(monomials)
