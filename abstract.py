"""
abstract.py

Implements the algebraic semiring structure on formal expressions of vector
bundles, with operations given by direct sum (+) and tensor product (*).

The two central classes are:

  Monomial          — a single weighted term: multiplicity * (G1 ⊗ G2 ⊗ …)[shift]
  AbstractExpression — a formal direct sum of Monomials

Cohomology is handled abstractly at this level: the rules for sums, scalar
multiples, and cohomological shifts [k] are implemented here, while the
actual computation for a single generator is delegated to the concrete
subclass method `_generator_cohomology` (overridden in HomogeneousIrreducible).

The homological shift [k] satisfies:
  (A + B)[k] = A[k] + B[k]
  (A * B)[k] = A[0] * B[k]      (shift distributes over tensor product)
  A[k][l]    = A[k+l]            (shifts compose)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CohomologyNotImplementedError(NotImplementedError):
    """
    Raised when .cohomology is called on an AbstractGenerator that hasn't
    overridden _generator_cohomology. This should never happen in a finished
    computation — it means a leaf generator has no concrete implementation.
    Useful for debugging
    """


class TensorCohomologyNotImplementedError(NotImplementedError):
    """Raised when .cohomology is called on an expression containing a tensor
    product of generators. Child classes must override _generator_cohomology
    (or handle tensor products) to make cohomology computable."""


# ---------------------------------------------------------------------------
# Monomial: a single term  multiplicity * (G1 ⊗ G2 ⊗ …)[shift]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Monomial:
    """
    A single term in a formal expression: multiplicity * (G1 ⊗ G2 ⊗ …)[shift].

    - `factors`      : tuple of AbstractGenerator instances (the tensor product)
    - `multiplicity` : integer coefficient (direct-sum multiplicity)
    - `shift`        : integer cohomological shift applied to the whole term
    """
    multiplicity: int
    shift: int
    factors: Tuple  # tuple of AbstractGenerator instances

    def scaled(self, k: int) -> Monomial:
        return Monomial(self.multiplicity * k, self.shift, self.factors)

    def shifted(self, k: int) -> Monomial:
        return Monomial(self.multiplicity, self.shift + k, self.factors)


# ---------------------------------------------------------------------------
# AbstractExpression: a formal sum of Monomials
# ---------------------------------------------------------------------------

class AbstractExpression:
    """
    A formal linear combination of tensor-product monomials of generators, living
    in the semiring described in README.md
    """

    def __init__(self, monomials: List[Monomial]):
        self._monomials: List[Monomial] = _gather_monomials(monomials)

    @property
    def monomials(self) -> List[Monomial]:
        return list(self._monomials)

    # --- arithmetic operations ---

    def __add__(self, other: Any) -> AbstractExpression:
        other = _coerce(other)
        return AbstractExpression(self._monomials + other._monomials)

    def __radd__(self, other: Any) -> AbstractExpression:
        return _coerce(other).__add__(self)

    def __mul__(self, other: Any) -> AbstractExpression:
        if isinstance(other, int):
            if other == 0:
                return AbstractExpression([])
            return AbstractExpression([m.scaled(other) for m in self._monomials])
        other = _coerce(other)
        # tensor product, distributed over sums
        result = [
            Monomial(
                m1.multiplicity * m2.multiplicity,
                m1.shift + m2.shift,
                m1.factors + m2.factors,
            )
            for m1 in self._monomials
            for m2 in other._monomials
        ]
        return AbstractExpression(result)

    def __rmul__(self, other: Any) -> AbstractExpression:
        if isinstance(other, int):
            return self.__mul__(other)
        return _coerce(other).__mul__(self)

    def __getitem__(self, k: int) -> AbstractExpression:
        """Homological shift: expr[k] shifts every term's degree by k."""
        if not isinstance(k, int):
            raise TypeError(f"Homological shift must be an integer, got {type(k).__name__}")
        return AbstractExpression([m.shifted(k) for m in self._monomials])

    # --- cohomology ---

    @property
    def cohomology(self) -> List[Dict[str, Any]]:
        """Compute cohomology of the expression.

        Each cohomology entry is a dict:
            {'weight': list[int], 'dimension': int, 'degree': int, 'multiplicity': int}

        Rules:
          - (A + B).cohomology  = combined list, gathered by (weight, dimension, degree)
          - (k * A).cohomology  = A.cohomology with multiplicities multiplied by k
          - A[k].cohomology     = A.cohomology with degrees shifted by k
          - tensor products     → TensorCohomologyNotImplementedError (not implemented at this level!)
        """
        results: List[Dict[str, Any]] = []
        for m in self._monomials:
            if len(m.factors) > 1:
                raise TensorCohomologyNotImplementedError(
                    f"Cohomology of a tensor product of {len(m.factors)} generators "
                    "is not implemented at the abstract level. Ensure all generators "
                    "are instances of a concrete subclass."
                )
            gen = m.factors[0] # type: ignore
            for entry in gen._generator_cohomology():
                results.append({
                    'weight': entry['weight'],
                    'dimension': entry['dimension'],
                    'degree': entry['degree'] + m.shift,
                    'multiplicity': entry['multiplicity'] * m.multiplicity,
                })
        return _gather_cohomology(results)

    # --- display ---

    def __repr__(self) -> str:
        if not self._monomials:
            return "0"
        parts = []
        for m in self._monomials:
            factors_str = " ⊗ ".join(repr(f) for f in m.factors)
            body = f"({factors_str})" if len(m.factors) > 1 else factors_str
            if m.shift != 0:
                body = f"{body}[{m.shift}]"
            if m.multiplicity != 1:
                body = f"{m.multiplicity}*{body}"
            parts.append(body)
        return " + ".join(parts)


# ---------------------------------------------------------------------------
# AbstractGenerator: a leaf in the expression tree
# ---------------------------------------------------------------------------

class AbstractGenerator(AbstractExpression):
    """Base class for a single indecomposable generator (e.g. a vector bundle).
    Subclasses must override _generator_cohomology to provide actual math."""

    def __init__(self):
        # Bypass AbstractExpression.__init__ to avoid calling _gather before
        # self is fully constructed; a generator is its own single monomial.
        self._monomials: List[Monomial] = [Monomial(1, 0, (self,))]

    def _generator_cohomology(self) -> List[Dict[str, Any]]:
        """Stub. Subclasses must override this with actual cohomology data."""
        raise CohomologyNotImplementedError(
            f"{type(self).__name__} has not implemented _generator_cohomology. "
            "Override this method in a concrete subclass."
        )

    def __repr__(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce(obj: Any) -> AbstractExpression:
    if isinstance(obj, AbstractExpression):
        return obj
    raise TypeError(
        f"Expected an AbstractExpression or AbstractGenerator, got {type(obj).__name__}"
    )


def _gather_monomials(monomials: List[Monomial]) -> List[Monomial]:
    """Combine monomials sharing the same (factors, shift); drop zero terms."""
    totals: Dict[Tuple, int] = {}
    for m in monomials:
        key = (m.factors, m.shift)
        totals[key] = totals.get(key, 0) + m.multiplicity
    return [
        Monomial(mult, key[1], key[0])
        for key, mult in totals.items()
        if mult != 0
    ]


def _freeze(p: Any) -> Any:
    """
    Recursively convert (possibly nested) lists/tuples into nested tuples so a
    weight can be used as a dict key.  For a single variety a weight is a flat
    list of ints; for a ProductVariety it is a list of per-factor weights, frozen to a tuple of
    tuples.
    """
    if isinstance(p, (list, tuple)):
        return tuple(_freeze(x) for x in p)
    return p


def _gather_cohomology(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine cohomology entries that agree on (weight, dimension, degree)."""
    totals: Dict[Tuple, int] = {}
    reps: Dict[Tuple, Any] = {}   # frozen key -> original weight (for faithful output)
    for e in entries:
        key = (_freeze(e['weight']), e['dimension'], e['degree'])
        totals[key] = totals.get(key, 0) + e['multiplicity']
        reps[key] = e['weight']
    return [
        {'weight': reps[key], 'dimension': key[1], 'degree': key[2], 'multiplicity': mult}
        for key, mult in totals.items()
        if mult != 0
    ]
