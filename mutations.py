"""
mutations.py

Mutations of exceptional objects in a derived category, built entirely on
top of the existing RHom/shift/sum algebra (grassmannians.py, abstract.py)
and the ZeroLocus restriction machinery (varieties.py).

Given exceptional, semiorthogonal E, F (Hom^*(F,E) = 0), the left and right
mutations are defined by the triangles

    RHom(E,F) ⊗ E --ev--> F --> L_E(F) --> RHom(E,F) ⊗ E [1]
    R_F(E) --> E --coev--> RHom(E,F)^* ⊗ F --> R_F(E) [1]

(shifts here in the *standard* triangulated-category convention).

This codebase's own `[k]` operator is the *negative* of that standard shift:
code[k] = std[-k]. (Confirmed against the Koszul resolution in varieties.py,
which represents a quotient C of a short exact sequence 0->A->B->C->0 as
B + A[-1], and independently by rotating the triangle X -> 0 -> X[1]_std.)

Translating the two triangles above into this codebase's `[k]`, and writing
RHom(E,F) via its cohomology dict (graded by degree j = the Ext^j piece):

    L_E(F) = F + Σ_j dim Ext^j(E,F) · E[j - 1]
    R_F(E) = E + Σ_j dim Ext^j(E,F) · F[1 - j]

Both formulas are exact for Euler characteristics and K-classes: cone additivity
in the Grothendieck group is unconditional. They are the *formal K-theory
representative* of the mutation, built by literally forming the graded direct
sum F + E[shift] (resp. E + F[shift]) rather than tracking an actual
connecting morphism (this codebase has no notion of morphisms, only formal
sums/shifts and their cohomology).

Caveat: because there is no morphism data, this formal sum is NOT guaranteed
to reproduce the individual Ext-group dimensions of RHom(G, L_E(F)) (or
R_F(E)) for a third object G whenever G's Ext against E/F collides in
cohomological degree with G's Ext against F/E respectively -- most sharply
when G = E (resp. G = F), where the defining property of mutation forces a
vanishing that this naive sum does not know about. Concretely,
RHom(E, L_E(F)) is a theorem-guaranteed 0, but the naive formal sum can
return a nonzero-looking (though Euler-characteristic-0) list; the same
caveat applies to RHom(R_F(E), F). chi(...) remains exact. The .rank property
ignores shifts and is not the virtual rank of a complex; compute the
alternating sum of term ranks instead.


Mutations on a zero locus
-------------------------
E and F may be BundleOnZeroLocus objects, including X.canonical_bundle and
X.trivial_bundle, as long as both live on the same ZeroLocus X. RHom and chi
pair them intrinsically on X. The helpers below keep ambient lifts while
building the formal mutation and then wrap the result back onto X. Restricting
only the target inside these helpers applies the Koszul resolution once,
just as pairing two intrinsic bundle objects directly does.

For the different ambient Ext pairing between pushforwards, explicitly pass
E.pushforward() and F.pushforward() to RHom or chi.
Individual-degree reliability on X inherits the ZeroLocus caveat from the
README: the Koszul spectral sequence need not degenerate even on a single
Grassmannian. Without a separate degeneration argument, only the Euler
characteristic is certified. Prefer numerical_* for K-theory calculations.


numerical_left_mutation / numerical_right_mutation
----------------------------------------------------
[L_E(F)] and [R_F(E)] also satisfy clean identities in the Grothendieck
group, obtained by taking the K-class of the defining triangles directly
(using [X[1]_std] = -[X], the K-theory shift, and the fact that
Σ_j dim Ext^j(E,F)·(-1)^j = chi(E,F) by definition):

    [L_E(F)] = [F] - chi(E,F)·[E]
    [R_F(E)] = [E] - chi(E,F)·[F]

Unlike the graded formulas above, these are K-class identities with no
morphism-data caveat at all -- they hold unconditionally, for any exceptional
semiorthogonal pair, regardless of how RHom(E,F) is distributed across
degrees. numerical_left_mutation / numerical_right_mutation realize them as
single-term formal sums (one shift, one integer coefficient -- possibly
negative, since this is a virtual K-theory class rather than a literal sheaf)
rather than left_mutation/right_mutation's one-term-per-Ext-degree sum:

    numerical_left_mutation(E,F)  = F + chi(E,F) · E[-1]
    numerical_right_mutation(F,E) = E + chi(E,F) · F[1]

chi(G, ...) against these agrees with chi(G, left_mutation(E,F)) /
chi(G, right_mutation(F,E)) for every G (both reduce to the same K-class
identity above). The two constructions coincide termwise when RHom(E,F) is
concentrated in degree zero, as for a strong exceptional collection. In other
degrees they can have different shifts or coefficients while representing the
same K-class.
"""

from __future__ import annotations
from grassmannians import RHom, chi
from varieties import BundleOnZeroLocus


# ---------------------------------------------------------------------------
# Internal helpers: unwrap zero-locus tags, pair intrinsically, re-wrap
# ---------------------------------------------------------------------------

def _decompose(obj):
    """Return (ambient_lift, zero_locus_or_None) for a mutation operand."""
    if isinstance(obj, BundleOnZeroLocus):
        return obj.ambient_lift, obj.variety
    return obj, None


def _prepare(E, F):
    """
    Unwrap E, F to their ambient lifts and determine the common zero locus.

    Returns (E_amb, F_amb, locus) where `locus` is the shared ZeroLocus (or
    None if both operands are plain ambient bundles). Raises if the two
    operands are tagged with different zero loci.
    """
    E_amb, E_zl = _decompose(E)
    F_amb, F_zl = _decompose(F)
    if E_zl is not None or F_zl is not None:
        if E_zl is not F_zl:
            raise ValueError(
                "left/right mutation requires both objects on the same zero "
                f"locus; got {E_zl!r} and {F_zl!r}."
            )
    return E_amb, F_amb, E_zl if E_zl is not None else F_zl


def _intrinsic_rhom(E_amb, F_amb, locus):
    """Graded Ext^*(E,F): intrinsic on `locus` (restrict only the target) if
    there is one, else the plain ambient RHom."""
    if locus is None:
        return RHom(E_amb, F_amb)
    return RHom(E_amb, F_amb.restrict_support(locus))


def _intrinsic_chi(E_amb, F_amb, locus):
    """chi(E,F): intrinsic on `locus` (restrict only the target) if there is
    one, else the plain ambient chi."""
    if locus is None:
        return chi(E_amb, F_amb)
    return chi(E_amb, F_amb.restrict_support(locus))


def _wrap(bundle, locus):
    """Re-tag an ambient expression as living on `locus` (no-op if None)."""
    return bundle if locus is None else BundleOnZeroLocus(bundle, locus)


# ---------------------------------------------------------------------------
# Graded (formal K-theory representative) mutations
# ---------------------------------------------------------------------------

def left_mutation(E, F):
    """
    L_E(F): the left mutation of F through E, formally

        L_E(F) = F + Σ_j dim Ext^j(E,F) · E[j - 1]

    Requires E, F exceptional and Hom^*(F,E) = 0 (semiorthogonal, E left of F).
    E and F may both be BundleOnZeroLocus objects on a common zero locus, in
    which case Ext^j(E,F) is taken intrinsically on that locus and the result
    is returned tagged on the same locus.
    """
    E_amb, F_amb, locus = _prepare(E, F)
    result = F_amb
    for entry in _intrinsic_rhom(E_amb, F_amb, locus):
        mult = entry['multiplicity'] * entry['dimension']
        if mult:
            result = result + mult * E_amb[entry['degree'] - 1]
    return _wrap(result, locus)


def right_mutation(F, E):
    """
    R_F(E): the right mutation of E through F, formally

        R_F(E) = E + Σ_j dim Ext^j(E,F) · F[1 - j]

    Requires E, F exceptional and Hom^*(F,E) = 0 (semiorthogonal, E left of F).
    E and F may both be BundleOnZeroLocus objects on a common zero locus, in
    which case Ext^j(E,F) is taken intrinsically on that locus and the result
    is returned tagged on the same locus.
    """
    E_amb, F_amb, locus = _prepare(E, F)
    result = E_amb
    for entry in _intrinsic_rhom(E_amb, F_amb, locus):
        mult = entry['multiplicity'] * entry['dimension']
        if mult:
            result = result + mult * F_amb[1 - entry['degree']]
    return _wrap(result, locus)


# ---------------------------------------------------------------------------
# Numerical (Grothendieck-group) mutations
# ---------------------------------------------------------------------------

def numerical_left_mutation(E, F):
    """
    Purely numerical (Grothendieck-group) left mutation:

        [L_E(F)] = [F] - chi(E,F) · [E]

    realized as the single-term formal sum F + chi(E,F)·E[-1]. Exact for
    chi(G, -) against any G, unconditionally (no morphism-data caveat) --
    see the module docstring. chi(E,F) may be negative; the resulting
    multiplicity is then a formal/virtual coefficient, not a literal sheaf
    multiplicity. E and F may both live on a common zero locus (chi(E,F) is
    then intrinsic and the result is tagged on that locus).
    """
    E_amb, F_amb, locus = _prepare(E, F)
    n = _intrinsic_chi(E_amb, F_amb, locus)
    result = F_amb if n == 0 else F_amb + n * E_amb[-1]
    return _wrap(result, locus)


def numerical_right_mutation(F, E):
    """
    Purely numerical (Grothendieck-group) right mutation:

        [R_F(E)] = [E] - chi(E,F) · [F]

    realized as the single-term formal sum E + chi(E,F)·F[1]. Exact for
    chi(G, -) against any G, unconditionally (no morphism-data caveat) --
    see the module docstring. E and F may both live on a common zero locus
    (chi(E,F) is then intrinsic and the result is tagged on that locus).
    """
    # _prepare(E, F) so that chi(E,F) is computed with the target F restricted
    E_amb, F_amb, locus = _prepare(E, F)
    n = _intrinsic_chi(E_amb, F_amb, locus)
    result = E_amb if n == 0 else E_amb + n * F_amb[1]
    return _wrap(result, locus)
