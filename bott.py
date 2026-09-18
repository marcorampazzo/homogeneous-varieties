"""Borel--Weil--Bott for every finite Cartan type.

This is the cleaned, non-mutating successor of ``marcorampazzo/bott-theorem``.
The public ``coh`` function is retained for small standalone computations, while
the bundle library uses :func:`borel_weil_bott` and keeps the resulting highest
weight as well as its dimension and cohomological degree.
"""

from __future__ import annotations

from root_systems import RootSystem


def borel_weil_bott(cartan_type, weight) -> dict | None:
    """Compute BWB for an integral weight in fundamental-weight coordinates.

    Returns ``None`` when ``weight + rho`` is singular.  Otherwise returns a
    dictionary with the dominant highest weight, representation dimension,
    unique nonzero cohomological degree, and multiplicity one.
    """
    root_system = RootSystem.from_type(cartan_type)
    return root_system.borel_weil_bott(tuple(weight))


def coh(algebra: str, weight: list[int]):
    """Compatibility API from the original ``bott-theorem`` repository.

    ``algebra`` is one of A--G and the rank is inferred from ``len(weight)``.
    Unlike the old script, this function does not modify its input list.
    """
    result = borel_weil_bott((algebra, len(weight)), weight)
    return "acyclic" if result is None else [result["dimension"], result["degree"]]


__all__ = ["borel_weil_bott", "coh"]
