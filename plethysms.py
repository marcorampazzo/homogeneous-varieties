"""
plethysms.py

Computes Sym^n(S_λ) and ∧^n(S_λ) for a partition λ, returning the
decomposition into irreducibles as a dict {partition_tuple: multiplicity}.

The symmetric-group part is implemented from scratch using:

  1. Murnaghan-Nakayama rule  →  characters χ^λ_ρ of the symmetric group S_n
  2. Power-sum expansion of s_λ:  s_λ = Σ_ρ (χ^λ_ρ / z_ρ) p_ρ
  3. Plethystic substitution:  p_k[s_λ] = s_λ(x_1^k, x_2^k, ...) = Σ_ρ (χ^λ_ρ / z_ρ) p_{k·ρ}
  4. h_n (resp. e_n) in power-sum basis: h_n = Σ_{μ⊢n} z_μ^{-1} p_μ,
                                          e_n = Σ_{μ⊢n} ε(μ) z_μ^{-1} p_μ
  5. Inverse formula:  p_σ = Σ_{λ⊢|σ|} χ^λ_σ s_λ

This is exact arithmetic (fractions cancel to integers by Schur-positivity).
"""

from __future__ import annotations
from functools import lru_cache
from collections import Counter, defaultdict
from fractions import Fraction
import math


# ============================================================
# 1. Partition utilities
# ============================================================

@lru_cache(maxsize=None)
def partitions_of(n: int) -> tuple:
    """All partitions of n as weakly-decreasing tuples."""
    if n == 0:
        return ((),)
    def _gen(n, max_part):
        if n == 0:
            yield ()
            return
        for p in range(min(n, max_part), 0, -1):
            for rest in _gen(n - p, p):
                yield (p,) + rest
    return tuple(_gen(n, n))


def z_mu(mu: tuple) -> int:
    """z_μ = Π_i  m_i! · i^{m_i},  where m_i = #{parts equal to i}."""
    counts = Counter(mu)
    result = 1
    for i, m in counts.items():
        result *= math.factorial(m) * (i ** m)
    return result


def sign_mu(mu: tuple) -> int:
    """Sign character of S_n at cycle type μ: ε(μ) = (-1)^{n - l(μ)}."""
    return (-1) ** (sum(mu) - len(mu))


# ============================================================
# 2. Rim hook enumeration  (beta-set / abacus method)
# ============================================================

def _rim_hooks(lam: tuple, r: int) -> list[tuple[tuple, int]]:
    """
    All ways to remove a rim hook of size r from partition lam.
    Returns list of (new_partition, height) where height = #rows_spanned - 1.

    Uses the beta-set: β_i = λ_i + (n-1-i).  Removing a rim hook of size r
    corresponds to replacing one β_j by β_j - r, provided the new value is
    non-negative and not already present in β.
    The height equals #{β_k : β_j - r  <  β_k  <  β_j,  k ≠ j}.
    """
    n = len(lam)
    if n == 0:
        return []
    beta = [lam[i] + n - 1 - i for i in range(n)]
    beta_set = set(beta)
    results = []
    for j in range(n):
        new_val = beta[j] - r
        if new_val < 0 or new_val in beta_set:
            continue
        new_beta = sorted(beta[:j] + [new_val] + beta[j + 1:], reverse=True)
        new_lam_raw = [new_beta[i] - (n - 1 - i) for i in range(n)]
        if any(x < 0 for x in new_lam_raw):
            continue
        new_lam = tuple(x for x in new_lam_raw if x > 0)
        height = sum(1 for bk in beta if new_val < bk < beta[j])
        results.append((new_lam, height))
    return results


# ============================================================
# 3. Murnaghan-Nakayama rule
# ============================================================

@lru_cache(maxsize=None)
def chi_MN(lam: tuple, rho: tuple) -> int:
    """
    Character χ^λ_ρ of the symmetric group S_{|λ|} at cycle type ρ.
    Uses the Murnaghan-Nakayama rule:
        χ^λ_ρ = Σ_{rim hooks H of λ of size ρ_1}  (-1)^{ht(H)} · χ^{λ\\H}_{ρ\\{ρ_1}}
    """
    if not rho:
        return 1 if not lam else 0
    if not lam:
        return 0
    r, rest = rho[0], rho[1:]
    return sum((-1) ** ht * chi_MN(new_lam, rest)
               for new_lam, ht in _rim_hooks(lam, r))


# ============================================================
# 4. Core plethysm engine
# ============================================================

@lru_cache(maxsize=None)
def _plethysm(n: int, inner: tuple, use_sign: bool) -> dict[tuple, int]:
    """
    Compute s_{outer}[s_{inner}] where:
      - outer = (n)    (i.e. h_n) if use_sign=False  →  Sym^n
      - outer = (1^n)  (i.e. e_n) if use_sign=True   →  ∧^n

    Returns {partition: multiplicity}.
    """
    inner_size = sum(inner)
    total_size = n * inner_size

    if n == 0 or inner_size == 0:
        return {(): 1}

    # --- step 1: s_inner in power-sum basis ---
    inner_psum: dict[tuple, Fraction] = {}
    for rho in partitions_of(inner_size):
        c = chi_MN(inner, rho)
        if c != 0:
            inner_psum[rho] = Fraction(c, z_mu(rho))

    # --- step 2: substitute into h_n / e_n and accumulate in power-sum basis ---
    result_power: dict[tuple, Fraction] = defaultdict(Fraction)

    for mu in partitions_of(n):
        outer_coeff = Fraction(sign_mu(mu) if use_sign else 1, z_mu(mu))

        # Compute Π_{k ∈ μ} p_k[s_inner],  where p_k[s_inner] = Σ_ρ inner_psum[ρ] · p_{k·ρ}
        running: dict[tuple, Fraction] = {(): Fraction(1)}
        for k in mu:
            pk_inner = {
                tuple(sorted((k * x for x in rho), reverse=True)): coeff
                for rho, coeff in inner_psum.items()
            }
            new_running: dict[tuple, Fraction] = defaultdict(Fraction)
            for sigma, c1 in running.items():
                for sigma_k, c2 in pk_inner.items():
                    concat = tuple(sorted(sigma + sigma_k, reverse=True))
                    new_running[concat] += c1 * c2
            running = dict(new_running)

        for sigma, c_sigma in running.items():
            result_power[sigma] += outer_coeff * c_sigma

    # --- step 3: p_σ = Σ_{λ ⊢ total_size} χ^λ_σ s_λ ---
    result_schur: dict[tuple, Fraction] = defaultdict(Fraction)
    for sigma, c_sigma in result_power.items():
        if c_sigma == 0:
            continue
        for lam in partitions_of(total_size):
            chi_val = chi_MN(lam, sigma)
            if chi_val != 0:
                result_schur[lam] += c_sigma * chi_val

    return {lam: int(c) for lam, c in result_schur.items() if c != 0}


# ============================================================
# 5. Public API
# ============================================================

def sym_plethysm(n: int, inner_partition) -> dict[tuple, int]:
    """
    Decompose Sym^n(S_λ) into irreducibles via plethysm s_{(n)}[s_λ].

    Args:
        n: the symmetric power
        inner_partition: λ as a list or tuple of non-negative integers (weakly decreasing)

    Returns:
        dict mapping partition tuples to their multiplicities in the decomposition
    """
    inner = tuple(x for x in inner_partition if x > 0)
    return _plethysm(n, inner, use_sign=False)


def wedge_plethysm(n: int, inner_partition) -> dict[tuple, int]:
    """
    Decompose ∧^n(S_λ) into irreducibles via plethysm s_{(1^n)}[s_λ].

    Args:
        n: the exterior power
        inner_partition: λ as a list or tuple of non-negative integers (weakly decreasing)

    Returns:
        dict mapping partition tuples to their multiplicities in the decomposition
    """
    inner = tuple(x for x in inner_partition if x > 0)
    return _plethysm(n, inner, use_sign=True)


def general_plethysm(outer_partition, inner_partition) -> dict[tuple, int]:
    """
    Compute the general plethysm s_outer[s_inner] for arbitrary outer partition.

    This generalises sym_plethysm / wedge_plethysm: those are the special cases
    outer=(n) and outer=(1^n) respectively.

    Args:
        outer_partition: the outer Schur functor λ (list/tuple, weakly decreasing)
        inner_partition: the inner representation μ (list/tuple, weakly decreasing)

    Returns:
        dict mapping partition tuples to their multiplicities
    """
    outer = tuple(x for x in outer_partition if x > 0)
    inner = tuple(x for x in inner_partition if x > 0)
    return _general_plethysm_cached(outer, inner)


@lru_cache(maxsize=None)
def _general_plethysm_cached(outer: tuple, inner: tuple) -> dict[tuple, int]:
    if not outer:
        return {(): 1}          # s_{()} = 1 as a plethysm identity

    inner_size = sum(inner)

    # Trivial inner representation: s_λ[1] = 1 if λ is a single row, else 0.
    # (Corresponds to Sym^n of a line bundle = line bundle, ∧^k = 0 for k > 1.)
    if inner_size == 0:
        return {(): 1} if len(outer) <= 1 else {}

    outer_size = sum(outer)
    total_size = outer_size * inner_size

    # s_inner in power-sum basis
    inner_psum: dict[tuple, Fraction] = {}
    for rho in partitions_of(inner_size):
        c = chi_MN(inner, rho)
        if c != 0:
            inner_psum[rho] = Fraction(c, z_mu(rho))

    # s_outer = Σ_{μ ⊢ outer_size} (χ^outer_μ / z_μ) p_μ
    result_power: dict[tuple, Fraction] = defaultdict(Fraction)
    for mu in partitions_of(outer_size):
        outer_char = chi_MN(outer, mu)
        if outer_char == 0:
            continue
        outer_coeff = Fraction(outer_char, z_mu(mu))

        running: dict[tuple, Fraction] = {(): Fraction(1)}
        for k in mu:
            pk_inner = {
                tuple(sorted((k * x for x in rho), reverse=True)): coeff
                for rho, coeff in inner_psum.items()
            }
            new_running: dict[tuple, Fraction] = defaultdict(Fraction)
            for sigma, c1 in running.items():
                for sigma_k, c2 in pk_inner.items():
                    concat = tuple(sorted(sigma + sigma_k, reverse=True))
                    new_running[concat] += c1 * c2
            running = dict(new_running)

        for sigma, c_sigma in running.items():
            result_power[sigma] += outer_coeff * c_sigma

    result_schur: dict[tuple, Fraction] = defaultdict(Fraction)
    for sigma, c_sigma in result_power.items():
        if c_sigma == 0:
            continue
        for lam in partitions_of(total_size):
            chi_val = chi_MN(lam, sigma)
            if chi_val != 0:
                result_schur[lam] += c_sigma * chi_val

    return {lam: int(c) for lam, c in result_schur.items() if c != 0}


@lru_cache(maxsize=None)
def multi_kronecker(lam: tuple, mus: tuple) -> int:
    """
    Generalised Kronecker coefficient

        g^λ_{μ⁽¹⁾,…,μ⁽ᵐ⁾} = Σ_ρ  χ^λ_ρ · χ^{μ⁽¹⁾}_ρ · … · χ^{μ⁽ᵐ⁾}_ρ / z_ρ

    summed over cycle types ρ of S_n (n = |λ|).  This is the multiplicity of the
    irreducible S_n-module V_λ in the tensor product V_{μ⁽¹⁾} ⊗ … ⊗ V_{μ⁽ᵐ⁾},
    obtained from the orthogonality of characters.

    It governs the Schur functor of an external product of m bundles:
        S_λ(A₁ ⊠ … ⊠ A_m) = ⊕_{μ⁽¹⁾,…,μ⁽ᵐ⁾ ⊢ n}
                                g^λ_{μ⁽¹⁾,…,μ⁽ᵐ⁾} · S_{μ⁽¹⁾}(A₁) ⊠ … ⊠ S_{μ⁽ᵐ⁾}(A_m).

    For m = 1 it reduces to the Kronecker delta δ_{λ,μ⁽¹⁾} (character orthonormality);
    for m = 2 it is the ordinary Kronecker coefficient g^λ_{μν}.
    """
    lam = tuple(x for x in lam if x > 0)
    n = sum(lam)
    mus = tuple(tuple(x for x in mu if x > 0) for mu in mus)
    if any(sum(mu) != n for mu in mus):
        return 0
    total = Fraction(0)
    for rho in partitions_of(n):
        prod = chi_MN(lam, rho)
        if prod == 0:
            continue
        for mu in mus:
            prod *= chi_MN(mu, rho)
            if prod == 0:
                break
        if prod != 0:
            total += Fraction(prod, z_mu(rho))
    assert total.denominator == 1, "Kronecker coefficient must be an integer"
    return int(total)


def kronecker_coefficient(lam, mu, nu) -> int:
    """Ordinary Kronecker coefficient g^λ_{μ,ν} (special case of multi_kronecker)."""
    return multi_kronecker(tuple(lam), (tuple(mu), tuple(nu)))


def conjugate_partition(lam) -> tuple:
    """Return the conjugate (transpose) of a partition."""
    lam = tuple(x for x in lam if x > 0)
    if not lam:
        return ()
    return tuple(sum(1 for part in lam if part >= i) for i in range(1, lam[0] + 1))


def compositions_of(m: int, k: int):
    """
    Generate all k-tuples of non-negative integers that sum to m.
    Used in the splitting formula for Sym^m / ∧^m of a direct sum.
    """
    if k == 0:
        if m == 0:
            yield ()
        return
    if k == 1:
        yield (m,)
        return
    for n1 in range(m + 1):
        for rest in compositions_of(m - n1, k - 1):
            yield (n1,) + rest


def pretty(result: dict[tuple, int]) -> str:
    """Pretty-print a plethysm result as a sum of Schur functions."""
    if not result:
        return "0"
    terms = []
    for lam in sorted(result, key=lambda p: (sum(p), p)):
        m = result[lam]
        s = f"s{list(lam)}"
        terms.append(s if m == 1 else f"{m}·{s}")
    return " ⊕ ".join(terms)
