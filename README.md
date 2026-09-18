# Homogeneous varieties and vector bundles

**Authors: Michał Kapustka, Marco Rampazzo and Prajwal Samal.**

This Python library computes with completely reducible homogeneous vector bundles on complex rational homogeneous varieties $G/P$. It decomposes tensor products and Schur functors into irreducible summands and computes their cohomology by Borel–Weil–Bott. It also provides calculations on projective bundles, products and regular zero loci, including Euler pairings, numerical mutations and Hodge numbers within the limits explained below.

The input is: a Dynkin type, crossed nodes and highest weights. Calculations use exact integers and rational numbers. Python 3.10 or newer is required; the library has no third-party runtime dependencies (no SageMath, LiE, or NumPy). You can use it simply by downloading the files and keeping your calculation scripts in the same folder.

#### Easiest way to use it

Open the repository folder in Claude Code or OpenAI Codex and describe the calculation you want to perform in plain language. Ask the assistant to read [AGENTS.md](AGENTS.md), which explains the library's mathematical conventions, supported operations and limitations. If you prefer to work directly in Python, detailed instructions and copy-pastable examples follow below.

## Getting started

You need **Python 3.10 or newer**. If you do not already have it, download it from [python.org](https://www.python.org/downloads/). The following steps work on Windows, macOS and Linux, with no library installation required.

1. On the [repository page](https://github.com/marcorampazzo/homogeneous-varieties), choose **Code → Download ZIP**, then extract the ZIP file. While the repository is private, you need to sign in with a GitHub account that has access to it.
2. Save the example below as `my_calculation.py` **inside the extracted folder, alongside `grassmannians.py`**. Keep all the library's Python files together in that folder.
3. Open `my_calculation.py` in your Python editor and run it. For example, in IDLE, open the file with **File → Open**, then choose **Run → Run Module**.

```python
from grassmannians import Grassmannian

X = Grassmannian(2, 4)
print(X.dimension)
print(X.Udual.rank)
```

This prints the dimension of $\mathrm{Gr}(2,4)$ and the rank of its dual tautological bundle:

```text
4
2
```

You can run the other examples in the same way. To paste examples into an interactive Python session instead, start that session from the folder containing `grassmannians.py`. If you prefer to keep your scripts elsewhere, see [Optional installation with pip](#optional-installation-with-pip).

## Start with two bundles and their product

On $X=\mathrm{Gr}(3,5)$, let $U$ be the tautological rank-three subbundle. This entire snippet can be pasted into Python:

```python
from grassmannians import FlagVariety, HomogeneousIrreducible

X = FlagVariety("A4", [3])
E = HomogeneousIrreducible(X, [1, 0, 0, 0]) # this is the dual of U
F = E * HomogeneousIrreducible(X, [0, 0, -1, 0]) # this is a twist of E by O(-1)
print(E * F)
print((E * F).rank)
```

```text
E[2, 0, -1, 0] on A4/P_{3} ⊕ E[0, 1, -1, 0] on A4/P_{3}
9
```

Alternatively, we can use shorthand notations for $A_n$-type Grassmannians and their homogeneous vector bundles:

```python
from grassmannians import Grassmannian

X = Grassmannian(3, 5)
E = X.Udual                       # U^*
F = X.Udual * X.O(-1)             # U^*(-1)
print(E * F)
print((E * F).rank)
```

```text
E[2, 0, -1, 0] on G(3,5) ⊕ E[0, 1, -1, 0] on G(3,5)
9
```

Here `*` means tensor product and `+` means direct sum. The displayed answer is $\mathrm{Sym}^2\,U^*(-1)\oplus\bigwedge^2 U^*(-1)$. The ranks of the two summands are six and three.

## Describing a homogeneous variety and its bundles

In general, `crossed_nodes` lists the Bourbaki-numbered simple roots omitted from the Levi subgroup. Node numbers start at **one**. For example, `FlagVariety("C3", [3])` gives $C_3/P_3$, a Lagrangian Grassmannian. All finite irreducible Dynkin families $A,B,C,D,E,F,G$ are supported, with any (allowed) nonempty set of crossed nodes, including the full flag variety. The ambient group is understood in its simply connected form, so all integral fundamental weights are available.

A weight `[a1, ..., ar]` means $a_1\omega_1+\cdots+a_r\omega_r$, with one integer per node of the **ambient** Dynkin diagram. At uncrossed nodes its coefficients must be nonnegative: this is dominance for the Levi subgroup. Coefficients at crossed nodes may have either sign. A line bundle has zero coefficients at all uncrossed nodes; the crossed coefficients give its coefficients in a base of generators of the Picard group given by pullbacks from the generalized Grassmannians the flag admits extremal contractions to. `line_bundle([d1, ...])` lists these degrees in increasing node order.

For example, on $G_2/P_1$ the semisimple part of the Levi is $A_1$:

```python
from grassmannians import FlagVariety

X = FlagVariety("G2", crossed_nodes=[1])
E = X.bundle([0, 1])
F = X.line_bundle(1)
print(X.dimension, E.rank, F.rank)
print(E * F)
print(E.wedge_power(2))
```

```text
5 2 1
E[1, 1] on G2/P_{1}
E[3, 0] on G2/P_{1}
```

The central character matters: in this example $\det E=\mathcal O(3)$. It is retained during every Levi decomposition.

On an ordinary Grassmannian, `Grassmannian(k, n)` means $\mathrm{Gr}(k,n)=A_{n-1}/P_k$. The conveniences `X.Udual`, `X.Qdual` and `X.O(t)` denote $U^*,Q^*$ and the Plücker line bundle $\mathcal O(t)=(\det U^*)^{\otimes t}$. `X.trivial_bundle` denotes $\mathcal O_X$. The older constructor `HomogeneousIrreducible(X, lambda_, mu)` still accepts the two Grassmannian partitions and immediately converts them to ambient fundamental weights.

Use the same variety object for bundles on the same base: two separately constructed `Grassmannian(2, 4)` objects are treated as distinct bases.

## Bundle operations and cohomology

| Python | Mathematical meaning |
|---|---|
| `E + F`, `3 * E` | $E\oplus F$, $E^{\oplus3}$ |
| `E * F` | $E\otimes F$, decomposed in the Levi representation ring |
| `E.dual`, `E.rank`, `E.det` | Dual, rank and determinant of an ordinary bundle |
| `E.sym_power(m)` | $\mathrm{Sym}^m\,E$ |
| `E.wedge_power(m)` | $\bigwedge^m E$; zero if $m>\mathrm{rk}\,E$ |
| `E.schur_power([2, 1])` | $\mathbb S_{(2,1)}E$, for an irreducible input bundle |
| `E.cohomology` | A list of nonzero cohomology representations |
| `RHom(E, F)` | Cohomology of $E^*\otimes F$, for ordinary bundles |
| `chi(E, F)` | $\sum_q(-1)^q\dim\mathrm{Ext}^q(E,F)$ |

**`E ** m` means `E.sym_power(m)`, not an iterated tensor product.** These agree for a line bundle. Write `E * E` for its tensor square. Use `.dual` for inverse line bundles; negative `**` exponents are implemented for pure relative twists and line bundles on zero loci. Symmetric and exterior powers require nonnegative integer degrees. For a line bundle $L$, $\bigwedge^0L=\mathcal O$, $\bigwedge^1L=L$ and $\bigwedge^mL=0$ for $m>1$.

```python
from grassmannians import Grassmannian, RHom, chi

X = Grassmannian(2, 4)
E = X.bundle([1, -2, 1])
print(E.cohomology)
print(RHom(X.trivial_bundle, E))
print(chi(X.trivial_bundle, E))
```

```text
[{'weight': [0, 0, 0], 'dimension': 1, 'degree': 1, 'multiplicity': 1}]
[{'weight': [0, 0, 0], 'dimension': 1, 'degree': 1, 'multiplicity': 1}]
-1
```

Each entry gives the resulting dominant weight, representation dimension, cohomological degree and number of copies. The total dimension in a degree is the sum of `dimension * multiplicity`, not the number of entries. An empty list means acyclic. Here the bundle has a one-dimensional $H^1$.

The standalone interface is `bott.borel_weil_bott("A3", [1, -2, 1])`. It returns one dictionary or `None` for a singular shifted weight.

## Projective bundles and products

`ProjectiveBundle(X, F)` parametrizes **lines in $F$**: $\mathbf{P}(F)=\mathrm{Proj}_{X}\,\mathrm{Sym}(F^{\ast})$. Thus $\pi_{\ast}\mathcal{O}(m)=\mathrm{Sym}^{m}\,F^{\ast}$ for $m\geq0$, while for rank $r$ the other cases are

$$
R\pi_{\ast}\mathcal{O}(m)=0\quad(-r<m<0),\qquad
R^{r-1}\pi_{\ast}\mathcal{O}(m)=\mathrm{Sym}^{-m-r}\,F\otimes\det F
\quad(m\leq-r).
$$

The relative canonical line is $\omega_{\pi}=\mathcal{O}(-r)\otimes\pi^{\ast}\det(F^{\ast})$. This convention is dual to the quotient convention in the [Stacks Project](https://stacks.math.columbia.edu/tag/01OA).

```python
from grassmannians import Grassmannian
from varieties import ProjectiveBundle

X = Grassmannian(2, 4)
P = ProjectiveBundle(X, X.Udual.dual)  # flags of a line inside a 2-plane
L = P.O(1)
print(L.pushforward(X))              # pi_* O(1) = U^*
print(L.wedge_power(2))              # exterior square of a line bundle
```

```text
E[1, 0, 0] on G(2,4)
0
```

Base bundles may be multiplied directly by relative twists; `P.pullback(E)` is an identity operation on the representation. The pushforward formula can be iterated through towers over a homogeneous base. This does not record a new support for an untwisted pullback; see the limitations below.

For different factors use `ProductVariety.box`, which forms a **box product**. Cohomology then uses Künneth:

```python
from grassmannians import Grassmannian
from varieties import ProductVariety
from hodge import dim_dict

P1 = Grassmannian(1, 2)
X = ProductVariety(P1, P1)
L = X.box(P1.O(2), P1.O(-3))
print(dim_dict(L.cohomology))
```

```text
{1: 6}
```

## Regular zero loci: exact Euler characteristics

`ZeroLocus(X, E)` represents the zero locus of a general section of $E$. It uses the expected dimension $\dim X-\mathrm{rk}\,E$, adjunction $\omega_Z=(\omega_X\otimes\det E)|_Z$, and the Koszul resolution. The library does not specify the section or verify generality, nonemptiness or smoothness. Those are hypotheses supplied by the mathematician.

```python
from grassmannians import Grassmannian, chi
from varieties import ZeroLocus

P2 = Grassmannian(1, 3)
C = ZeroLocus(P2, P2.O(2))          # a smooth plane conic
O = C.trivial_bundle
omega_C = C.canonical_bundle
L = C.restrict_support(P2.O(2))
print(C.dimension)
print(chi(O, L))                   # chi(O_C(2)) = 5
print(chi(O, omega_C))             # chi(omega_C) = -1
print(omega_C.variety is C)
```

```text
1
5
-1
True
```

`Z.koszul_sequence` is the formal sum $\sum_i\bigwedge^iE^*[-i]$; `Z.ideal_sheaf` is its truncated resolution. `Z.canonical_bundle` and `Z.trivial_bundle` are **bundles on Z**. Both are properties, without parentheses, on every variety type. Their `.variety` is `Z`, and duals, sums, tensor products, powers and determinants preserve that locus. A further `.restrict_support(Z)` is unnecessary and is a no-op.

For a bundle `F` on `Z`, `F.ambient_lift` exposes its chosen ambient representative. `F.pushforward()` instead returns the formal Koszul expression for $i_*F$ on the immediate ambient. These are different objects. In particular, an ambient representative of the canonical bundle is `Z.canonical_bundle.ambient_lift`, while `Z.trivial_bundle.ambient_lift` is the ambient structure sheaf.

`chi(F, G)` and `RHom(F, G)` for bundles on `Z` use the **intrinsic pairing on Z**, applying its Koszul resolution once. A plain ambient operand is implicitly restricted to the same locus. Bundles on different loci are rejected. To compute ambient Ext between pushforwards, use `chi(F.pushforward(), G.pushforward())` or the analogous `RHom` call. For the conic, `chi(O, O)` is `1`, whereas `chi(O.pushforward(), O.pushforward())` is `-4`.

**Restriction cohomology is total-degree Koszul $E_1$ data.** The library does not compute the differentials. In the conic example, `L.cohomology` retains a six-dimensional term in degree zero and a one-dimensional term in degree minus one. The equation of the conic induces a rank-one map, leaving $h^0(C,\mathcal O_C(2))=5$. The Euler characteristic is already correct, but the individual terms are not the actual cohomology. This issue occurs even on a single Grassmannian.

## Hodge numbers

For $G/P$, Schubert cells give exact diagonal Hodge numbers. Products use Künneth and projective bundles use Leray–Hirsch:

```python
from grassmannians import Grassmannian
from hodge import hodge_numbers

X = Grassmannian(2, 4)
print(hodge_numbers(X))
```

```text
{(0, 0): 1, (1, 1): 1, (2, 2): 2, (3, 3): 1, (4, 4): 1}
```

For a smooth regular zero locus, `zero_locus_omega_chi(Z, p)` computes $\chi(\Omega_Z^p)$ from the conormal resolution and Koszul complex, whenever the required bundle operations are supported. These Euler characteristics remain exact even when the cotangent filtration does not split.

If weak Lefschetz is known, they determine the remaining middle Hodge numbers. Sums of ample homogeneous line bundles on a bare $G/P$, with positive expected dimension, are recognized automatically. Thus a smooth quartic surface in $\mathbf P^3$ gives:

```python
from grassmannians import Grassmannian
from varieties import ZeroLocus
from hodge import hodge_numbers

P3 = Grassmannian(1, 4)
S = ZeroLocus(P3, P3.O(4))
h = hodge_numbers(S)
print(h[2, 0], h[1, 1], h[0, 2])
```

```text
1 20 1
```

For other cases, `hodge_numbers(Z, assume_lefschetz=True)` explicitly supplies that mathematical hypothesis. `hodge_diamond(Z)` prints a diamond. Without that weak Lefschetz hypothesis, the algorithm keeps bounds for the actual cohomology through the Koszul, conormal, and cotangent-filtration calculations. It combines these with Hodge symmetry, Serre duality, hard Lefschetz, nonnegativity, and the exact equations `sum_q (-1)**q * h[p, q] = chi(Omega_Z^p)` for all `p` simultaneously. Thus a known `h[q, p] = 0` forces `h[p, q] = 0`, even when its long exact sequence alone leaves it undetermined. Koszul terms outside the dimension range are retained in Euler characteristics while their final cohomology is forced to vanish.

Hard Lefschetz supplies `h[p, q] <= h[p+1, q+1]` when `p+q < dim(Z)`. This holds intrinsically on every smooth projective `Z` and does not require `assume_lefschetz=True` or ampleness of the defining vector bundle. Together with symmetry and nonemptiness, it implies `h[p, p] >= h[0, 0] >= 1`.

Only uniquely determined values are returned. If bounds remain unresolved, `hodge_numbers` raises `ValueError` with their ranges; it does not guess differential ranks. This is a sufficient constraint procedure, not a complete solver for all spectral sequences. Smoothness and regularity of the zero locus are assumed. The raw `restrict_support(Z).cohomology` API still returns Koszul E1 data, as described above.

## How the calculations work

1. Construct the Cartan matrix and roots in Bourbaki numbering. Select the roots supported on uncrossed nodes to get the Levi root system, possibly with several components.
2. Compute Levi weight multiplicities by Freudenthal recurrence. Multiply characters for tensor products and use symmetric/exterior character formulas or Jacobi–Trudi for Schur functors. Subtract irreducible characters in highest-weight order. Full ambient coordinates retain central characters.
3. For each irreducible summand, add $\rho$, reflect negative simple walls, and count reflections. A singular shifted weight gives zero cohomology; otherwise subtract $\rho$ and apply the Weyl dimension formula.
4. Reduce supported geometric constructions to these bundle calculations, using projective pushforwards, Künneth, Koszul and conormal resolutions. Alternating sums can be computed without knowing resolution differentials.
5. Obtain homogeneous Hodge numbers from the quotient of Weyl-group length polynomials. This avoids enumerating the Weyl group, even for $E_8/B$, whose Schubert cells number 696,729,600.

## Scope and limitations

- The representation layer models $P$-modules with trivial unipotent action. It does not model extensions or arbitrary homogeneous bundles. For a cominuscule variety `X.cotangent_bundle` is available. Otherwise use `X.cotangent_associated_graded` explicitly; its cohomology need not equal that of the actual cotangent bundle.
- Formal expressions contain summands, multiplicities and shifts, with no morphisms. `E[k]` **adds** $k$ to reported degrees, the opposite of the usual derived-category shift. `.rank` ignores shifts and is not a virtual rank; `.det`, symmetric and exterior powers are for ordinary bundles.
- `mutations.py` computes formal representatives of mutation K-classes. `numerical_left_mutation(E, F)` represents $[F]-\chi(E,F)[E]$; the right analogue is `numerical_right_mutation(F, E)`. Euler pairings are meaningful; the formal `RHom` of a mutated object need not satisfy actual mutation orthogonality. A unitriangular Euler matrix alone does not prove exceptionality or fullness.
- General `schur_power` is exposed for irreducible bundles, not arbitrary direct sums. Symmetric/exterior powers of external products support a single term with one irreducible bundle per factor; sums or more complicated factor slots are not implemented. Projective bundles over products are not generally supported. Zero-locus Hodge calculations on projective towers require the immediate base to be a homogeneous $G/P$.
- There is no general support/pullback type system: zero expressions forget their variety, and untwisted pullbacks retain their base representation. Some unsupported combinations fail only when an operation is requested.
- Explicit weight enumeration, direct-sum expansions, Jacobi–Trudi and symmetric-group character sums can become expensive. Support for a Dynkin type is not a promise that every high-weight plethysm is feasible.

## Optional installation with pip

If you already use pip, you can install the downloaded library to use it from other folders. Open a terminal in the extracted folder containing `pyproject.toml` and run:

```text
python -m pip install .
```

Here `.` means the current folder, and `python` must be the Python 3.10+ interpreter you use for your calculations. If you normally launch Python as `python3` or `py`, use that name instead. After installation, imports such as `from grassmannians import Grassmannian` work as shown throughout this README.

The library is not yet published on PyPI, so installation by name alone (`pip install homogeneous-varieties`) is not available. Some Python installations restrict pip installation; in that case, use the [download-and-run instructions above](#getting-started).

## Using the repository with an LLM

[AGENTS.md](AGENTS.md) gives coding assistants a guide to the modules, mathematical conventions, bundle API and limits of the calculations. Ask an assistant to read it together with this README before using or modifying the code. All runnable examples are provided as copy-pastable snippets in this README.

## Repository map

| Reusable module | Purpose |
|---|---|
| `grassmannians.py` | $G/P$, Grassmannians, bundles, expressions, BWB cohomology, `chi` and `RHom` |
| `root_systems.py` | Exact root data, Weyl action, Levi characters and decompositions |
| `bott.py` | Standalone Borel–Weil–Bott interface |
| `abstract.py` | Formal sums, tensor terms, shifts and gathering cohomology entries |
| `varieties.py` | Projective bundles, products, zero loci and Koszul expressions |
| `hodge.py` | Hodge numbers, conormal calculations and Euler characteristics |
| `plethysms.py` | Symmetric-group characters, plethysms and Kronecker coefficients |
| `mutations.py` | Formal and numerical mutations, including intrinsic zero-locus pairings |
| `utils.py` | Standalone partition/list helpers and the older $GL_n$ dimension formula |

## Citation

If this library contributes to your research, please use the following citation data:

```bibtex
@misc{kapustka_rampazzo_samal_homogeneous_varieties,
  author = {Kapustka, Micha{\l} and Rampazzo, Marco and Samal, Prajwal},
  title = {Homogeneous varieties and vector bundles},
  year = {2026},
  howpublished = {Software},
  url = {https://github.com/marcorampazzo/homogeneous-varieties}
}
```

The file [CITATION.cff](CITATION.cff) contains machine-readable metadata for this repository.

## Mathematical background

- Pieter Belmans and Maxim Smirnov, [*Hochschild cohomology of generalised Grassmannians*](https://doi.org/10.4171/DM/912): homogeneous bundles, highest-weight conventions and cotangent filtrations.
- Humphreys, *Introduction to Lie Algebras and Representation Theory*: root systems, Weyl dimensions and weight multiplicities.
