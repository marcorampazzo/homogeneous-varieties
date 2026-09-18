# Guide for coding assistants

Read [README.md](README.md) before using or changing this repository. It is the user-facing API guide and the home of all copy-pastable examples.

## Running the code

- Use Python 3.10 or newer. Install from the repository root with `python -m pip install .`, or `python -m pip install -e .` for development. The modules use only the standard library at runtime; setuptools is a build dependency. Direct imports from the repository root also work.
- Import public geometric constructors from `grassmannians.py` and `varieties.py`, and Hodge calculations from `hodge.py`.
- Keep reusable calculations in the existing modules. Put runnable usage examples in the README, without adding standalone demonstration files.
- Keep citation metadata in `CITATION.cff` and the README consistent.
- `pyproject.toml` defines the distribution `homogeneous-varieties` and explicitly lists its nine top-level modules. Preserve the existing import names and update this list when adding a reusable module. Keep its version synchronized with `CITATION.cff`. `MANIFEST.in` includes the assistant guide and citation metadata in source archives.

## Where to work

| Module | Responsibility |
|---|---|
| `grassmannians.py` | Homogeneous varieties, bundles, bundle expressions, cohomology dispatch, `chi` and `RHom` |
| `root_systems.py` | Cartan matrices, roots, Weyl operations and Levi representation calculations |
| `bott.py` | Standalone Borel–Weil–Bott interface |
| `abstract.py` | Formal sums, tensor monomials, shifts and cohomology aggregation |
| `varieties.py` | Variety and bundle interfaces, projective bundles, products and intrinsic bundles on zero loci |
| `hodge.py` | Hodge numbers, exact Euler characteristics and cohomology bounds |
| `plethysms.py` | Partition characters, plethysms and Kronecker coefficients |
| `mutations.py` | Formal and numerical mutation representatives |
| `utils.py` | Partition and list utilities |

## Mathematical conventions

- Crossed nodes use Bourbaki numbering, starting at one. Bundle weights have one coefficient per node of the ambient Dynkin diagram, including central characters. Levi dominance requires nonnegative coefficients only at uncrossed nodes. Line-bundle degrees follow increasing crossed node order.
- Reuse the same variety instance for bundles on a common base; separately constructed objects represent distinct bases to the code.
- `E + F` is direct sum and `E * F` is tensor product. `E ** m` is the symmetric power, which agrees with a tensor power only for line bundles. `.dual` is a property. Do not assume negative powers work for every type.
- `canonical_bundle` and `trivial_bundle` are properties on every variety; neither takes parentheses. On a zero locus `Z`, they are intrinsic bundles with `.variety is Z`. Preserve that locus through bundle operations.
- For a bundle on `Z`, `.ambient_lift` is its chosen ambient representative; `.pushforward()` is its formal Koszul pushforward. They are different. `chi` and `RHom` on a common zero locus apply the Koszul resolution once. Use explicit pushforwards when computing ambient pairings.
- `ProjectiveBundle(X, F)` parametrizes lines in `F`. For nonnegative `m`, the pushforward of `O(m)` is `Sym^m(F.dual)`. The relative canonical bundle is `O(-rank(F)) * F.dual.det`, with the base factor implicitly pulled back. Untwisted pullbacks retain their base representation rather than acquiring a new support object.
- A cohomology entry contributes `dimension * multiplicity` in its `degree`. `E[k]` adds `k` to reported degrees, opposite to the usual derived-category shift convention. `.rank` ignores shifts and is not a virtual rank.
- Preserve exact integer and rational arithmetic. Use integer parity signs for Euler sums even when degrees are negative; Python's `(-1) ** degree` returns a float for negative integer degrees.

## Interpreting results

- The representation layer describes completely reducible homogeneous bundles. An associated graded object need not have the cohomology of a nonsplit filtered bundle.
- A zero locus assumes a regular section and uses its expected dimension. The code does not verify smoothness, generality or nonemptiness.
- Restricted `.cohomology` and `RHom` involving zero loci give total-degree Koszul E1 data. Their alternating sums are exact, but their individual entries need not be the final cohomology groups.
- Hodge calculations combine bounds, exact Euler characteristics, Hodge symmetry, Serre duality and intrinsic hard Lefschetz. Return a diamond only when all values are uniquely determined. Preserve informative failures when bounds remain unresolved; never choose differential ranks by guesswork.
- `assume_lefschetz=True` supplies a weak Lefschetz hypothesis relating the locus to its ambient variety. Use it only with mathematical justification; intrinsic hard Lefschetz alone does not justify this option.
- Mutations are formal representatives of K-classes. Numerical Euler orthogonality does not establish exceptionality or fullness.

## Checking changes

- Run the relevant README snippets from the repository root and compare their printed output with the documented output. Keep these checks in memory or temporary files outside the repository.
- For packaging changes, also install in a clean virtual environment and run the snippets from outside the checkout to verify the installed files. Check both ordinary and editable installation, and keep build artifacts in a temporary directory.
- For mathematical changes, also verify suitable exact identities, such as preservation of rank under decomposition, duality, Euler pairings or Hodge symmetries. Distinguish an exact result from a bound or an assumption.
- Update the README when public behavior changes. State limitations and unresolved ambiguities explicitly rather than silently weakening checks.
