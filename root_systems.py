"""Exact finite-root-system and Levi representation computations.

Weights are always written in the basis of the ambient fundamental weights.
If ``weight = (a_1, ..., a_r)``, then ``a_i = <weight, alpha_i^vee>``.
Simple roots and Dynkin nodes use Bourbaki's numbering.

The module is deliberately self-contained.  It implements the small amount of
highest-weight representation theory needed by homogeneous bundles on G/P:

* root generation and the Weyl action for every finite simple Cartan type;
* the Weyl dimension formula and Borel--Weil--Bott;
* exact weight multiplicities via Freudenthal's recurrence;
* decomposition of tensor and Schur functors for representations of a Levi.

The last item is the uniform replacement for Littlewood--Richardson rules on
the two GL factors of an ordinary Grassmannian.  A parabolic's unipotent radical
acts trivially on an irreducible homogeneous bundle, so all of these operations
take place in the semisimple representation category of its Levi quotient.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache
from itertools import permutations
from math import comb, gcd
from operator import index
from typing import Iterable, Mapping


Weight = tuple[int, ...]
Root = tuple[int, ...]  # coefficients in the basis of simple roots
Character = dict[Weight, int]


def _weyl_degrees(family: str, rank: int) -> tuple[int, ...]:
    """Degrees of basic invariants of an irreducible finite Weyl group."""
    if family == "A":
        return tuple(range(2, rank + 2))
    if family in {"B", "C"}:
        return tuple(2 * i for i in range(1, rank + 1))
    if family == "D":
        return tuple(sorted((*range(2, 2 * rank - 1, 2), rank)))
    exceptional = {
        ("E", 6): (2, 5, 6, 8, 9, 12),
        ("E", 7): (2, 6, 8, 10, 12, 14, 18),
        ("E", 8): (2, 8, 12, 14, 18, 20, 24, 30),
        ("F", 4): (2, 6, 8, 12),
        ("G", 2): (2, 6),
    }
    try:
        return exceptional[(family, rank)]
    except KeyError as error:
        raise ValueError(f"invalid finite Cartan type {family}{rank}") from error


def _multiply_polynomials(left: Iterable[int], right: Iterable[int]) -> tuple[int, ...]:
    left_t, right_t = tuple(left), tuple(right)
    result = [0] * (len(left_t) + len(right_t) - 1)
    for i, a in enumerate(left_t):
        for j, b in enumerate(right_t):
            result[i + j] += a * b
    return tuple(result)


def _divide_polynomials(numerator: Iterable[int], denominator: Iterable[int]) -> tuple[int, ...]:
    """Exact division in Z[q] for polynomials stored from low to high degree."""
    numerator_t, denominator_t = tuple(numerator), tuple(denominator)
    if not denominator_t or denominator_t[0] != 1:
        raise ValueError("polynomial divisor must have constant coefficient 1")
    quotient_degree = len(numerator_t) - len(denominator_t)
    if quotient_degree < 0:
        raise ArithmeticError("polynomial divisor has greater degree than dividend")
    remainder = list(numerator_t)
    quotient = [0] * (quotient_degree + 1)
    for degree in range(quotient_degree + 1):
        coefficient = remainder[degree]
        quotient[degree] = coefficient
        if coefficient:
            for offset, value in enumerate(denominator_t):
                remainder[degree + offset] -= coefficient * value
    if any(remainder[quotient_degree + 1:]):
        raise ArithmeticError("Poincare polynomial quotient is not exact")
    return tuple(quotient)


def _identity(n: int) -> list[list[Fraction]]:
    return [[Fraction(i == j) for j in range(n)] for i in range(n)]


def _inverse(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    """Invert a square integer matrix over the rationals."""
    n = len(matrix)
    work = [list(map(Fraction, row)) + eye for row, eye in zip(matrix, _identity(n))]
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column]), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            scale = work[row][column]
            if scale:
                work[row] = [a - scale * b for a, b in zip(work[row], work[column])]
    return tuple(tuple(row[n:]) for row in work)


def _matvec(matrix, vector) -> tuple[Fraction, ...]:
    return tuple(sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _cartan_matrix(family: str, rank: int) -> tuple[tuple[int, ...], ...]:
    family = family.upper()
    allowed = {
        "A": rank >= 1,
        "B": rank >= 2,
        "C": rank >= 2,
        "D": rank >= 4,
        "E": rank in (6, 7, 8),
        "F": rank == 4,
        "G": rank == 2,
    }
    if family not in allowed or not allowed[family]:
        raise ValueError(f"invalid finite Cartan type {family}{rank}")

    matrix = [[0] * rank for _ in range(rank)]
    for i in range(rank):
        matrix[i][i] = 2

    if family in {"A", "B", "C"}:
        edges = [(i, i + 1) for i in range(rank - 1)]
    elif family == "D":
        edges = [(i, i + 1) for i in range(rank - 3)]
        edges += [(rank - 3, rank - 2), (rank - 3, rank - 1)]
    elif family == "E":
        # Bourbaki: 1--3--4--5--6(--7--8), with node 2 attached to node 4.
        edges = [(0, 2), (2, 3), (1, 3)]
        edges += [(i, i + 1) for i in range(3, rank - 1)]
    else:
        edges = []

    for i, j in edges:
        matrix[i][j] = matrix[j][i] = -1

    if family == "B":
        matrix[rank - 1][rank - 2] = -2
    elif family == "C":
        matrix[rank - 2][rank - 1] = -2
    elif family == "F":
        matrix = [
            [2, -1, 0, 0],
            [-1, 2, -1, 0],
            [0, -2, 2, -1],
            [0, 0, -1, 2],
        ]
    elif family == "G":
        # alpha_1 is short, alpha_2 is long (Bourbaki numbering).
        matrix = [[2, -3], [-1, 2]]

    return tuple(tuple(row) for row in matrix)


def _parse_cartan_type(cartan_type: str | tuple[str, int]) -> tuple[str, int]:
    if isinstance(cartan_type, tuple):
        family, rank = cartan_type
        return str(family).upper(), index(rank)
    text = cartan_type.strip().upper().replace("_", "")
    if len(text) < 2 or not text[1:].isdigit():
        raise ValueError("Cartan type must look like 'A4', 'E6', or ('A', 4)")
    return text[0], int(text[1:])


@dataclass(frozen=True)
class RootSystem:
    """A finite irreducible root system in Bourbaki numbering."""

    family: str
    rank: int

    @classmethod
    def from_type(cls, cartan_type: str | tuple[str, int] | "RootSystem") -> "RootSystem":
        if isinstance(cartan_type, cls):
            return cartan_type
        family, rank = _parse_cartan_type(cartan_type)
        # Validate eagerly.
        _cartan_matrix(family, rank)
        return cls(family, rank)

    @cached_property
    def cartan(self) -> tuple[tuple[int, ...], ...]:
        # Rows are simple coroots and columns are simple roots:
        # A_ij = <alpha_j, alpha_i^vee>.
        return _cartan_matrix(self.family, self.rank)

    @cached_property
    def inverse_cartan(self) -> tuple[tuple[Fraction, ...], ...]:
        return _inverse(self.cartan)

    @cached_property
    def symmetrizer(self) -> tuple[int, ...]:
        """Minimal positive integers d_i with d_i A_ij = d_j A_ji."""
        values: list[Fraction | None] = [None] * self.rank
        for start in range(self.rank):
            if values[start] is not None:
                continue
            values[start] = Fraction(1)
            queue = deque([start])
            while queue:
                i = queue.popleft()
                for j in range(self.rank):
                    if self.cartan[i][j] == 0 or i == j:
                        continue
                    candidate = values[i] * Fraction(self.cartan[i][j], self.cartan[j][i])
                    if values[j] is None:
                        values[j] = candidate
                        queue.append(j)
                    elif values[j] != candidate:
                        raise ValueError("Cartan matrix is not symmetrizable")
        fractions = [value for value in values if value is not None]
        denominator = 1
        for value in fractions:
            denominator = _lcm(denominator, value.denominator)
        integers = [int(value * denominator) for value in fractions]
        divisor = 0
        for value in integers:
            divisor = gcd(divisor, value)
        return tuple(value // divisor for value in integers)

    @cached_property
    def root_gram(self) -> tuple[tuple[Fraction, ...], ...]:
        """Gram matrix of simple roots, up to one harmless common scalar."""
        return tuple(
            tuple(Fraction(self.symmetrizer[i] * self.cartan[i][j]) for j in range(self.rank))
            for i in range(self.rank)
        )

    @cached_property
    def roots(self) -> tuple[Root, ...]:
        roots: set[Root] = set()
        queue: deque[Root] = deque()
        for i in range(self.rank):
            simple = tuple(int(j == i) for j in range(self.rank))
            for root in (simple, tuple(-x for x in simple)):
                roots.add(root)
                queue.append(root)
        while queue:
            root = queue.popleft()
            for i in range(self.rank):
                pairing = sum(self.cartan[i][j] * root[j] for j in range(self.rank))
                reflected = list(root)
                reflected[i] -= pairing
                reflected_t = tuple(reflected)
                if reflected_t not in roots:
                    roots.add(reflected_t)
                    queue.append(reflected_t)
        return tuple(sorted(roots, key=lambda root: (sum(root), root)))

    @cached_property
    def positive_roots(self) -> tuple[Root, ...]:
        return tuple(root for root in self.roots if all(value >= 0 for value in root))

    def positive_roots_for(self, nodes: Iterable[int]) -> tuple[Root, ...]:
        node_set = {node - 1 for node in nodes}
        return tuple(
            root for root in self.positive_roots
            if all(value == 0 for i, value in enumerate(root) if i not in node_set)
        )

    def dynkin_components(self, nodes: Iterable[int]) -> tuple[tuple[int, ...], ...]:
        """Connected components of the Dynkin subdiagram induced by ``nodes``."""
        remaining = {int(node) for node in nodes}
        if any(node < 1 or node > self.rank for node in remaining):
            raise ValueError("Dynkin node is outside the diagram")
        components = []
        while remaining:
            start = min(remaining)
            component = {start}
            queue = deque([start])
            remaining.remove(start)
            while queue:
                node = queue.popleft()
                i = node - 1
                neighbours = {
                    other
                    for other in tuple(remaining)
                    if self.cartan[i][other - 1] or self.cartan[other - 1][i]
                }
                component.update(neighbours)
                remaining.difference_update(neighbours)
                queue.extend(sorted(neighbours))
            components.append(tuple(sorted(component)))
        return tuple(components)

    def _component_type(self, component: Iterable[int]) -> tuple[str, int]:
        """Classify an induced connected finite Dynkin subdiagram up to B/C duality."""
        nodes = tuple(component)
        rank = len(nodes)
        if rank == 1:
            return "A", 1

        adjacency = {node: set() for node in nodes}
        edge_multiplicities = {}
        for offset, left in enumerate(nodes):
            for right in nodes[offset + 1:]:
                multiplicity = abs(
                    self.cartan[left - 1][right - 1]
                    * self.cartan[right - 1][left - 1]
                )
                if multiplicity:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
                    edge_multiplicities[frozenset((left, right))] = multiplicity

        if any(value == 3 for value in edge_multiplicities.values()):
            if rank != 2:
                raise ArithmeticError("unexpected triple edge in a Dynkin subdiagram")
            return "G", 2

        double_edges = [edge for edge, value in edge_multiplicities.items() if value == 2]
        if double_edges:
            if len(double_edges) != 1:
                raise ArithmeticError("unexpected multiple double edges in a Dynkin subdiagram")
            endpoints = tuple(double_edges[0])
            if rank == 4 and all(len(adjacency[node]) == 2 for node in endpoints):
                return "F", 4
            # B_n and C_n have identical Weyl degrees, so their orientation is irrelevant here.
            return "B", rank

        branch_nodes = [node for node in nodes if len(adjacency[node]) == 3]
        if not branch_nodes:
            return "A", rank
        if len(branch_nodes) != 1:
            raise ArithmeticError("unexpected simply-laced Dynkin subdiagram")

        branch = branch_nodes[0]
        arm_lengths = []
        for neighbour in adjacency[branch]:
            length = 1
            previous, current = branch, neighbour
            while len(adjacency[current] - {previous}) == 1:
                next_node = next(iter(adjacency[current] - {previous}))
                previous, current = current, next_node
                length += 1
            arm_lengths.append(length)
        arms = tuple(sorted(arm_lengths))
        if arms[:2] == (1, 1):
            return "D", rank
        exceptional_arms = {
            (1, 2, 2): ("E", 6),
            (1, 2, 3): ("E", 7),
            (1, 2, 4): ("E", 8),
        }
        try:
            return exceptional_arms[arms]
        except KeyError as error:
            raise ArithmeticError(f"unrecognized Dynkin component with arm lengths {arms}") from error

    def weyl_degrees(self, nodes: Iterable[int] | None = None) -> tuple[int, ...]:
        """Invariant degrees of the Weyl group, or of a standard parabolic subgroup."""
        if nodes is None:
            return _weyl_degrees(self.family, self.rank)
        result = []
        for component in self.dynkin_components(nodes):
            result.extend(_weyl_degrees(*self._component_type(component)))
        return tuple(sorted(result))

    def weyl_poincare_polynomial(self, nodes: Iterable[int] | None = None) -> tuple[int, ...]:
        """Length-generating polynomial ``sum_w q^length(w)``."""
        polynomial = (1,)
        for degree in self.weyl_degrees(nodes):
            polynomial = _multiply_polynomials(polynomial, (1,) * degree)
        return polynomial

    def parabolic_poincare_polynomial(self, levi_nodes: Iterable[int]) -> tuple[int, ...]:
        """Length polynomial of minimal coset representatives ``W/W_L``."""
        return _divide_polynomials(
            self.weyl_poincare_polynomial(),
            self.weyl_poincare_polynomial(levi_nodes),
        )

    def root_to_weight(self, root: Iterable[int | Fraction]) -> tuple[Fraction, ...]:
        """Convert simple-root coordinates to ambient fundamental-weight coordinates."""
        vector = tuple(root)
        return _matvec(self.cartan, vector)

    def weight_to_root(self, weight: Iterable[int | Fraction]) -> tuple[Fraction, ...]:
        return _matvec(self.inverse_cartan, tuple(weight))

    def reflect_weight(self, weight: Iterable[int | Fraction], node: int) -> tuple[Fraction, ...]:
        i = node - 1
        vector = tuple(weight)
        coefficient = vector[i]
        return tuple(vector[j] - coefficient * self.cartan[j][i] for j in range(self.rank))

    def inner_roots(self, left: Iterable[int | Fraction], right: Iterable[int | Fraction]) -> Fraction:
        left_t, right_t = tuple(left), tuple(right)
        return sum(
            left_t[i] * self.root_gram[i][j] * right_t[j]
            for i in range(self.rank) for j in range(self.rank)
        )

    def inner_weights(self, left: Iterable[int | Fraction], right: Iterable[int | Fraction]) -> Fraction:
        return self.inner_roots(self.weight_to_root(left), self.weight_to_root(right))

    def inner_weight_root(self, weight: Iterable[int | Fraction], root: Iterable[int | Fraction]) -> Fraction:
        return self.inner_roots(self.weight_to_root(weight), root)

    def dominantize(
        self,
        weight: Iterable[int | Fraction],
        nodes: Iterable[int] | None = None,
        *,
        reject_walls: bool = False,
    ) -> tuple[tuple[Fraction, ...] | None, int]:
        """Move a weight to the chosen dominant chamber by simple reflections."""
        active = tuple(nodes) if nodes is not None else tuple(range(1, self.rank + 1))
        current = tuple(weight)
        length = 0
        while True:
            if reject_walls and any(current[node - 1] == 0 for node in active):
                return None, length
            negative = next((node for node in active if current[node - 1] < 0), None)
            if negative is None:
                return current, length
            current = self.reflect_weight(current, negative)
            length += 1

    def rho(self, nodes: Iterable[int] | None = None) -> tuple[Fraction, ...]:
        roots = self.positive_roots if nodes is None else self.positive_roots_for(nodes)
        total = [0] * self.rank
        for root in roots:
            for i, value in enumerate(root):
                total[i] += value
        return tuple(Fraction(value, 2) for value in self.root_to_weight(total))

    def weyl_dimension(self, highest_weight: Iterable[int], nodes: Iterable[int] | None = None) -> int:
        active = tuple(nodes) if nodes is not None else tuple(range(1, self.rank + 1))
        weight = tuple(index(value) for value in highest_weight)
        if len(weight) != self.rank:
            raise ValueError(f"expected a weight of length {self.rank}")
        if any(weight[node - 1] < 0 for node in active):
            raise ValueError(f"weight {list(weight)} is not dominant on nodes {active}")
        roots = self.positive_roots if nodes is None else self.positive_roots_for(active)
        rho = self.rho(None if nodes is None else active)
        shifted = tuple(Fraction(weight[i]) + rho[i] for i in range(self.rank))
        dimension = Fraction(1)
        for root in roots:
            dimension *= self.inner_weight_root(shifted, root) / self.inner_weight_root(rho, root)
        if dimension.denominator != 1:
            raise ArithmeticError(f"non-integral Weyl dimension {dimension}")
        return int(dimension)

    def borel_weil_bott(self, weight: Iterable[int]) -> dict | None:
        """Return the unique BWB cohomology representation, or ``None`` if singular."""
        weight_t = tuple(index(value) for value in weight)
        if len(weight_t) != self.rank:
            raise ValueError(f"expected a weight of length {self.rank}, got {len(weight_t)}")
        shifted = tuple(value + 1 for value in weight_t)  # rho has Dynkin labels (1,...,1)
        dominant, degree = self.dominantize(shifted, reject_walls=True)
        if dominant is None:
            return None
        highest = tuple(int(value - 1) for value in dominant)
        return {
            "weight": list(highest),
            "dimension": self.weyl_dimension(highest),
            "degree": degree,
            "multiplicity": 1,
        }


def _add_weights(left: Weight, right: Weight, scale: int = 1) -> Weight:
    return tuple(a + scale * b for a, b in zip(left, right))


def _convolve(left: Mapping[Weight, int], right: Mapping[Weight, int]) -> Character:
    result: defaultdict[Weight, int] = defaultdict(int)
    for weight_a, mult_a in left.items():
        if not mult_a:
            continue
        for weight_b, mult_b in right.items():
            if mult_b:
                result[_add_weights(weight_a, weight_b)] += mult_a * mult_b
    return {weight: mult for weight, mult in result.items() if mult}


class LeviRepresentationRing:
    """Character ring of the Levi quotient of a parabolic subgroup."""

    def __init__(self, root_system: RootSystem, levi_nodes: Iterable[int]):
        self.root_system = root_system
        self.nodes = tuple(sorted(set(levi_nodes)))
        if any(node < 1 or node > root_system.rank for node in self.nodes):
            raise ValueError("Levi node is outside the Dynkin diagram")

    @cached_property
    def positive_roots(self) -> tuple[Root, ...]:
        return self.root_system.positive_roots_for(self.nodes)

    @cached_property
    def rho(self) -> tuple[Fraction, ...]:
        return self.root_system.rho(self.nodes)

    @cached_property
    def dominance_coefficients(self) -> tuple[Fraction, ...]:
        """Coefficients of rho_L^vee in the basis of simple Levi coroots."""
        if not self.nodes:
            return ()
        indices = [node - 1 for node in self.nodes]
        submatrix = tuple(
            tuple(self.root_system.cartan[i][j] for j in indices)
            for i in indices
        )
        inverse_transpose = tuple(zip(*_inverse(submatrix)))
        return tuple(sum(row) for row in inverse_transpose)

    def validate_highest_weight(self, weight: Iterable[int]) -> Weight:
        result = tuple(index(value) for value in weight)
        if len(result) != self.root_system.rank:
            raise ValueError(
                f"expected {self.root_system.rank} fundamental-weight coefficients, "
                f"got {len(result)}"
            )
        bad = [node for node in self.nodes if result[node - 1] < 0]
        if bad:
            raise ValueError(
                f"weight {list(result)} is not Levi-dominant; negative coefficients "
                f"at uncrossed nodes {bad}"
            )
        return result

    def dimension(self, highest_weight: Iterable[int]) -> int:
        weight = self.validate_highest_weight(highest_weight)
        return self.root_system.weyl_dimension(weight, self.nodes)

    @lru_cache(maxsize=None)
    def _character_items(self, highest_weight: Weight) -> tuple[tuple[Weight, int], ...]:
        highest = self.validate_highest_weight(highest_weight)
        if not self.nodes:
            return ((highest, 1),)

        candidates: dict[Weight, int] = {highest: 0}
        queue: deque[Weight] = deque([highest])
        simple_weight_roots = {
            node: tuple(int(value) for value in self.root_system.root_to_weight(
                tuple(int(i == node - 1) for i in range(self.root_system.rank))
            ))
            for node in self.nodes
        }
        while queue:
            weight = queue.popleft()
            depth = candidates[weight]
            for node in self.nodes:
                string_length = weight[node - 1]
                if string_length <= 0:
                    continue
                root = simple_weight_roots[node]
                for step in range(1, string_length + 1):
                    lowered = _add_weights(weight, root, -step)
                    new_depth = depth + step
                    if lowered not in candidates:
                        candidates[lowered] = new_depth
                        queue.append(lowered)

        multiplicities: dict[Weight, int] = {highest: 1}
        highest_shifted = tuple(Fraction(highest[i]) + self.rho[i] for i in range(self.root_system.rank))
        highest_norm = self.root_system.inner_weights(highest_shifted, highest_shifted)
        root_weights = [
            (root, tuple(int(value) for value in self.root_system.root_to_weight(root)), sum(root))
            for root in self.positive_roots
        ]
        for weight, depth in sorted(candidates.items(), key=lambda item: item[1]):
            if weight == highest:
                continue
            shifted = tuple(Fraction(weight[i]) + self.rho[i] for i in range(self.root_system.rank))
            denominator = highest_norm - self.root_system.inner_weights(shifted, shifted)
            numerator = Fraction(0)
            for root, root_weight, root_height in root_weights:
                step = 1
                while depth - step * root_height >= 0:
                    above = _add_weights(weight, root_weight, step)
                    above_mult = multiplicities.get(above, 0)
                    if above_mult:
                        numerator += above_mult * self.root_system.inner_weight_root(above, root)
                    step += 1
            value = 2 * numerator / denominator if denominator else Fraction(0)
            if value.denominator != 1 or value < 0:
                raise ArithmeticError(
                    f"Freudenthal recurrence failed for {list(highest)} at {list(weight)}: {value}"
                )
            multiplicities[weight] = int(value)

        result = tuple(sorted(
            ((weight, mult) for weight, mult in multiplicities.items() if mult),
            key=lambda item: item[0],
        ))
        computed_dimension = sum(mult for _, mult in result)
        expected_dimension = self.dimension(highest)
        if computed_dimension != expected_dimension:
            raise ArithmeticError(
                f"incomplete character for {list(highest)}: got dimension "
                f"{computed_dimension}, expected {expected_dimension}"
            )
        return result

    def character(self, highest_weight: Iterable[int]) -> Character:
        highest = self.validate_highest_weight(highest_weight)
        return dict(self._character_items(highest))

    def dual_highest_weight(self, highest_weight: Iterable[int]) -> Weight:
        highest = self.validate_highest_weight(highest_weight)
        dominant, _ = self.root_system.dominantize(
            tuple(-value for value in highest), self.nodes
        )
        assert dominant is not None
        if any(value.denominator != 1 for value in dominant):
            raise ArithmeticError("dual highest weight is non-integral")
        return tuple(int(value) for value in dominant)

    def determinant_weight(self, highest_weight: Iterable[int]) -> Weight:
        character = self.character(highest_weight)
        return tuple(
            sum(mult * weight[i] for weight, mult in character.items())
            for i in range(self.root_system.rank)
        )

    def _dominance_score(self, weight: Weight) -> Fraction:
        return sum(
            coefficient * weight[node - 1]
            for node, coefficient in zip(self.nodes, self.dominance_coefficients)
        )

    def decompose_character(self, character: Mapping[Weight, int]) -> dict[Weight, int]:
        remaining = {weight: mult for weight, mult in character.items() if mult}
        result: defaultdict[Weight, int] = defaultdict(int)
        while remaining:
            dominant = [
                weight for weight, mult in remaining.items()
                if mult and all(weight[node - 1] >= 0 for node in self.nodes)
            ]
            if not dominant:
                raise ArithmeticError("character has no Levi-dominant maximal weight")
            highest = max(
                dominant,
                key=lambda weight: (
                    self._dominance_score(weight),
                    tuple(weight[node - 1] for node in self.nodes),
                    weight,
                ),
            )
            copies = remaining[highest]
            if copies <= 0:
                raise ArithmeticError(
                    f"virtual character is not effective at highest weight {list(highest)}"
                )
            result[highest] += copies
            for weight, mult in self.character(highest).items():
                new_value = remaining.get(weight, 0) - copies * mult
                if new_value:
                    remaining[weight] = new_value
                else:
                    remaining.pop(weight, None)
        return dict(result)

    def tensor_product(
        self, left: Iterable[int], right: Iterable[int]
    ) -> dict[Weight, int]:
        return self.decompose_character(_convolve(self.character(left), self.character(right)))

    def _power_character(self, highest_weight: Iterable[int], degree: int, *, wedge: bool) -> Character:
        degree = index(degree)
        if degree < 0:
            raise ValueError("power must be non-negative")
        highest = self.validate_highest_weight(highest_weight)
        rank = self.dimension(highest)
        if wedge and degree > rank:
            return {}
        zero = (0,) * self.root_system.rank
        pieces: list[Character] = [{zero: 1}] + [{} for _ in range(degree)]
        for weight, multiplicity in self.character(highest).items():
            updated = [dict(piece) for piece in pieces]
            for old_degree in range(degree + 1):
                for old_weight, old_mult in pieces[old_degree].items():
                    limit = min(multiplicity, degree - old_degree) if wedge else degree - old_degree
                    for count in range(1, limit + 1):
                        coefficient = (
                            comb(multiplicity, count) if wedge
                            else comb(multiplicity + count - 1, count)
                        )
                        new_weight = _add_weights(old_weight, weight, count)
                        updated[old_degree + count][new_weight] = (
                            updated[old_degree + count].get(new_weight, 0)
                            + old_mult * coefficient
                        )
            pieces = updated
        return pieces[degree]

    def symmetric_power(self, highest_weight: Iterable[int], degree: int) -> dict[Weight, int]:
        return self.decompose_character(self._power_character(highest_weight, degree, wedge=False))

    def exterior_power(self, highest_weight: Iterable[int], degree: int) -> dict[Weight, int]:
        character = self._power_character(highest_weight, degree, wedge=True)
        return self.decompose_character(character) if character else {}

    def schur_power(self, highest_weight: Iterable[int], partition: Iterable[int]) -> dict[Weight, int]:
        """Decompose a Schur functor using Jacobi--Trudi in the character ring."""
        highest_weight = self.validate_highest_weight(highest_weight)
        parts = tuple(index(value) for value in partition)
        if any(value < 0 for value in parts):
            raise ValueError("Schur partition must have non-negative parts")
        if any(parts[i] < parts[i + 1] for i in range(len(parts) - 1)):
            raise ValueError("Schur partition must be weakly decreasing")
        shape = tuple(value for value in parts if value > 0)
        if any(shape[i] < shape[i + 1] for i in range(len(shape) - 1)):
            raise ValueError("Schur partition must be weakly decreasing")
        if not shape:
            return {(0,) * self.root_system.rank: 1}
        if len(shape) > self.dimension(highest_weight):
            return {}
        length = len(shape)
        required = {
            shape[i] - i + j
            for i in range(length) for j in range(length)
            if shape[i] - i + j >= 0
        }
        complete: dict[int, Character] = {
            degree: self._power_character(highest_weight, degree, wedge=False)
            for degree in required
        }
        complete[0] = {(0,) * self.root_system.rank: 1}
        total: defaultdict[Weight, int] = defaultdict(int)
        for permutation in permutations(range(length)):
            inversions = sum(
                permutation[i] > permutation[j]
                for i in range(length) for j in range(i + 1, length)
            )
            term: Character = {(0,) * self.root_system.rank: 1}
            for i, j in enumerate(permutation):
                degree = shape[i] - i + j
                if degree < 0:
                    term = {}
                    break
                term = _convolve(term, complete[degree])
            sign = -1 if inversions % 2 else 1
            for weight, mult in term.items():
                total[weight] += sign * mult
        return self.decompose_character({weight: mult for weight, mult in total.items() if mult})


__all__ = ["Character", "LeviRepresentationRing", "RootSystem", "Weight"]
