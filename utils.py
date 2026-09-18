"""
utils.py

Small standalone utility functions used across the library.
Kept in a separate module to avoid cluttering the main files.
"""

def has_repeats(lst):
    """
    check whether a given list has repeated entries
    """
    seen = set()
    for item in lst:
        if item in seen:
            return True
        seen.add(item)
    return False

def is_decreasing(lst):
    """
    Return True if the list of integers `lst` is ordered decreasingly,
    otherwise False.
    """
    return all(lst[i] > lst[i+1] for i in range(len(lst) - 1))


def swap_first_increase(lst):
    """
    If `lst` is not nonincreasing, find the first index i where lst[i] < lst[i+1],
    swap lst[i] and lst[i+1], and return True.
    If no such pair exists (i.e. lst is nonincreasing), do nothing and return False.
    """
    for i in range(len(lst) - 1):
        if lst[i] < lst[i + 1]:
            # swap the first increasing neighboring entries
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            return True
    return False

def add_rho(a):
    """
    Given a list of integers a = [a1, a2, ..., an],
    returns [a1 + n, a2 + (n-1), ..., a_{n-1} + 2, a_n + 1].
    """
    n = len(a)
    # b[i] = a[i] + (n - i)
    return [a[i] + (n - i) for i in range(n)]

def subtract_rho(a):
    """
    Given a list of integers a = [a1, a2, ..., an],
    returns [a1 - n, a2 - (n-1), ..., a_{n-1} - 2, a_n - 1].
    """
    n = len(a)
    # b[i] = a[i] - (n - i)
    return [a[i] - (n - i) for i in range(n)]

def normalize_partition(a):
    """
    Given a list of nonnegative integers `a = [a1, a2, ..., an]`,
    subtract the same amount from each entry so that the smallest entry becomes zero.
    Returns the new list.
    """
    if not a:
        return []
    m = min(a)
    return [x - m for x in a]

from fractions import Fraction

def weyl_dim(partition):
    """
    Compute the dimension of the irreducible GL_n representation
    with highest weight given by `partition` using the Weyl dimension formula.
    
    Args:
        partition (list of int): A list [λ1, λ2, ..., λn] of nonnegative integers.
        
    Returns:
        int: The dimension of the representation.
    """
    n = len(partition)
    # Work in exact rationals to avoid floating‐point error
    dim = Fraction(1, 1)
    for i in range(n):
        for j in range(i+1, n):
            num = partition[i] - partition[j] + (j - i)
            den = j - i
            dim *= Fraction(num, den)
    return int(dim)


def flatten(my_list):
    """
    flattens a nested list, e.g. flatten([[1,2,[3]], 4, [5,6]]) = [1,2,3,4,5,6]
    """
    out = []
    for item in my_list:
        if not isinstance(item, list):
            out.append(item)
        else:
            out.extend(flatten(item))
    return out