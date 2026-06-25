"""
partition.py

Solves the Partition Problem using three approaches:
  1. Recursive (exhaustive)
  2. Dynamic Programming (top-down with memoization)
  3. Genetic Algorithm (using the GA framework)

Problem:
    Given a list of numbers, determine if it can be partitioned into two
    subsets with equal sum. The GA version minimizes the difference
    between the two subsets' sums.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from genetic_algorithm import GeneticAlgorithm



# 1. RECURSIVE (exhaustive)


def partition_recursive(numbers: list[int]) -> tuple[bool, list[int], list[int]]:
    total = sum(numbers)
    if total % 2 != 0:
        return False, [], []

    target = total // 2

    def _helper(index: int, remaining: int) -> list[int] | None:
        if remaining == 0:
            return []
        if index == len(numbers) or remaining < 0:
            return None

        include = _helper(index + 1, remaining - numbers[index])
        if include is not None:
            return [numbers[index]] + include

        return _helper(index + 1, remaining)

    subset_a = _helper(0, target)
    if subset_a is not None:
        subset_b = numbers[:]
        for elem in subset_a:
            subset_b.remove(elem)
        return True, subset_a, subset_b

    return False, [], []


# 2. DYNAMIC PROGRAMMING (top-down / memoization)

def partition_dp(numbers: list[int]) -> tuple[bool, list[int], list[int]]:
    total = sum(numbers)
    if total % 2 != 0:
        return False, [], []

    target = total // 2
    memo: dict = {}

    def _helper(index: int, remaining: int) -> list[int] | None:
        if remaining == 0:
            return []
        if index == len(numbers) or remaining < 0:
            return None

        key = (index, remaining)
        if key in memo:
            return memo[key]

        include = _helper(index + 1, remaining - numbers[index])
        if include is not None:
            memo[key] = [numbers[index]] + include
            return memo[key]

        exclude = _helper(index + 1, remaining)
        memo[key] = exclude
        return memo[key]

    subset_a = _helper(0, target)
    if subset_a is not None:
        subset_b = numbers[:]
        for elem in subset_a:
            subset_b.remove(elem)
        return True, subset_a, subset_b

    return False, [], []


# 3. GENETIC ALGORITHM


def partition_ga(
    numbers: list[int],
    population_size: int = 100,
    selection_method: str = "tournament",
    tournament_size: int = 3,
    crossover_method: str = "one_point",
    crossover_rate: float = 0.8,
    mutation_method: str = "bit_flip",
    mutation_rate: float = 0.01,
    elitism: float = 0.1,
    max_generations: int = 500,
    patience: int = 50,
) -> tuple[list[int], list[int], int, list[float]]:

    def fitness_fn(chromosome: list[int]) -> float:
        sum_a = sum(n for n, bit in zip(numbers, chromosome) if bit == 1)
        sum_b = sum(n for n, bit in zip(numbers, chromosome) if bit == 0)
        diff = abs(sum_a - sum_b)
        return 1.0 / (1.0 + diff)

    ga = GeneticAlgorithm(
        fitness_fn=fitness_fn,
        chromosome_length=len(numbers),
        population_size=population_size,
        selection_method=selection_method,
        tournament_size=tournament_size,
        crossover_method=crossover_method,
        crossover_rate=crossover_rate,
        mutation_method=mutation_method,
        mutation_rate=mutation_rate,
        elitism=elitism,
        max_generations=max_generations,
        patience=patience,
    )

    best_chromosome, best_fitness, history = ga.run()
    subset_a = [n for n, bit in zip(numbers, best_chromosome) if bit == 1]
    subset_b = [n for n, bit in zip(numbers, best_chromosome) if bit == 0]
    difference = abs(sum(subset_a) - sum(subset_b))

    return subset_a, subset_b, difference, history



# QUICK TEST


if __name__ == "__main__":
    numbers = [1, 5, 11, 5, 3, 8, 2, 7]

    print("=" * 50)
    print(f"Numbers: {numbers}")
    print(f"Total:   {sum(numbers)}")
    print("=" * 50)

    found, a, b = partition_recursive(numbers)
    print(f"\n[Recursive]  Found: {found}")
    if found:
        print(f"             A={a} sum={sum(a)}  |  B={b} sum={sum(b)}")

    found, a, b = partition_dp(numbers)
    print(f"[DP]         Found: {found}")
    if found:
        print(f"             A={a} sum={sum(a)}  |  B={b} sum={sum(b)}")

    a, b, diff, history = partition_ga(numbers)
    print(f"[GA]         Difference: {diff}")
    print(f"             A={a} sum={sum(a)}  |  B={b} sum={sum(b)}")
    print(f"             Generations run: {len(history)}")