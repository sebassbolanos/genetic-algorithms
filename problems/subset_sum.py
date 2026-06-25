"""
subset_sum.py

Solves the Subset Sum problem using three approaches:
  1. Recursive (exhaustive)
  2. Dynamic Programming (top-down with memoization)
  3. Genetic Algorithm (using the GA framework)

Problem:
    Given a list of numbers and a target value, determine if any subset
    of the numbers sums exactly to the target. The GA version finds the
    subset whose sum is closest to the target.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from genetic_algorithm import GeneticAlgorithm



# 1. RECURSIVE (exhaustive)


def subset_sum_recursive(numbers: list[int], target: int) -> tuple[bool, list[int]]:
    def _helper(index: int, remaining: int) -> list[int] | None:
        if remaining == 0:
            return []
        if index == len(numbers) or remaining < 0:
            return None

        include = _helper(index + 1, remaining - numbers[index])
        if include is not None:
            return [numbers[index]] + include

        return _helper(index + 1, remaining)

    result = _helper(0, target)
    if result is not None:
        return True, result
    return False, []



# 2. DYNAMIC PROGRAMMING (top-down / memoization)


def subset_sum_dp(numbers: list[int], target: int) -> tuple[bool, list[int]]:
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

    result = _helper(0, target)
    if result is not None:
        return True, result
    return False, []



# 3. GENETIC ALGORITHM


def subset_sum_ga(
    numbers: list[int],
    target: int,
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
) -> tuple[list[int], int, int, list[float]]:

    def fitness_fn(chromosome: list[int]) -> float:
        total = sum(n for n, bit in zip(numbers, chromosome) if bit)
        distance = abs(total - target)
        return 1.0 / (1.0 + distance)

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
    selected_numbers = [n for n, bit in zip(numbers, best_chromosome) if bit]
    achieved_sum = sum(selected_numbers)

    return selected_numbers, achieved_sum, target, history


# QUICK TEST


if __name__ == "__main__":
    numbers = [3, 7, 1, 8, 5, 12, 4, 9]
    target = 20

    print("=" * 50)
    print(f"Numbers: {numbers}")
    print(f"Target:  {target}")
    print("=" * 50)

    found, subset = subset_sum_recursive(numbers, target)
    print(f"\n[Recursive]  Found: {found}  |  Subset: {subset}  |  Sum: {sum(subset)}")

    found, subset = subset_sum_dp(numbers, target)
    print(f"[DP]         Found: {found}  |  Subset: {subset}  |  Sum: {sum(subset)}")

    selected, achieved, tgt, history = subset_sum_ga(numbers, target)
    exact = achieved == tgt
    print(f"[GA]         Exact: {exact}  |  Subset: {selected}  |  Sum: {achieved}  |  Target: {tgt}")
    print(f"             Generations run: {len(history)}")