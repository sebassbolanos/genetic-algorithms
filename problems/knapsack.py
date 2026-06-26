import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from core.genetic_algorithm import GeneticAlgorithm

def _is_feasible(selected_indices: list[int], weights: list[list[int]], capacities: list[int]) -> bool:
   
    num_dimensions = len(capacities)
    for d in range(num_dimensions):
        if sum(weights[i][d] for i in selected_indices) > capacities[d]:
            return False
    return True

def knapsack_recursive(
    values: list[int],
    weights: list[list[int]],
    capacities: list[int]
) -> tuple[int, list[int]]:

    n = len(values)
    num_dimensions = len(capacities)

    def _helper(index: int, remaining: list[int]) -> tuple[int, list[int]]:
        
        if index == n:
            return 0, []

        skip_val, skip_items = _helper(index + 1, remaining)

        fits = all(remaining[d] >= weights[index][d] for d in range(num_dimensions))
        if fits:
            new_remaining = [remaining[d] - weights[index][d] for d in range(num_dimensions)]
            include_val, include_items = _helper(index + 1, new_remaining)
            include_val += values[index]

            if include_val > skip_val:
                return include_val, [index] + include_items

        return skip_val, skip_items

    best_value, best_items = _helper(0, capacities[:])
    return best_value, best_items


def knapsack_dp(
    values: list[int],
    weights: list[list[int]],
    capacities: list[int]
) -> tuple[int, list[int]]:

    n = len(values)
    num_dimensions = len(capacities)
    memo = {}

    def _helper(index: int, remaining: tuple) -> tuple[int, list[int]]:
       
        if index == n:
            return 0, []

        key = (index, remaining)
        if key in memo:
            return memo[key]

        skip_val, skip_items = _helper(index + 1, remaining)

        fits = all(remaining[d] >= weights[index][d] for d in range(num_dimensions))
        best_val, best_items = skip_val, skip_items

        if fits:
            new_remaining = tuple(remaining[d] - weights[index][d] for d in range(num_dimensions))
            include_val, include_items = _helper(index + 1, new_remaining)
            include_val += values[index]

            if include_val > skip_val:
                best_val = include_val
                best_items = [index] + include_items

        memo[key] = (best_val, best_items)
        return best_val, best_items

    best_value, best_items = _helper(0, tuple(capacities))
    return best_value, best_items

def knapsack_ga(
    values: list[int],
    weights: list[list[int]],
    capacities: list[int],
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
) -> tuple[int, list[int], list[float]]:
    n = len(values)
    num_dimensions = len(capacities)

    def fitness_fn(chromosome: list[int]) -> float:
    
        selected = [i for i, bit in enumerate(chromosome) if bit == 1]

        total_value = sum(values[i] for i in selected)

        total_penalty = 0
        for d in range(num_dimensions):
            used = sum(weights[i][d] for i in selected)
            excess = max(0, used - capacities[d])
            total_penalty += excess

        if total_penalty > 0:
            return total_value / (1.0 + total_penalty * 10)

        return float(total_value)

    ga = GeneticAlgorithm(
        fitness_fn=fitness_fn,
        chromosome_length=n,
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

    best_items = [i for i, bit in enumerate(best_chromosome) if bit == 1]
    best_value = sum(values[i] for i in best_items)

    return best_value, best_items, history

if __name__ == "__main__":
    values =     [10,  6,  5,  8,  3,  7,  4,  9]
    weights = [
        [3,  2,  4,  1,  5,  3,  2,  4],   # dimension 0: weight
        [2,  3,  1,  5,  2,  4,  3,  2],   # dimension 1: volume
        [4,  1,  3,  2,  3,  2,  5,  3],   # dimension 2: cost
    ]
    capacities = [10, 10, 10]

    weights_by_item = [[weights[d][i] for d in range(len(weights))] for i in range(len(values))]

    print("=" * 55)
    print(f"Items:      {len(values)}")
    print(f"Dimensions: {len(capacities)}")
    print(f"Capacities: {capacities}")
    print(f"Values:     {values}")
    print("=" * 55)

    best_val, best_items = knapsack_recursive(values, weights_by_item, capacities)
    print(f"\n[Recursive]  Value: {best_val}  |  Items: {best_items}")
    for i in best_items:
        print(f"             Item {i}: value={values[i]}  weights={weights_by_item[i]}")

    best_val_dp, best_items_dp = knapsack_dp(values, weights_by_item, capacities)
    print(f"\n[DP]         Value: {best_val_dp}  |  Items: {best_items_dp}")
    for i in best_items_dp:
        print(f"             Item {i}: value={values[i]}  weights={weights_by_item[i]}")

    best_val_ga, best_items_ga, history = knapsack_ga(values, weights_by_item, capacities)
    feasible = _is_feasible(best_items_ga, weights_by_item, capacities)
    print(f"\n[GA]         Value: {best_val_ga}  |  Items: {best_items_ga}  |  Feasible: {feasible}")
    for i in best_items_ga:
        print(f"             Item {i}: value={values[i]}  weights={weights_by_item[i]}")
    print(f"             Generations run: {len(history)}")