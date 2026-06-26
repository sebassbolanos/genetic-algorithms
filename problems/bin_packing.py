import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from core.genetic_algorithm import GeneticAlgorithm


def _count_bins(assignment, items, capacity):
  
    bins = {}
    for item, bin_id in zip(items, assignment):
        bins[bin_id] = bins.get(bin_id, 0) + item

    valid = all(total <= capacity for total in bins.values())
    return len(bins), valid


def bin_packing_recursive(items: list[int], capacity: int) -> tuple[int, list[int]]:

    n = len(items)
    best = [n]  # worst case: one bin per item
    best_assignment = list(range(n))

    def _helper(index: int, bins: list[int], assignment: list[int]):
       
        if index == n:
            used = len([b for b in bins if b > 0])
            distinct = len(set(assignment))
            if distinct < best[0]:
                best[0] = distinct
                best_assignment[:] = assignment[:]
            return

        item = items[index]
        tried_new = False

        for bin_id in range(len(bins)):
            if bins[bin_id] + item <= capacity:
                bins[bin_id] += item
                assignment.append(bin_id)
                _helper(index + 1, bins, assignment)
                assignment.pop()
                bins[bin_id] -= item

            if not tried_new:
                tried_new = True
                new_id = len(bins)
                bins.append(item)
                assignment.append(new_id)
                _helper(index + 1, bins, assignment)
                assignment.pop()
                bins.pop()

    _helper(0, [], [])
    return best[0], best_assignment


def bin_packing_dp(items: list[int], capacity: int) -> tuple[int, list[int]]:

    n = len(items)
    memo = {}

    def _helper(index: int, bin_loads: tuple) -> tuple[int, list[int]]:
      
        if index == n:
            return len(bin_loads), []

        key = (index, bin_loads)
        if key in memo:
            return memo[key]

        item = items[index]
        best_count = n + 1
        best_assign = []

        loads = list(bin_loads)

        for i in range(len(loads)):
            if loads[i] + item <= capacity:
                loads[i] += item
                new_loads = tuple(sorted(loads))
                count, assign = _helper(index + 1, new_loads)
                # Map sorted bin back to original index i
                total = count
                if total < best_count:
                    best_count = total
                    best_assign = [i] + assign
                loads[i] -= item

        loads.append(item)
        new_loads = tuple(sorted(loads))
        count, assign = _helper(index + 1, new_loads)
        total = count
        if total < best_count:
            best_count = total
            best_assign = [len(bin_loads)] + assign
        loads.pop()

        memo[key] = (best_count, best_assign)
        return best_count, best_assign

    min_bins, assignment = _helper(0, ())
    return min_bins, assignment



def bin_packing_ga(
    items: list[int],
    capacity: int,
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
) -> tuple[int, list[list[int]], list[float]]:
   
    n = len(items)
  
    import math
    num_bits = max(1, math.ceil(math.log2(n + 1)))
    chromosome_length = n * num_bits

    def decode(chromosome: list[int]) -> list[int]:
        assignment = []
        for i in range(n):
            bits = chromosome[i * num_bits:(i + 1) * num_bits]
            bin_id = int(''.join(map(str, bits)), 2) % n
            assignment.append(bin_id)
        return assignment

    def fitness_fn(chromosome: list[int]) -> float:
       
        assignment = decode(chromosome)
        bin_loads = {}
        for item, bin_id in zip(items, assignment):
            bin_loads[bin_id] = bin_loads.get(bin_id, 0) + item

        bins_used = len(bin_loads)
        violations = sum(
            max(0, load - capacity) for load in bin_loads.values()
        )

        if violations > 0:
            return 1.0 / (1.0 + bins_used + violations * 10)

        return 1.0 / (1.0 + bins_used)

    ga = GeneticAlgorithm(
        fitness_fn=fitness_fn,
        chromosome_length=chromosome_length,
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
    assignment = decode(best_chromosome)

    # Build bin contents
    bin_contents = {}
    for item, bin_id in zip(items, assignment):
        bin_contents.setdefault(bin_id, []).append(item)

    bins_used = len(bin_contents)
    bins_list = list(bin_contents.values())

    return bins_used, bins_list, history


if __name__ == "__main__":
    items = [6, 3, 4, 7, 2, 5, 1, 8]
    capacity = 10

    print("=" * 50)
    print(f"Items:    {items}")
    print(f"Capacity: {capacity}")
    print(f"Total items: {len(items)}")
    print("=" * 50)

    min_bins, assignment = bin_packing_recursive(items, capacity)
    print(f"\n[Recursive]  Bins used: {min_bins}")
    bin_view = {}
    for item, b in zip(items, assignment):
        bin_view.setdefault(b, []).append(item)
    for b, content in sorted(bin_view.items()):
        print(f"             Bin {b}: {content}  sum={sum(content)}")

    min_bins_dp, assignment_dp = bin_packing_dp(items, capacity)
    print(f"\n[DP]         Bins used: {min_bins_dp}")
    bin_view_dp = {}
    for item, b in zip(items, assignment_dp):
        bin_view_dp.setdefault(b, []).append(item)
    for b, content in sorted(bin_view_dp.items()):
        print(f"             Bin {b}: {content}  sum={sum(content)}")

    bins_used, bins_list, history = bin_packing_ga(items, capacity)
    print(f"\n[GA]         Bins used: {bins_used}")
    for i, content in enumerate(bins_list):
        print(f"             Bin {i}: {content}  sum={sum(content)}")
    print(f"             Generations run: {len(history)}")