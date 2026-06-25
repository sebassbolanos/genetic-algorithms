import random


def roulette_selection(population, fitnesses):
    """
    Selects a parent using roulette wheel (fitness-proportionate) selection.
    Individuals with higher fitness have a higher probability of being selected.

    Args:
        population: list of chromosomes.
        fitnesses: list of fitness values (must be >= 0) corresponding to each chromosome.

    Returns:
        One selected chromosome.
    """
    total = sum(fitnesses)

    if total == 0:
        return random.choice(population)

    pick = random.uniform(0, total)
    accumulated = 0
    for chromosome, fitness in zip(population, fitnesses):
        accumulated += fitness
        if accumulated >= pick:
            return chromosome

    return population[-1]


def ranking_selection(population, fitnesses):
    """
    Selects a parent using ranking selection.
    Individuals are ranked by fitness; selection probability is proportional to rank,
    not raw fitness value. This avoids dominance by very high fitness individuals.

    Args:
        population: list of chromosomes.
        fitnesses: list of fitness values corresponding to each chromosome.

    Returns:
        One selected chromosome.
    """
    paired = sorted(zip(fitnesses, population), key=lambda x: x[0])

    n = len(paired)
    ranks = list(range(1, n + 1))
    total_rank = sum(ranks)

    pick = random.uniform(0, total_rank)
    accumulated = 0
    for rank, (fitness, chromosome) in zip(ranks, paired):
        accumulated += rank
        if accumulated >= pick:
            return chromosome

    return paired[-1][1]


def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Selects a parent using tournament selection.
    Randomly picks `tournament_size` individuals and returns the one with best fitness.

    Args:
        population: list of chromosomes.
        fitnesses: list of fitness values corresponding to each chromosome.
        tournament_size: number of individuals competing in each tournament.

    Returns:
        One selected chromosome.
    """
    tournament_size = min(tournament_size, len(population))

    indices = random.sample(range(len(population)), tournament_size)
    best_index = max(indices, key=lambda i: fitnesses[i])
    return population[best_index]