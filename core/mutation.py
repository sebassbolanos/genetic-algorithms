import random


def bit_flip_mutation(chromosome, mutation_rate):
    """
    Performs bit-flip mutation on a binary chromosome.
    Each gene is flipped (0->1 or 1->0) with probability mutation_rate.

    Args:
        chromosome: list of bits (0s and 1s).
        mutation_rate: probability of flipping each bit.

    Returns:
        Mutated chromosome.
    """
    mutated = chromosome[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]  # flip: 0->1 or 1->0
    return mutated


def swap_mutation(chromosome, mutation_rate):
    """
    Performs swap mutation on a chromosome.
    With probability mutation_rate, two random positions are selected and swapped.

    Args:
        chromosome: list of genes.
        mutation_rate: probability of performing a swap.

    Returns:
        Mutated chromosome.
    """
    mutated = chromosome[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

