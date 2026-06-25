import random


def one_point_crossover(parent1, parent2):
    """
    Performs one-point crossover between two parents.
    A random point is chosen; genes before it come from parent1,
    genes after come from parent2.

    Args:
        parent1: first chromosome (list of bits).
        parent2: second chromosome (list of bits).

    Returns:
        Two children as a tuple (child1, child2).
    """
    n = len(parent1)
    point = random.randint(1, n - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Performs two-point crossover between two parents.
    Two random points are chosen; the segment between them is swapped.

    Args:
        parent1: first chromosome (list of bits).
        parent2: second chromosome (list of bits).

    Returns:
        Two children as a tuple (child1, child2).
    """
    n = len(parent1)
    point1 = random.randint(1, n - 2)
    point2 = random.randint(point1 + 1, n - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


def uniform_crossover(parent1, parent2, prob=0.5):
    """
    Performs uniform crossover between two parents.
    Each gene is independently taken from either parent with probability prob.

    Args:
        parent1: first chromosome (list of bits).
        parent2: second chromosome (list of bits).
        prob: probability of taking gene from parent1 (default 0.5).

    Returns:
        Two children as a tuple (child1, child2).
    """
    child1 = []
    child2 = []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < prob:
            child1.append(g1)
            child2.append(g2)
        else:
            child1.append(g2)
            child2.append(g1)
    return child1, child2
