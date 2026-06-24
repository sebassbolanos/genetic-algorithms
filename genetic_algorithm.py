import random
from selection import roulette_selection, ranking_selection, tournament_selection
from crossover import one_point_crossover, two_point_crossover, uniform_crossover
from mutation import bit_flip_mutation, swap_mutation


class GeneticAlgorithm:
    """
    A modular Genetic Algorithm framework.
    The fitness function is passed as a parameter, making it reusable
    for any optimization problem by only changing the fitness function.
    """

    def __init__(
        self,
        fitness_fn,
        chromosome_length,
        population_size=100,
        selection_method="tournament",
        tournament_size=3,
        crossover_method="one_point",
        crossover_rate=0.8,
        mutation_method="bit_flip",
        mutation_rate=0.01,
        elitism=0.1,
        max_generations=500,
        patience=50,
    ):
        """
        Args:
            fitness_fn: function that receives a chromosome and returns a fitness value.
            chromosome_length: number of genes in each chromosome.
            population_size: number of individuals in the population.
            selection_method: 'roulette', 'ranking', or 'tournament'.
            tournament_size: size of tournament (only used if selection_method='tournament').
            crossover_method: 'one_point', 'two_point', or 'uniform'.
            crossover_rate: probability of performing crossover.
            mutation_method: 'bit_flip' or 'swap'.
            mutation_rate: probability of mutation per gene (bit_flip) or per chromosome (swap).
            elitism: proportion of best individuals copied directly to next generation.
            max_generations: maximum number of generations to run.
            patience: stop early if no improvement after this many generations.
        """
        self.fitness_fn = fitness_fn
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.crossover_method = crossover_method
        self.crossover_rate = crossover_rate
        self.mutation_method = mutation_method
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.max_generations = max_generations
        self.patience = patience

    def _initialize_population(self):
        """Creates a random binary population."""
        return [
            [random.randint(0, 1) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

    def _evaluate(self, population):
        """Evaluates fitness for each chromosome in the population."""
        return [self.fitness_fn(chromosome) for chromosome in population]

    def _select_parent(self, population, fitnesses):
        """Selects one parent using the configured selection method."""
        if self.selection_method == "roulette":
            return roulette_selection(population, fitnesses)
        elif self.selection_method == "ranking":
            return ranking_selection(population, fitnesses)
        else:
            return tournament_selection(population, fitnesses, self.tournament_size)

    def _crossover(self, parent1, parent2):
        """Performs crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        if self.crossover_method == "one_point":
            return one_point_crossover(parent1, parent2)
        elif self.crossover_method == "two_point":
            return two_point_crossover(parent1, parent2)
        else:
            return uniform_crossover(parent1, parent2)

    def _mutate(self, chromosome):
        """Applies mutation to a chromosome."""
        if self.mutation_method == "bit_flip":
            return bit_flip_mutation(chromosome, self.mutation_rate)
        else:
            return swap_mutation(chromosome, self.mutation_rate)

    def run(self):
        """
        Runs the genetic algorithm.

        Returns:
            best_chromosome: the best solution found.
            best_fitness: fitness value of the best solution.
            history: list of best fitness per generation.
        """
        population = self._initialize_population()
        fitnesses = self._evaluate(population)

        best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_chromosome = population[best_idx][:]
        best_fitness = fitnesses[best_idx]
        history = [best_fitness]
        no_improvement = 0

        elite_count = max(1, int(self.elitism * self.population_size))

        for generation in range(self.max_generations):
            # Elitism: carry best individuals directly to next generation
            sorted_pairs = sorted(
                zip(fitnesses, population), key=lambda x: x[0], reverse=True
            )
            new_population = [chromo[:] for _, chromo in sorted_pairs[:elite_count]]

            # Fill rest of population with children
            while len(new_population) < self.population_size:
                parent1 = self._select_parent(population, fitnesses)
                parent2 = self._select_parent(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            fitnesses = self._evaluate(population)

            # Track best solution
            gen_best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            gen_best_fitness = fitnesses[gen_best_idx]

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_chromosome = population[gen_best_idx][:]
                no_improvement = 0
            else:
                no_improvement += 1

            history.append(best_fitness)

            # Early stopping
            if no_improvement >= self.patience:
                print(f"Early stopping at generation {generation + 1}")
                break

        return best_chromosome, best_fitness, history
        