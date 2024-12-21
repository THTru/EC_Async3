from itertools import count
import random
import os
from deap import base, creator, tools, algorithms

# Define the problem as a maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize the toolbox
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_tile", random.randint, 0, 4)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_tile, 1600)

# Define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the fitness function

def evalMap(individual):
    size = 16
    mountain_score = 0
    lake_score = 0
    isle_score = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    lake_start = size // 8
    lake_end = lake_start + (size * 3) // 4

    isle_start = size // 4
    isle_end = isle_start + size // 2

    # Convert the individual list to a 2D map
    map_2d = [individual[i * size:(i + 1) * size] for i in range(size)]

    for i in range(size):
        for j in range(size):
            if map_2d[i][j] == 1:
                count_1 += 1
            if map_2d[i][j] == 2:
                count_2 += 1
            if map_2d[i][j] == 3:
                count_3 += 1
            if map_2d[i][j] == 4:
                count_4 += 1
            if map_2d[i][j] == 0:  # Mountain
                mountain_score += 1
            if lake_start <= i < lake_end and lake_start <= j < lake_end:
                if map_2d[i][j] == 1 or map_2d[i][j] == 4:  # Lake
                    lake_score += 2
            if isle_start <= i < isle_end and isle_start <= j < isle_end:
                if map_2d[i][j] == 2 or map_2d[i][j] == 3:  # Isle
                    isle_score += 4
    lake_score = lake_score * (1 + 1.0 / (abs(count_1 - 8 * count_4) + 1))
    isle_score = isle_score * (1 + 1.0 / (abs(count_2 - 8 * count_3) + 1))

    fitness = mountain_score + lake_score + isle_score
    return fitness,

# Mutation strategy
def custom_mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            gene = individual[i]
            if gene == 0:
                individual[i] = random.choices([1, 2], [0.5, 0.5])[0]
            elif gene == 1:
                individual[i] = random.choices([0, 2, 4], [0.4, 0.4, 0.2])[0]
            elif gene == 2:
                individual[i] = random.choices([0, 2, 3], [0.4, 0.4, 0.2])[0]
            elif gene == 3:
                individual[i] = random.choices([0, 1, 2], [0.3, 0.3, 0.4])[0]
            elif gene == 4:
                individual[i] = random.choices([0, 1, 2], [0.3, 0.4, 0.3])[0]
    return individual,

# Register the genetic operators
toolbox.register("evaluate", evalMap)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=5)

# Genetic Algorithm parameters
population_size = 500
probability_crossover = 0.5
probability_mutation = 0.05
number_of_generations = 2000
num_elites = 10  # Number of elite individuals to keep

# Create an initial population
population = toolbox.population(n=population_size)

# Apply the genetic algorithm with elitism
for gen in range(number_of_generations):
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population) - num_elites)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < probability_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probability_mutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the elite individuals
    elites = tools.selBest(population, num_elites)
    # Add the elite individuals to the offspring
    offspring.extend(elites)

    # Replace the old population with the new offspring
    population[:] = offspring

    if (gen + 1) % 200 == 0:
        fits = [ind.fitness.values[0] for ind in population]

        best_individual = tools.selBest(population, 1)[0]

        size = 16
        map_2d = [best_individual[i * size:(i + 1) * size] for i in range(size)]

        filename = 'default_' + str((gen + 1) // 200) + '.map'
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                pass

        with open(filename, 'w') as f:
            for row in map_2d:
                f.write(''.join(map(str, row)) + '\n')