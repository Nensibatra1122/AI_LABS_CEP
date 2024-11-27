import numpy as np
import pandas as pd
import random

# Define the function to maximize
def fitness(x):
    return 40 * x - 0.1*x ** 2-10

# Load initial population from CSV
def initialize_population_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data.iloc[:, 0].values.tolist()

# Selection: Tournament Selection
def select(population, fitnesses):
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    return parent1 if fitnesses[population.index(parent1)] > fitnesses[population.index(parent2)] else parent2

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    alpha = random.random()
    return alpha * parent1 + (1 - alpha) * parent2

# Mutation: Small random adjustment
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        individual += np.random.normal(0, 1)
        individual = max(min(individual, upper_bound), lower_bound)
    return individual

# Genetic Algorithm
def genetic_algorithm(pop_size, generations, mutation_rate, lower_bound, upper_bound, csv_file):
    population = initialize_population_from_csv(csv_file)
    for _ in range(generations):
        fitnesses = [fitness(x) for x in population]
        new_population = []
        for _ in range(pop_size):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, lower_bound, upper_bound)
            new_population.append(child)
        population = new_population
    best_individual = max(population, key=fitness)
    return best_individual, fitness(best_individual)

# Run the genetic algorithm
csv_file_path = "C:\\Users\\konoz\\Desktop\\AI_OEL\\Data.csv"  # Path to your CSV file
best_x, max_value = genetic_algorithm(
    pop_size=100,
    generations=50,
    mutation_rate=0.1,
    lower_bound=0,
    upper_bound=4,
    csv_file=csv_file_path
)

print(f"Best x: {best_x}, Maximum value: {max_value}")
