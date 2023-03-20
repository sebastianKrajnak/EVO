import numpy as np
import matplotlib as plt
import random
import pygad


# Generate graph---------------------------------------------------------------------
# graph will be represented by an adjacency matrix where 1's mean the vertices are connected
# the graph is unordered so it is symmetrical and has 0s on the main diagonal

print("Generating adjacency matrix...")
num_vertices = 200 
graph = [[0] * num_vertices for _ in range(num_vertices)] 

for i in range(num_vertices):
    vertex1, vertex2 = random.sample(range(num_vertices),2)
    graph[vertex1][vertex2] = 1
    graph[vertex2][vertex1] = 1

print("done!")

# -----------------------------------------------------------------------------------

def fitness(solution, solution_idx):
    fitness = 0
    for i in range(num_vertices):
        for j in range(i, num_vertices):
            if(solution[i] == solution[j] and graph[i][j] == 0):
                fitness += 1
    return fitness
                
num_generations = 100
num_parents_mating = 6
sol_per_pop = 50
num_genes = num_vertices
mutation_probability = 0.2
crossover_probability = 0.5

fitness_function = fitness
init_range_low = 0
init_range_high = num_vertices/2

parent_selection_type = "sss"
keep_parents = 2

crossover_type = "single_point"

mutation_type = "random"

def on_generation(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    
ga_instance = pygad.GA(
    num_generations = num_generations,
    num_parents_mating = num_parents_mating,
    sol_per_pop = sol_per_pop,
    num_genes = num_genes,
    mutation_probability = mutation_probability,
    crossover_probability = crossover_probability,
    fitness_func = fitness_function,
    init_range_low = init_range_low,
    init_range_high = init_range_high,
    parent_selection_type = parent_selection_type,
    keep_parents = keep_parents,
    crossover_type = crossover_type,
    mutation_type = mutation_type
)

ga_instance.run()

ga_instance.plot_fitness()
