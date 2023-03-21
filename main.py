# Genetic Algorithm - Graph Colouring Problem
# Project for Evolutionary Algorithms class at VUT FIT Brno
# Date: March 2023
# Author: Bc. Sebastian Krajnak
# Genetic operations, chromosone representation based on:
# Genetic Algorithm Applied to the Graph Coloring Problem (Musa M Hindi, Roman Yampolskiy)
# Some functions used were either copied or edited from EVO Lab 2 code by Ing. Martin Hurta

import numpy as np
import matplotlib as plt
import random

# Generate graph---------------------------------------------------------------------
# graph will be represented by an adjacency matrix where 1's mean the vertices are connected
# the graph is unordered so it is symmetrical and has 0s on the main diagonal
# THIS IS BIN ONLY

print("Generating adjacency matrix...")
num_vertices = 200 
graph = [[0] * num_vertices for _ in range(num_vertices)] 

for i in range(num_vertices):
    vertex1, vertex2 = random.sample(range(num_vertices),2)
    graph[vertex1][vertex2] = 1
    graph[vertex2][vertex1] = 1

print("done!")

# Load graphs from .col files -------------------------------------------------------
# EVO ONLY

def parse_file(filename):
    print(f"Creating a graph from file {filename}")
    with open(filename, 'r') as f:
        # Skip lines until we reach the line starting with 'p edge'
        line = f.readline().strip()
        while not line.startswith('p edge'):
            line = f.readline().strip()

        # Get the number of vertices from the 'p edge' line
        num_vertices = int(line.split()[2])

        # Create an empty adjacency matrix
        adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]

        # Parse the edges and update the adjacency matrix
        for line in f:
            if line.startswith('e'):
                _, v1, v2 = line.split()
                v1, v2 = int(v1) - 1, int(v2) - 1  # Convert to 0-indexed
                adjacency_matrix[v1][v2] = 1
                adjacency_matrix[v2][v1] = 1  # Assuming undirected graph
        print("...done!")
        return adjacency_matrix, num_vertices

graph, num_vertices = parse_file('/content/queen5_5.col')

# Variables to tweak with -----------------------------------------------------------
# all variables used for the evolution are easily found on one place here
max_colors = 5
num_runs = 30
                
num_generations = 500
population_size = 30
mutation_probability = 0.3
crossover_probability = 0.8
all_fitness = np.array([])

# Genetical operators ---------------------------------------------------------------

def tournament_selection(population, scores, k=2): # Author: Ing. Martin Hurta 

    # Ziskani nahodneho poradi jedincu
    random_order = list(range(0, len(population)))
    np.random.shuffle(random_order)

    # Vyber prvniho jedince
    best_idx = random_order[0]
 
    # Turnaj se zbyvajicim poctem jedincu a ulozeni jedince s nejmensim score
    for i in range(1, k):

        if scores[random_order[i]] < scores[best_idx]:
            best_idx = random_order[i]

    return population[best_idx]

def fitness(chromosome): # Penalty for every vertex connecting 2 nodes with the same color
    fitness = 0
    for i in range(num_vertices):
        for j in range(i, num_vertices):
            if(chromosome[i] == chromosome[j] and graph[i][j] == 1):
                fitness += 1
    return fitness

def single_point_crossover(p1, p2, crossover_probability): # Author: Ing. Martin Hurta
    
    # Inicializace potomku
    c1, c2 = p1.copy(), p2.copy()

    # Zjisteni, zda se provede krizeni
    if np.random.uniform() < crossover_probability:
        
        # Vyber nekoncoveho mista krizeni
        pt = np.random.randint(1, len(p1)-2)
        
        # Provedeni krizeni
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# Helpter function for mutation, simply returns all the colors of adjecent vertices for
# a given vertex
def find_adjecent_colors(vertex, chromosone):
  adj_vertices = graph[vertex]
  adj_colors = [chromosone[i] for i in range(len(chromosone)) if adj_vertices[i] == 1]

  return adj_colors

# Mutation function used when the fitness value is > 4
def mutation_high_fit(chromosone, mutation_probability):
  for vertex in range(len(chromosone)):
    if np.random.uniform() > mutation_probability:
      continue

    adj_colors = find_adjecent_colors(vertex, chromosone)
    if chromosone[vertex] in adj_colors:
      all_colors = [i for i in range(max_colors)]
      # Valid colors = all colors - adjecent colors
      valid_colors = [all_colors[i] for i in range(max_colors) if all_colors[i] not in adj_colors]
      
      if(valid_colors): # If list is empty choose random color
        new_color = random.choice(valid_colors)
      else:
        new_color = random.choice(all_colors)
      chromosone[vertex] = new_color

# Mutation function when fitness value is <= 4 
def mutation_low_fit(chromosone, mutation_probability):
  all_colors = [i for i in range(max_colors)]

  for vertex in range(len(chromosone)):
    if np.random.uniform() > mutation_probability:
      continue

    adj_colors = find_adjecent_colors(vertex, chromosone)
    if chromosone[vertex] in adj_colors:
      new_color = random.choice(all_colors)
      chromosone[vertex] = new_color

# Genetic algorithm ----------------------------------------------------------------------
# Author of the original algorithm Ing. Martin Hurta with editions by me
def genetic_algorithm(num_generations, population_size, crossover_probability, mutation_probability, all_fitness):
	
    # Radnom initial population inicialization
    population = [np.random.randint(0, max_colors-1, num_vertices).tolist() for _ in range(population_size)]

    # Inicialization of the best initial individuals
    best_individual = 0
    best_eval = fitness(population[0])
    new_best = -1
    best_history = []
    
    # Evolution
    gen = 0
    while(gen != num_generations and best_eval != 0):
    
        # Fitness score evaluation in canditate population
        scores = [fitness(indivitual) for indivitual in population]

        # Get best solution in population
        for i in range(population_size):
            if scores[i] < best_eval:
                best_individual = population[i]
                best_eval = scores[i]
                new_best = i

        # Display information about new best solution
        if new_best != -1:
            print(">%d, new best f(%s) = %d, chromatic number = " % (gen,  population[new_best], scores[new_best]), max_colors)
            best_history.append([gen, scores[new_best]])
            new_best = -1

        all_fitness = np.append(all_fitness, best_eval)

        # Possible parents selection in the size of the whole population
        selected = [tournament_selection(population, scores) for _ in range(population_size)]
        next_gen_parents = selected[:int(population_size/2)]
        population = next_gen_parents

        # Creating the next generation
        for i in range(0, int(population_size/2), 2):
            
            # Parent selection for crossover
            p1, p2 = selected[i], selected[i+1]

            # Crossover 
            for c in single_point_crossover(p1, p2, crossover_probability):

                # Mutation of individuals
                if best_eval > 4:
                  mutation_high_fit(c, mutation_probability)
                else:
                  mutation_low_fit(c, mutation_probability)

                # Ulozeni jedincu do nove generace
                population.append(c)

        gen += 1
        if(gen == num_generations):
          print("Max number of generations reached ! Stopping evolution...")

    return [best_individual, best_eval, best_history, all_fitness]


# Execution and result plotting ------------------------------------------------------------

# Execution of genetic algorithm
results = [genetic_algorithm(num_generations, population_size, crossover_probability, mutation_probability, all_fitness) for _ in range(num_runs)]

# Get total success rate of all runs
num_successes = sum([1 for result in results if result[1] == 0])
success_rate = num_successes/num_runs

# Get average fitness for all runs
# This is absolutely fucking dumb way to sum it all but I need it fast for the pojednani
fit_sum = 0
len_fit = 0
for i in range(num_runs):
  fit_sum += np.sum(results[i][3])
  len_fit+= len(results[i][3])
avg_fitness = fit_sum/len_fit

print(f'Done! Total success rate is {success_rate*100} % with average fitness {avg_fitness}')

# Plot the course of the fitness function for each run
figure, ax = plt.subplots(figsize=(10, 5))
for i in range(len(results)):
    ax.plot(*zip(*results[i][2]) ) #,label=f"Run #{i}" )
#ax.legend()
ax.grid()
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness value')
plt.title('Prubeh konfliktnych uzlu jednotlivych behu')
plt.tight_layout()
plt.show()