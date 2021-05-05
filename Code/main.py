secretKey = 'IXwMSaezPNU05FBfZVSof8uq4nKJcugCqxq0bR9dy1qEnSoXb4'

import sys
import array
import random
import math
import numpy as np
import pandas as pd
import dill
from client import *
np.set_printoptions(precision=4,
                       threshold=3,
                       linewidth=150)
np.set_printoptions(threshold=sys.maxsize)



# ERRORS FOR GENERATION
def calculate_generation_error(last_generation):
    errors = []
    for i in last_generation:
        i_list = list(i)
        errors.append(get_errors(secretKey, i_list))
    return errors

# FITNESS
def fitness(error):
    return (error[0] + error[1]) + 2 * abs(error[0]-error[1])

# FITNESS FOR GENERATION
def calculate_generation_fitness(last_generation_errors):
    return [fitness(i) for i in last_generation_errors]

# CROSSOVER
def crossover(parent1, parent2, error1, error2):
    weight1 = error1/ (error1 + error2)
    weight2 = error2/ (error1 + error2)
    
    return (parent1*weight1 + weight2*parent2) / (weight1+weight2)

# PROBABILITY OF SELECTION
def calculate_generation_probability(population_fitness):
    population_log = np.log10(population_fitness)
    population_inverse_log = -population_log
    population_exp = np.exp(population_inverse_log)
    population_probability = population_exp/ np.sum(population_exp)
    return population_probability


# SELECTION
def selection(population,population_probability):
    df = pd.DataFrame()
    df['population'] = population
    df['population_probability'] = population_probability
    df = df.sort_values(by = ['population_probability'])
    pool = (df.tail(10).index)
    parent1, parent2 = np.random.choice(pool, 2)
    return parent1, parent2

# MUTATION
def mutate(child):
    for i in range(0,11):
        if random.random() <= 6/11:
            child[i] = np.random.normal(child[i], abs(child[i]/20))
            child[i] = max(min(child[i],10),-10)
    return child


# GENETIC ALGORITHM
def keep_doing(generation, generation_errors, generation_fitness, generation_probability, generation_parents, times):

    start = len(generation) - 1

    for i in range(0,times):
        generation_errors.append(calculate_generation_error(generation[start+i]))
        generation_fitness.append(calculate_generation_fitness(generation_errors[start+i]))
        generation_probability.append(calculate_generation_probability(generation_fitness[start+i]))

        picked_parents = []
        children = []


        for j in range(0,population_size):
            parent1, parent2 = selection(generation[start+i], generation_probability[start+i])
            picked_parents.append( (np.array(generation_errors[start+i][parent1]), np.array(generation_errors[start+i][parent2])) )
            print("picked " + str(np.array(generation_errors[start+i][parent1])) + " and " + str(np.array(generation_errors[start+i][parent2])) )
            children.append(mutate(crossover(generation[start+i][parent1], generation[start+i][parent2], generation_fitness[start+i][parent1], generation_fitness[start+i][parent2])))
        generation.append(children)
        generation_parents.append(picked_parents)
        print("\n")

    return generation, generation_errors, generation_fitness, generation_probability, generation_parents



########################## THIS PART RUNS #############################################################

##### SET UP INITIAL POPULATION ##########

population_size = 20
initial_population = []

best_vec = [0.0, 1.6060214068505298e-13, -9.31764664310612e-14, 7.7434269121267985,
                -1.0341715514875065, 1.5742601750306104e-17, 7.493411158765831e-17,
                6.440430161496442e-07, -2.2411108804040348e-14, 1.4221464275627044e-14,3.7989499869642936e-15]

for i in range(0,population_size):
    child = np.array(best_vec)
    child = np.random.normal(child, abs(child)/10)
    child = [max(min(i,10),-10) for i in child]
    child = np.array(child)

    print(child)
    initial_population.append(child)


############# SET UP ALL VARIABLES #############

generation = []
generation_errors = []
generation_fitness = []
generation_probability = []
generation_parents = [[]]

generation.append(initial_population)


# Last generation is the latest child, but error and fitness are available till its parent
times = 20
generation, generation_errors, generation_fitness, generation_probability, generation_parents = keep_doing(generation, generation_errors, generation_fitness, generation_probability ,generation_parents, times)

# This store all the data as a dill file
dill.dump_session('dill_logs/file_name')