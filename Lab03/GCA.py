'''
Implemented by Łukasz Gut
10/28/2018
'''
import numpy as np
import random as rnd
import math
from datetime import datetime

rnd.seed(datetime.now())


class Chromosome:
    def __init__(self, size):
        self.size = size
        self.gen_tab = np.zeros(size)

    def randomize(self):  # Randomize chromosome
        for i in range(self.gen_tab.size):
            self.gen_tab[i] = bool(rnd.randint(0, 1))

    def bin_to_dec(self):
        dec = 0
        for i in range(self.size):
            dec += self.gen_tab[self.size - i - 1] * 2 ** i
        dec = dec / (2 ** self.size / 3)
        dec = dec - 1
        return dec

    def fitness(self):  # Returns fitness of chromosome
        dec_x = self.bin_to_dec()
        return dec_x * math.sin(10 * math.pi * dec_x) + 1

    def mutate(self):
        locus = rnd.randint(0, self.size - 1)
        self.gen_tab[locus] = not self.gen_tab[locus]

    def print_chromosome(self):
        print(self.gen_tab)


class Population:
    def __init__(self, pop_size, chrm_size):
        self.pop_size = pop_size
        self.pop_tab = []
        for i in range(pop_size):
            self.pop_tab.append(Chromosome(chrm_size))
        for x in self.pop_tab:
            x.randomize()

    def cross(self, cross_probability):
        cross_amount = int(cross_probability * self.pop_size / 2)  # Amount of pairs to cross
        chrm_size = self.pop_tab[0].size

        for i in range(cross_amount):
            locus = rnd.randint(0, chrm_size - 1)
            pair1 = rnd.randint(0, self.pop_size - 1)  # Pair = (pair1, pair2)
            pair2 = rnd.randint(0, self.pop_size - 1)
            for j in range(locus, chrm_size):
                top = self.pop_tab[pair1].gen_tab[j]
                bot = self.pop_tab[pair2].gen_tab[j]
                self.pop_tab[pair1].gen_tab[j] = bot
                self.pop_tab[pair2].gen_tab[j] = top

    def mutate(self, mutate_probability):
        chrm_size = self.pop_tab[0].size
        mutate_amount = int(mutate_probability * self.pop_size * chrm_size)
        for i in range(mutate_amount):
            mutate_index = rnd.randint(0, self.pop_size - 1)
            self.pop_tab[mutate_index].mutate()

    def upgrade_population(self, new_population_tab):
        for i in range(self.pop_size):
            self.pop_tab[i] = new_population_tab[i]

    def print_population(self):
        for x in self.pop_tab:
            print(x.print_chromosome())


def val_function(dec_x):
    return dec_x * math.sin(10 * math.pi * dec_x) + 1


def best(pop_tab):  # Returns index of best chromosome in population
    fitness_tab = []
    for i in range(len(pop_tab)):
        fitness_tab.append(pop_tab[i].fitness())

    index = 0
    for i in range(len(pop_tab)):
        if fitness_tab[i] > fitness_tab[index]:
            index = i

    return index


def roulette_selection(pop_tab, pop_size):
    new_population = []
    sum = 0
    for i in range(pop_size):
        sum += pop_tab[i].fitness()

    pick = rnd.uniform(0, sum)

    for i in range(pop_size):
        for j in range(pop_size):
            pick -= pop_tab[j].fitness()
            if pick <= 0:
                new_population.append(pop_tab[j])

    return new_population


# Data
cross_probability = 0.5
mutate_probability = 0.1

pop_size = 20
chrm_size = 22

age = 100

# Algorithm
population = Population(pop_size, chrm_size)

# Finding the best chap in population 0 (his index in population.pop_tab and fitness value)
best_index = best(population.pop_tab)
best_value = population.pop_tab[best_index].fitness()

print(best_value)

i = 0
while i < age:
    next_population = roulette_selection(population.pop_tab, pop_size)
    population.upgrade_population(next_population)
    population.cross(cross_probability)
    population.mutate(mutate_probability)

    curr_best_index = best(population.pop_tab)
    curr_best_value = next_population[curr_best_index].fitness()
    if curr_best_value > best_value:
        print("x = ", population.pop_tab[curr_best_index].bin_to_dec(), " y = ", curr_best_value)
        best_value = curr_best_value
    i += 1