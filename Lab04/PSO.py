import random as rnd
import math
from datetime import datetime
import matplotlib.pyplot as plt

rnd.seed(datetime.now())


class Particle:
    def __init__(self, v_max):
        self.x = rnd.uniform(-10, 10)
        self.y = rnd.uniform(-10, 10)
        self.v_x = rnd.uniform(-0.01, 0.01)
        self.v_y = rnd.uniform(-0.01, 0.01)
        self.v_max = v_max
        self.c1 = 2
        self.c2 = 2

    def fitness(self):
        return self.x ** 2 + self.y ** 2 - 20 * (math.cos(math.pi * self.x) + math.cos(math.pi * self.y) - 2)

    def update_velocity(self, p_best_x, p_best_y, g_best_x, g_best_y):
        offset = U(0, self.c1) * (p_best_x - self.x) + U(0, self.c2) * (g_best_x - self.x)
        velocity_x = self.v_x + offset
        if velocity_x > self.v_max:
            self.v_x = self.v_max
        elif velocity_x < (-1) * self.v_max:
            self.v_x = (-1) * self.v_max
        else:
            self.v_x = velocity_x

        velocity_y = self.v_y + (U(0, self.c1) * (p_best_y - self.y) + U(0, self.c2) * (g_best_y - self.y))
        if velocity_y > self.v_max:
            self.v_y = self.v_max
        elif velocity_y < (-1) * self.v_max:
            self.v_y = (-1) * self.v_max
        else:
            self.v_y = velocity_y

    def update_position(self):
        x_max = 10.0
        y_max = 10.0

        self.x = self.x + self.v_x
        self.y = self.y + self.v_y

        if self.x > x_max:
            offset = abs(self.x + self.v_x) - 10.0
            self.x = 10 - offset
        if self.x < (-1) * x_max:
            offset = abs(self.x + self.v_x) - 10.0
            self.x = -10 + offset
        if self.y > y_max:
            offset = abs(self.y + self.v_y) - 10.0
            self.y = 10 - offset
        if self.y < (-1) * y_max:
            offset = abs(self.y + self.v_y) - 10.0
            self.y = -10 + offset


class Population:
    def __init__(self, pop_size, v_max):
        self.pop_size = pop_size
        self.v_max = v_max
        self.pop_tab = []
        for i in range(pop_size):
            self.pop_tab.append(Particle(v_max))

    def fitness_of_particles(self):
        fit_tab = []
        for i in range(len(self.pop_tab)):
            fit_tab.append(self.pop_tab[i].fitness())

        return fit_tab

    def best_x_y(self):
        index_best = 0
        value_best = self.pop_tab[0].fitness()

        for i in range(len(self.pop_tab)):
            if self.pop_tab[i].fitness() < value_best:
                index_best = i

        return index_best


def U(x, y):
    return rnd.uniform(x, y)


def search_best_index(fit_tab):
    index = 0
    for i in range(len(fit_tab)):
        if fit_tab[i] < fit_tab[index]:
            index = i
    return index


def plot(population, pop_size):
    x = []
    y = []
    for i in range(pop_size):
        x.append(population.pop_tab[i].x)
        y.append(population.pop_tab[i].y)
    plt.scatter(x, y)
    plt.axis([-10, 10, -10, 10])
    plt.show()


pop_size = 20
v_max = 0.01

population = Population(pop_size, v_max)

g_best_index = population.best_x_y()
g_best_particle = [population.pop_tab[g_best_index].x, population.pop_tab[g_best_index].y]
g_best_value = population.pop_tab[g_best_index].fitness()

k = 0
while (k < 2000):

    # if i % 100 == 0:
    #    plot(population, pop_size)

    fitness_tab = population.fitness_of_particles()

    p_best_index = search_best_index(fitness_tab)
    p_best_particle = [population.pop_tab[p_best_index].x, population.pop_tab[p_best_index].y]
    p_best_value = population.pop_tab[p_best_index].fitness()

    if p_best_value < g_best_value:
        g_best_index = p_best_index
        g_best_particle = p_best_particle
        g_best_value = p_best_value

    for i in range(pop_size):
        population.pop_tab[i].update_velocity(population.pop_tab[p_best_index].x, population.pop_tab[p_best_index].y,
                                              population.pop_tab[g_best_index].x, population.pop_tab[g_best_index].y)
        population.pop_tab[i].update_position()

    print("(x,y) = (", format(g_best_particle[0], '.10f'), ", ", format(g_best_particle[1], '.10f'), ")")
    print("f(x,y) = ", format(g_best_value, '.10f'))
    print("\n\n")

    k += 1
