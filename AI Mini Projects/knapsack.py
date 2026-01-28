import itertools
import random
import matplotlib.pyplot as plt


def initial_pop():
    n = 12
    # getting all the possible permutations of what can go in the bag
    total_solutions = list(itertools.product([1, 0], repeat=n))
    count = 0
    pop = []
    while count < 1500:  # making my initial population be 1500
        r = random.randint(0, len(total_solutions)-1)
        pop.append(total_solutions[r])
        count = count + 1
    return pop


def fitness(individual):
    values = [6, 5, 8, 7, 6, 9, 4, 5, 4, 9, 2, 1]
    w = [20, 30, 60, 90, 50, 70, 30, 30, 70, 20, 20, 60]
    val = 0
    weight = 0
    for i in range(len(individual)):
        # individual[i] will either be a 0 or 1 which denotes whether the item is chosen to be in the bag
        val = val + individual[i]*values[i]
        weight = weight + individual[i]*w[i]
    if weight <= 250:  # only returning permutations that don't exceed the weight limit
        return val
    return 0


def pop_fitness(pop):
    fit = 0
    for i in range(len(pop)-1):
        score = fitness(pop[i])
        fit = fit + score  # adding up the individual fitness scores of each permutation
    # finding the average of the scores to represent the fitness of the whole population
    pop_fit = fit/len(pop)
    return pop_fit


def pop_weights(population):
    weights = []
    for i in population:
        # my weight array consists of the fitness score of every permutation in my population
        weights.append(fitness(i))
    return weights


def fifty_cull(population, weights):
    p = population.copy()
    cull_pop = []
    for i in range(round(len(p)/2)):
        # only getting the permutations with the top fitness score
        ind = weights.index(max(weights))
        k = p.pop(ind)
        cull_pop.append(k)
        weights.pop(ind)
    return cull_pop


def reproduce(parent1, parent2):
    n = len(parent1)
    parent1 = list(parent1)
    parent2 = list(parent2)
    r1 = random.randint(1, n)
    child1 = parent1[0:r1] + parent2[r1:]
    r2 = random.randint(1, n)
    # Since I cull my population by half, each pair of parents produce 2 offspring so we can keep the population at a constant 1500
    child2 = parent1[0:r2] + parent2[r2:]
    return child1, child2


def mutate(child):
    child = list(child)
    r = random.randint(0, len(child)-1)  # Doing a double mutation
    r1 = random.randint(0, len(child)-1)
    if child[r] == 0:
        child[r] = 1
    else:
        child[r] = 0

    if child[r1] == 0:
        child[r1] = 1
    else:
        child[r1] = 0
    return child


def genetic_alg(init_pop):
    end = 50
    pop_fit = []
    while end > 0:
        end = end - 1  # only trying 50 generations
        weights = pop_weights(init_pop)
        pop2 = []
        cullp = fifty_cull(init_pop, weights)
        cullcopy = cullp.copy()
        while len(cullp) > 0:
            r1 = random.randint(0, len(cullp)-1)
            parent1 = cullp.pop(r1)
            # parents get removed from the population once they have kids
            r2 = random.randint(0, len(cullp)-1)
            parent2 = cullp.pop(r2)
            child1, child2 = reproduce(parent1, parent2)
            mut1 = random.randint(0, 10000)
            mut2 = random.randint(0, 10000)
            if mut1 < 2:
                child1 = mutate(child1)
            pop2.append(child1)
            if mut2 < 2:
                child2 = mutate(child2)
            pop2.append(child2)
        # adding the offspring to the parent population to create the new population
        init_pop = cullcopy + pop2
        pop_f = pop_fitness(init_pop)
        pop_fit.append(pop_f)
    w = pop_weights(init_pop)
    ind = w.index(max(w))
    return init_pop[ind], w[ind], pop_fit


pop = initial_pop()
x, y, f = genetic_alg(pop)
bag = ['#1', '#2', '#3', '#4', '#5', '#6',
       '#7', '#8', '#9', '#10', '#11', '#12']
vbag = []
for k in range(len(x)):
    if x[k] == 1:
        vbag.append(bag[k])

print('Most valuable bag:', vbag)
print('Value Score:', y)

xpoints = []
for j in range(1, 51):
    xpoints.append(j)
ypoints = f

plt.plot(xpoints, ypoints)
plt.ylabel('Average Fitness of Population')
plt.xlabel('Each Generation')
plt.title('Population Fitness per Generation')
plt.show()
