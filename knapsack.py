#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

IND_INIT_SIZE = 5           # Number of items in the knapsack
MAX_ITEM = 50            # Maximum number of items in the knapsack
MAX_WEIGHT = 50          # Maximum weight of the knapsack
NBR_ITEMS = 20          # Number of items available

# To assure reproducibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(72)

randint = random.randint(1, 11)
numRuns = 20

# def randBoolSequence(intForBoolean):
#     if intForBoolean  5:
#         return True
#     else:
#         return False

maxSeq = 1
seqLen = 20

def randomSequence(maxSequences, sequenceLength):
    num_sequences = random.randint(1, maxSequences)
    sequences = []
    for i in range(num_sequences):
        sequence = [random.randint(0, 10) 
                    for i in range(sequenceLength)]  # Adjust range as needed
        sequences.append(sequence)
    return sequence

boolean_list = [item >= 5 for item in randomSequence(maxSeq, seqLen)]




# Create the item dictionary: item name is an integer, and value is 
# a (weight, value) 2-tuple.
items = {}
# Create random items and store them in the items' dictionary.
for i in range(NBR_ITEMS): # 20 items
    items[i] = (random.randint(1, 10), random.uniform(0, 100)) # (weight, value)

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0)) # Minimize weight, maximize value
creator.create("Individual", set, fitness=creator.Fitness) # Set of items

toolbox = base.Toolbox()    # Create a toolbox

# Attribute generator
toolbox.register("attr_item", random.randrange, NBR_ITEMS)      # Randomly select an item

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE) # Initialize an individual with 5 items
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Initialize a population

def evalKnapsack(individual): # Evaluate the fitness of an individual
    weight = 0.0 # Initialize the weight of the bag
    value = 0.0 # Initialize the value of the bag
    for item in individual: 
        weight += items[item][0] # Add the weight of the item to the total weight
        value += items[item][1] # Add the value of the item to the total value
    if len(individual) > MAX_ITEM or weight > MAX_WEIGHT:   # Ensure overweighted bags are dominated
        return 10000, 0             # Ensure overweighted bags are dominated
    return weight, value # Return the weight and value of the bag

def cxSet(ind1, ind2): # Crossover operation
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                   
    return ind1, ind2

def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2

def cxTwoPoint(ind1, ind2):
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2

def mutSet(individual): # Mutation operation
    """Mutation that pops or add an element.""" 
    if random.random() < 0.5: # Remove an element
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual)))) # Remove an item randomly selected
    else:                          # Add an element
        individual.add(random.randrange(NBR_ITEMS)) # Add an item randomly selected
    return individual, 

toolbox.register("evaluate", evalKnapsack) # Evaluation function
toolbox.register("mate", cxSet) 
# toolbox.register("mate", cxOnePoint) 
toolbox.register("mutate", mutSet) # Mutation operation
toolbox.register("select", tools.selNSGA2)  # Select the best individuals based on the NSGA-II algorithm

def main():
    # random.seed(64)
    random.seed(None)
    NGEN = 50 # Number of generations
    MU = 50 # Number of individuals to select for the next generation
    LAMBDA = 100 # Number of children to produce at each generation
    CXPB = 0.8 # Probability of mating two individuals
    MUTPB = 0.2 # Probability of mutating an individual

    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    # stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    
    return stats

if __name__ == "__main__":
    main()                 