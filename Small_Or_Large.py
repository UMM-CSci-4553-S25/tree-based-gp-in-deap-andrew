import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap import algorithms

NUM_RUNS = 20

# Protected division to avoid divide-by-zero errors in evolved expressions
def protected_div(a, b):
    return a / b if b != 0 else 1  # Avoid zero division


pset = gp.PrimitiveSet("MAIN", arity=1) # Program take one input x
pset.renameArguments(ARG0='x')

# Genes use for evolve solution
# Add mathematical operations
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)

# Add some constants for tree
pset.addTerminal(1.0)
pset.addTerminal(2.0)
pset.addTerminal(3.0)
pset.addTerminal(-1.0)
pset.addTerminal(0.5)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Toolbox area
toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=2) # Generates tree expressions
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) # One tree = individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # create population of one individual

# === Training Data Setup ===

# Dataset A – n < 10 → small (-1) , n ≥ 20 → large (1)
# Middle zone: 10 ≤ n < 20 (no expected output)
train_data_a = [
    (0, -1),
    (5, -1),
    (9, -1),
    (10, None),  # middle zone starts
    (15, None),
    (19, None),  # middle zone ends
    (20, 1),
    (25, 1),
    (30, 1)
]

# Dataset B – n < 20 → small (-1) , n ≥ 30 → large (1)
# Middle zone: 20 ≤ n < 30

train_data_b = [
    (10, -1),
    (15, -1),
    (19, -1),
    (20, None),  # middle zone starts
    (25, None),
    (29, None),  # middle zone ends
    (30, 1),
    (35, 1),
    (40, 1)
]

USE_DATASET_A = False
train_data = train_data_a if USE_DATASET_A else train_data_b

# Compile tree
toolbox.register("compile", gp.compile, pset=pset)

# Run on all training data examples
def eval_small_or_large(individual):
    func = toolbox.compile(expr=individual)
    error = 0
    for x, expected in train_data:
        if expected is None:
            continue  # Skip middle-zone inputs
        try:
            output = func(x)
            predicted = 1 if output >= 0 else -1
            if predicted != expected:
                error += 1
        except:
            error += 1  # Any crash is an error
    return (error,) # Number of incorrect outputs (this needed to be low)

# Toolbox area
toolbox.register("evaluate", eval_small_or_large)

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=17))

def run_single_gp(seed_value):
    random.seed(seed_value)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.5, mutpb=0.2,
                                   ngen=60, stats=stats,
                                   halloffame=hof, verbose=False)

    best_fitness = hof[0].fitness.values[0]
    avg_fitness = log[-1]['avg']
    size = len(hof[0])
    return best_fitness, avg_fitness, size, hof[0], log

def main():
    results = []
    logs = []  # Store logs for each run
    best_run_index = None
    best_run_fitness = float("inf")

    for run in range(NUM_RUNS):
        best_fit, avg_fit, size, best_ind, log = run_single_gp(seed_value=run)
        results.append((run + 1, best_fit, avg_fit, size, best_ind, log))
        logs.append(log)

        if best_fit < best_run_fitness:
            best_run_fitness = best_fit
            best_run_index = run

        print(f"Run {run + 1}: Best Fit = {best_fit}, Avg Fit = {avg_fit:.2f}, Size = {size}")

    # Extract best run info
    run_id, best_fit, avg_fit, size, best_ind, best_log = results[best_run_index]

    print(f"\n🏆 Best individual (Run {run_id}): {best_ind}")
    func = toolbox.compile(expr=best_ind)
    print("Input → Predicted → Expected")
    for x, expected in train_data:
        if expected is None:
            print(f"{x:>5} →  Middle Zone → Middle Zone")
            continue
        try:
            raw = func(x)
            predicted = "large" if raw >= 0 else "small"
            expected_str = "large" if expected == 1 else "small"
            print(f"{x:>5} → {predicted:>12} → {expected_str}")
        except Exception as e:
            print(f"{x:>5} →     ERROR     → {expected_str}  (Reason: {e})")

    # Summary Table
    print("\n=== Summary Table ===")
    print("Run | Best Fit | Avg Fit | Size")
    for run, best, avg, size, *_ in results:
        print(f"{run:>3} | {best:>8} | {avg:>8.2f} | {size}")

    # Plot best run's generation-wise performance
    gen = best_log.select("gen")
    min_fit = best_log.select("min")
    plt.plot(gen, min_fit, label=f"Best Fitness (Run {run_id})")
    plt.xlabel("Generation")
    plt.ylabel("Errors")
    plt.title("Fitness over Generations (Best Run)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
