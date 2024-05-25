import random
from deap import base, creator, tools, algorithms
import numpy as np


class GeneticAlgorithm:
    def __init__(self, objective_func, constraints, nonlinear_eq_constraints, var_ranges, n_vars=10, pop_size=50,
                 cxpb=0.7, mutpb=0.2, ngen=100, seed=42, penalty_factor=10000):
        self.objective_func = objective_func
        self.constraints = constraints
        self.nonlinear_eq_constraints = nonlinear_eq_constraints
        self.var_ranges = var_ranges
        self.n_vars = n_vars
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.seed = seed
        self.penalty_factor = penalty_factor

        # 初始化DEAP组件
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.random_var, self.var_ranges)
        self.toolbox.register("individual", self.generate_individual, creator.Individual, self.toolbox.attr_float)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.mate_with_constraints)
        self.toolbox.register("mutate", self.mutate_with_constraints, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.feasible_evaluate)

        # 初始化统计信息
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", lambda x: sum(val[0] for val in x) / len(x))
        self.stats.register("min", lambda x: min(val[0] for val in x))
        self.stats.register("max", lambda x: max(val[0] for val in x))
        self.hof = tools.HallOfFame(1)

    def random_var(self, ranges):
        return [random.uniform(low, high) for low, high in ranges]

    def generate_individual(self, icls, attr_float):
        x = attr_float()
        self.apply_nonlinear_eq_constraints(x)
        return icls(x)

    def apply_nonlinear_eq_constraints(self, individual):
        for constraint in self.nonlinear_eq_constraints:
            constraint(individual)

    def mate_with_constraints(self, ind1, ind2):
        tools.cxBlend(ind1, ind2, alpha=0.5)
        self.apply_nonlinear_eq_constraints(ind1)
        self.apply_nonlinear_eq_constraints(ind2)
        return ind1, ind2

    def mutate_with_constraints(self, individual, mu, sigma, indpb):
        tools.mutGaussian(individual, mu, sigma, indpb)
        self.apply_nonlinear_eq_constraints(individual)
        return individual,

    def feasible_evaluate(self, individual):
        """
        Evaluates the individual and applies penalties for constraint violations.
        """
        fitness = self.objective_func(individual)
        penalties = sum(self.constraint_penalty(individual, constraint) for constraint in self.constraints)
        total_penalty = penalties
        return fitness[0] + total_penalty,

    def constraint_penalty(self, individual, constraint):
        """
        Calculates penalty for a given constraint.
        """
        return max(0, constraint(individual)) * self.penalty_factor

    def run(self):
        random.seed(self.seed)
        pop = self.toolbox.population(n=self.pop_size)
        algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen,
                            stats=self.stats, halloffame=self.hof, verbose=True)
        return pop, self.stats, self.hof


# 定义目标函数
def objective(individual):
    return sum((x - 5) ** 2 for x in individual),  # 简单的高维目标函数


# 批量定义不等式约束条件
constraints = []
for i in range(10):
    def constraint(individual, i=i):
        return individual[i] - 10  # 每个变量都需要小于等于10 -> individual[i] <= 10 -> individual[i] - 10 <= 0


    constraints.append(constraint)


# 定义非线性等式约束
def nonlinear_eq_constraint1(individual):
    x, y = individual[0], individual[1]
    if x > 0:
        individual[1] = x ** 2  # x > 0 时 y = x**2
    else:
        if individual[1] >= x:
            individual[1] = x - 1  # x <= 0 时 y < x


nonlinear_eq_constraints = [nonlinear_eq_constraint1]

# 定义变量范围
var_ranges = [(-10, 10), (0, 100)] + [(-10, 10) for _ in range(8)]

if __name__ == "__main__":
    ga = GeneticAlgorithm(objective_func=objective, constraints=constraints,
                          nonlinear_eq_constraints=nonlinear_eq_constraints, var_ranges=var_ranges,
                          n_vars=len(var_ranges))
    pop, log, hof = ga.run()
    print("Best individual is:", hof[0])
    print("Best fitness is:", hof[0].fitness.values[0])
