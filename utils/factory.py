# Constants initialization
import time

import pandas as pd

from EEE.utils.constants import *
from EEE.utils.helpers.genetic_algorithm import GeneticAlgorithm
import math


class Factory:

    def __init__(self, name: str,
                 wind_power_list: list[float] = None, solar_power_list: list[float] = None, power_demand_list: list[float] = None,
                 battery_power: int = 50, battery_capacity: int = 100,
                 wind_power_list2: list[float] = None, solar_power_list2: list[float] = None,
                 wind_power_list3: list[float] = None, solar_power_list3: list[float] = None,
                 wind_power_list24: list[list[float]] = None, solar_power_list24: list[list[float]] = None):
        """
        :param name: Name of the factory
        :param wind_power_list: List of wind power values, in 24-hour format
        :param solar_power_list: List of solar power values, in 24-hour format
        :param power_demand_list: List of power demand values, in 24-hour format
        :param battery_power: Power of the battery in kW
        :param battery_capacity: Capacity of the battery in kWh
        """
        self.name = name
        self.wind_power_list = wind_power_list
        self.solar_power_list = solar_power_list
        self.power_demand_list = power_demand_list
        self.battery_power = battery_power
        self.battery_capacity = battery_capacity
        self.wind_power_list2 = wind_power_list2
        self.solar_power_list2 = solar_power_list2
        self.wind_power_list3 = wind_power_list3
        self.solar_power_list3 = solar_power_list3
        self.wind_power_list24 = wind_power_list24
        self.solar_power_list24 = solar_power_list24

    def calculate_cost_without_battery(self):
        """
        Calculates the cost of power consumption for the factory without a battery.
        :return: Total cost in yuan
        """
        # Initialize
        total_power = 0
        abandon_power = 0
        total_cost = 0
        for i in range(24):
            # Calculate the power shortage
            power_shortage = self.power_demand_list[i] - (self.wind_power_list[i] + self.solar_power_list[i])
            if power_shortage < 0:
                abandon_power += -power_shortage
            total_power = sum(self.power_demand_list)
            # Calculate the cost
            total_cost += self.wind_power_list[i] * wind_power_price + self.solar_power_list[i] * solar_power_price
            if power_shortage > 0:
                total_cost += power_shortage * power_demand_price
        average_cost = total_cost / total_power
        return total_power+abandon_power, abandon_power, total_cost, average_cost

    def calculate_cost_with_battery(self):
        """
        Calculates the cost of power consumption for the factory with a battery.
        Using the linear programming method.
        :return: Total cost in yuan
        """


        """
        Target: minimize the total cost, which means minimizing the sum of power bought from the grid
        Variables: battery_power_by_hour * 24, bivariate
        """

        def objective(individual):
            result = []
            for i in range(24):
                result.append(individual[3 * i + 1])
            return sum(result),

        constraints = []
        nonlinear_eq_constraints = []
        var_ranges = []

        for i in range(24):

            if 8 <= i <= 16:
                var_ranges.append((-self.battery_power, 0))
            else:
                var_ranges.append((0, self.battery_power))

            def battery_power_upper_bound(individual, i=i):
                return individual[3 * i] - self.battery_power
            constraints.append(battery_power_upper_bound)

            def battery_power_lower_bound(individual, i=i):
                return -self.battery_power - individual[3 * i]
            constraints.append(battery_power_lower_bound)

            var_ranges.append((0, math.inf))

            def power_grid_lower_bound(individual, i=i):
                return -individual[3 * i + 1]
            constraints.append(power_grid_lower_bound)

            var_ranges.append((0.1 * self.battery_capacity, 0.9 * self.battery_capacity))

            def SOC_upper_bound(individual, i=i):
                return individual[3 * i + 2] - 0.9 * self.battery_capacity
            constraints.append(SOC_upper_bound)

            def SOC_lower_bound(individual, i=i):
                return 0.1 * self.battery_capacity - individual[3 * i + 2]
            constraints.append(SOC_lower_bound)

            if i == 23:
                def final_SOC_upper_bound(individual, i=i):
                    return individual[3 * i + 2] - individual[3 * i] - (individual[72] * 1.01) * self.battery_capacity
                constraints.append(final_SOC_upper_bound)

                def final_SOC_lower_bound(individual, i=i):
                    return (individual[72] * 0.99) * self.battery_capacity - (individual[3 * i + 2] - individual[3 * i])
                constraints.append(final_SOC_lower_bound)

            def power_balance(individual, i=i):
                shortage = self.power_demand_list[i] - self.wind_power_list[i] - self.solar_power_list[i]
                if shortage < 0 < individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i + 1] > 0 > individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i] > 0:
                    temp = (self.power_demand_list[i] - self.wind_power_list[i] - self.solar_power_list[i] -
                            individual[3 * i] * battery_efficiency)
                else:
                    temp = self.power_demand_list[i] - self.wind_power_list[i] - self.solar_power_list[i] - \
                           individual[3 * i] / battery_efficiency
                if temp < 0:
                    individual[3 * i + 1] = 0
                else:
                    individual[3 * i + 1] = temp
            nonlinear_eq_constraints.append(power_balance)

            def SOC_balance(individual, i=i):
                if i == 0:
                    individual[3 * i + 2] = individual[72] * self.battery_capacity
                else:
                    individual[3 * i + 2] = individual[3 * (i - 1) + 2] - individual[3 * (i - 1)]
            nonlinear_eq_constraints.append(SOC_balance)

        var_ranges.append((0.1, 0.9))

        ga = GeneticAlgorithm(objective_func=objective, constraints=constraints,
                               nonlinear_eq_constraints=nonlinear_eq_constraints, n_vars=24 * 3 + 1, var_ranges=var_ranges, ngen=10000, pop_size=100, cxpb=0.8, mutpb=0.3, penalty_factor=1e6)
        pop, log, hof = ga.run()
        dataframe = pd.DataFrame(columns=['battery_power', 'power_grid', 'SOC'])
        for i in range(24):
            dataframe.loc[i] = [hof[0][3 * i], hof[0][3 * i + 1], hof[0][3 * i + 2]]
        dataframe.loc[24] = ["init_SOC", "", ""]
        dataframe.loc[25] = [hof[0][72], "", ""]
        result = hof[0].fitness.values[0] + sum(self.wind_power_list[i] * wind_power_price + self.solar_power_list[i] * solar_power_price for i in range(24)) + \
                 (self.battery_power * battery_power_price + self.battery_capacity * battery_capacity_price) / 10 / 365
        dataframe.loc[26] = ["result", "", ""]
        dataframe.loc[27] = [result, "", ""]
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dataframe.to_csv(f"C://Users/Jawk/PycharmProjects/EEE/EEE/data/result_{self.name}_1_2_{timestamp}.csv")




    def calculate_cost_with_battery_change(self):
        """
        Calculates the cost of power consumption for the factory with a battery.
        Using the linear programming method.
        :return: Total cost in yuan
        """


        """
        Target: minimize the total cost, which means minimizing the sum of power bought from the grid
        Variables: battery_power_by_hour * 24, bivariate
        """

        def objective(individual):
            result = []
            for i in range(24):
                result.append(individual[3 * i + 1] * power_demand_price)
            result.append(individual[72] * battery_power_price / 10 / 365)
            result.append(individual[73] * battery_capacity_price / 10 / 365)
            return sum(result),

        constraints = []
        nonlinear_eq_constraints = []
        var_ranges = []

        for i in range(24):

            if 8 <= i <= 16:
                var_ranges.append((-self.battery_power, 0))
            else:
                var_ranges.append((0, self.battery_power))

            def battery_power_upper_bound(individual, i=i):
                return individual[3 * i] - individual[72]
            constraints.append(battery_power_upper_bound)

            def battery_power_lower_bound(individual, i=i):
                return -individual[72] - individual[3 * i]
            constraints.append(battery_power_lower_bound)

            var_ranges.append((0, math.inf))

            def power_grid_lower_bound(individual, i=i):
                return -individual[3 * i + 1]
            constraints.append(power_grid_lower_bound)

            var_ranges.append((0.1 * self.battery_capacity, 0.9 * self.battery_capacity))

            def SOC_upper_bound(individual, i=i):
                return individual[3 * i + 2] - 0.9 * individual[73]
            constraints.append(SOC_upper_bound)

            def SOC_lower_bound(individual, i=i):
                return 0.1 * individual[73] - individual[3 * i + 2]
            constraints.append(SOC_lower_bound)

            if i == 23:
                def final_SOC_upper_bound(individual, i=i):
                    return individual[3 * i + 2] - individual[3 * i] - (individual[74]*1.01) * individual[73]

                constraints.append(final_SOC_upper_bound)

                def final_SOC_lower_bound(individual, i=i):
                    return (individual[74]*0.99) * individual[73] - (individual[3 * i + 2] - individual[3 * i])

                constraints.append(final_SOC_lower_bound)

            def power_balance(individual, i=i):
                shortage = self.power_demand_list[i] - self.wind_power_list[i] - self.solar_power_list[i]
                if shortage < 0 < individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i + 1] > 0 > individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i] > 0:
                    temp = (self.power_demand_list[i] - self.wind_power_list[i] - self.solar_power_list[i] -
                            individual[3 * i] * battery_efficiency)
                else:
                    temp = self.power_demand_list[i] - self.wind_power_list[i] - self.solar_power_list[i] - \
                           individual[3 * i] / battery_efficiency
                if temp < 0:
                    individual[3 * i + 1] = 0
                else:
                    individual[3 * i + 1] = temp
            nonlinear_eq_constraints.append(power_balance)

            def SOC_balance(individual, i=i):
                if i == 0:
                    individual[3 * i + 2] = individual[74] * individual[73]
                else:
                    individual[3 * i + 2] = individual[3 * (i - 1) + 2] - individual[3 * (i - 1)]
            nonlinear_eq_constraints.append(SOC_balance)

        var_ranges.append((0.5 * self.battery_power, 1.5 * self.battery_power))
        var_ranges.append((0.5 * self.battery_capacity, 1.5 * self.battery_capacity))
        var_ranges.append((0.1, 0.9))

        ga = GeneticAlgorithm(objective_func=objective, constraints=constraints,
                               nonlinear_eq_constraints=nonlinear_eq_constraints, n_vars=24 * 3 + 3, var_ranges=var_ranges, ngen=10000, pop_size=100, cxpb=0.8, mutpb=0.3, penalty_factor=1e6)
        pop, log, hof = ga.run()
        dataframe = pd.DataFrame(columns=['battery_power', 'power_grid', 'SOC'])
        for i in range(24):
            dataframe.loc[i] = [hof[0][3 * i], hof[0][3 * i + 1], hof[0][3 * i + 2]]
        dataframe.loc[24] = ["Power", "Capacity", "init_SOC"]
        dataframe.loc[25] = [hof[0][72], hof[0][73], hof[0][74]]
        result = hof[0].fitness.values[0] + sum(self.wind_power_list[i] * wind_power_price + self.solar_power_list[i] * solar_power_price for i in range(24))
        dataframe.loc[26] = ["result", "", ""]
        dataframe.loc[27] = [result, "", ""]
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dataframe.to_csv(f"C://Users/Jawk/PycharmProjects/EEE/EEE/data/result_{self.name}_1_3_{timestamp}.csv")



    def calculate_cost_with_wind_and_solar_change(self):
        """
        Calculates the cost of power consumption for the factory with a battery.
        Attention: When recalling this function, the wind_power_list and solar_power_list should in standard format.
        :return: Total cost in yuan
        """

        def objective(individual):
            result = []
            for i in range(24):
                result.append(individual[3 * i + 1] * power_demand_price)
            result.append(individual[72] * battery_power_price / 10 / 365)
            result.append(individual[73] * battery_capacity_price / 10 / 365)
            # Cost of constructing new wind and solar power plants
            result.append(individual[75] * wind_capacity_price / 5 / 365)
            result.append(individual[76] * solar_capacity_price / 5 / 365)
            return sum(result),

        constraints = []
        nonlinear_eq_constraints = []
        var_ranges = []

        for i in range(24):

            if 8 <= i <= 16:
                var_ranges.append((-self.battery_power, 0))
            else:
                var_ranges.append((0, self.battery_power))

            def battery_power_upper_bound(individual, i=i):
                return individual[3 * i] - individual[72]
            constraints.append(battery_power_upper_bound)

            def battery_power_lower_bound(individual, i=i):
                return -individual[72] - individual[3 * i]
            constraints.append(battery_power_lower_bound)

            var_ranges.append((0, math.inf))

            def power_grid_lower_bound(individual, i=i):
                return -individual[3 * i + 1]
            constraints.append(power_grid_lower_bound)

            var_ranges.append((0.1 * self.battery_capacity, 0.9 * self.battery_capacity))

            def SOC_upper_bound(individual, i=i):
                return individual[3 * i + 2] - 0.9 * individual[73]
            constraints.append(SOC_upper_bound)

            def SOC_lower_bound(individual, i=i):
                return 0.1 * individual[73] - individual[3 * i + 2]
            constraints.append(SOC_lower_bound)

            if i == 23:
                def final_SOC_upper_bound(individual, i=i):
                    return individual[3 * i + 2] - individual[3 * i] - (individual[74]*1.01) * individual[73]

                constraints.append(final_SOC_upper_bound)

                def final_SOC_lower_bound(individual, i=i):
                    return (individual[74]*0.99) * individual[73] - (individual[3 * i + 2] - individual[3 * i])

                constraints.append(final_SOC_lower_bound)

            def power_balance(individual, i=i):
                wind_power = self.wind_power_list[i] * individual[75]
                solar_power = self.solar_power_list[i] * individual[76]
                shortage = self.power_demand_list[i] - wind_power - solar_power
                if shortage < 0 < individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i + 1] > 0 > individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i] > 0:
                    temp = self.power_demand_list[i] - wind_power - solar_power - \
                           individual[3 * i] * battery_efficiency
                else:
                    temp = self.power_demand_list[i] - wind_power - solar_power - \
                           individual[3 * i] / battery_efficiency
                if temp < 0:
                    individual[3 * i + 1] = 0
                else:
                    individual[3 * i + 1] = temp
            nonlinear_eq_constraints.append(power_balance)

            def SOC_balance(individual, i=i):
                if i == 0:
                    individual[3 * i + 2] = individual[74] * individual[73]
                else:
                    individual[3 * i + 2] = individual[3 * (i - 1) + 2] - individual[3 * (i - 1)]
            nonlinear_eq_constraints.append(SOC_balance)

        var_ranges.append((0.5 * self.battery_power, 2 * self.battery_power))
        var_ranges.append((0.5 * self.battery_capacity, 2 * self.battery_capacity))
        var_ranges.append((0.1, 0.9))
        var_ranges.append((500, 1500))

        def wind_power_lower_bound(individual):
            return -individual[75]
        var_ranges.append((500, 1500))

        def solar_power_lower_bound(individual):
            return -individual[76]
        constraints.append(wind_power_lower_bound)
        constraints.append(solar_power_lower_bound)

        ga = GeneticAlgorithm(objective_func=objective, constraints=constraints,
                               nonlinear_eq_constraints=nonlinear_eq_constraints, n_vars=24 * 3 + 5, var_ranges=var_ranges, ngen=10000, pop_size=100, cxpb=0.8, mutpb=0.3, penalty_factor=1e6)
        pop, log, hof = ga.run()
        dataframe = pd.DataFrame(columns=['battery_power', 'power_grid', 'SOC'])
        for i in range(24):
            dataframe.loc[i] = [hof[0][3 * i], hof[0][3 * i + 1], hof[0][3 * i + 2]]
        dataframe.loc[24] = ["Power", "Capacity", "init_SOC"]
        dataframe.loc[25] = [hof[0][72], hof[0][73], hof[0][74]]
        result = hof[0].fitness.values[0]
        dataframe.loc[26] = ["result", "wind", "solar"]
        dataframe.loc[27] = [result, hof[0][75], hof[0][76]]
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dataframe.to_csv(f"C://Users/Jawk/PycharmProjects/EEE/EEE/data/result_{self.name}_3_1_{timestamp}.csv")

    def calculate_cost_with_wind_and_solar_change_united(self):
        """
        Calculates the cost of power consumption for the factory with a battery.
        Attention: When recalling this function, the wind_power_list and solar_power_list should in standard format.
        :return: Total cost in yuan
        """

        def objective(individual):
            result = []
            for i in range(24):
                result.append(individual[3 * i + 1] * power_demand_price)
            result.append(individual[72] * battery_power_price / 10 / 365)
            result.append(individual[73] * battery_capacity_price / 10 / 365)
            # Cost of constructing new wind and solar power plants
            result.append((individual[75] + individual[77] + individual[79]) * wind_capacity_price / 5 / 365)
            result.append((individual[76] + individual[78] + individual[80]) * solar_capacity_price / 5 / 365)
            return sum(result),

        constraints = []
        nonlinear_eq_constraints = []
        var_ranges = []

        for i in range(24):

            if 8 <= i <= 16:
                var_ranges.append((-self.battery_power, 0))
            else:
                var_ranges.append((0, self.battery_power))

            def battery_power_upper_bound(individual, i=i):
                return individual[3 * i] - individual[72]
            constraints.append(battery_power_upper_bound)

            def battery_power_lower_bound(individual, i=i):
                return -individual[72] - individual[3 * i]
            constraints.append(battery_power_lower_bound)

            var_ranges.append((0, math.inf))

            def power_grid_lower_bound(individual, i=i):
                return -individual[3 * i + 1]
            constraints.append(power_grid_lower_bound)

            var_ranges.append((0.1 * self.battery_capacity, 0.9 * self.battery_capacity))

            def SOC_upper_bound(individual, i=i):
                return individual[3 * i + 2] - 0.9 * individual[73]
            constraints.append(SOC_upper_bound)

            def SOC_lower_bound(individual, i=i):
                return 0.1 * individual[73] - individual[3 * i + 2]
            constraints.append(SOC_lower_bound)

            if i == 23:
                def final_SOC_upper_bound(individual, i=i):
                    return individual[3 * i + 2] - individual[3 * i] - (individual[74]*1.01) * individual[73]

                constraints.append(final_SOC_upper_bound)

                def final_SOC_lower_bound(individual, i=i):
                    return (individual[74]*0.99) * individual[73] - (individual[3 * i + 2] - individual[3 * i])

                constraints.append(final_SOC_lower_bound)

            def power_balance(individual, i=i):
                wind_power = self.wind_power_list[i] * individual[75] + self.wind_power_list2[i] * individual[77] + self.wind_power_list3[i] * individual[79]
                solar_power = self.solar_power_list[i] * individual[76] + self.solar_power_list2[i] * individual[78] + self.solar_power_list3[i] * individual[80]
                shortage = self.power_demand_list[i] - wind_power - solar_power
                if shortage < 0 < individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i + 1] > 0 > individual[3 * i]:
                    individual[3 * i] = 0
                if individual[3 * i] > 0:
                    temp = self.power_demand_list[i] - wind_power - solar_power - \
                           individual[3 * i] * battery_efficiency
                else:
                    temp = self.power_demand_list[i] - wind_power - solar_power - \
                           individual[3 * i] / battery_efficiency
                if temp < 0:
                    individual[3 * i + 1] = 0
                else:
                    individual[3 * i + 1] = temp
            nonlinear_eq_constraints.append(power_balance)

            def SOC_balance(individual, i=i):
                if i == 0:
                    individual[3 * i + 2] = individual[74] * individual[73]
                else:
                    individual[3 * i + 2] = individual[3 * (i - 1) + 2] - individual[3 * (i - 1)]
            nonlinear_eq_constraints.append(SOC_balance)

        var_ranges.append((0.5 * self.battery_power, 2 * self.battery_power))
        var_ranges.append((0.5 * self.battery_capacity, 2 * self.battery_capacity))
        var_ranges.append((0.1, 0.9))
        var_ranges.append((500, 1500))

        def wind_power_lower_bound(individual):
            return -individual[75]
        var_ranges.append((500, 1500))

        def solar_power_lower_bound(individual):
            return -individual[76]
        constraints.append(wind_power_lower_bound)
        constraints.append(solar_power_lower_bound)

        var_ranges.append((500, 1500))

        def wind_power_lower_bound2(individual):
            return -individual[77]
        var_ranges.append((500, 1500))

        def solar_power_lower_bound2(individual):
            return -individual[78]
        constraints.append(wind_power_lower_bound2)
        constraints.append(solar_power_lower_bound2)

        var_ranges.append((500, 1500))

        def wind_power_lower_bound3(individual):
            return -individual[79]
        var_ranges.append((500, 1500))

        def solar_power_lower_bound3(individual):
            return -individual[80]
        constraints.append(wind_power_lower_bound3)
        constraints.append(solar_power_lower_bound3)

        ga = GeneticAlgorithm(objective_func=objective, constraints=constraints,
                               nonlinear_eq_constraints=nonlinear_eq_constraints, n_vars=24 * 3 + 9, var_ranges=var_ranges, ngen=10000, pop_size=200, cxpb=0.8, mutpb=0.3, penalty_factor=1e6)
        pop, log, hof = ga.run()
        dataframe = pd.DataFrame(columns=['battery_power', 'power_grid', 'SOC'])
        for i in range(24):
            dataframe.loc[i] = [hof[0][3 * i], hof[0][3 * i + 1], hof[0][3 * i + 2]]
        dataframe.loc[24] = ["Power", "Capacity", "init_SOC"]
        dataframe.loc[25] = [hof[0][72], hof[0][73], hof[0][74]]
        result = hof[0].fitness.values[0]
        dataframe.loc[26] = ["result", "wind", "solar"]
        dataframe.loc[27] = [result, hof[0][75], hof[0][76]]
        dataframe.loc[28] = ["wind2", "solar2", ""]
        dataframe.loc[29] = [hof[0][77], hof[0][78], ""]
        dataframe.loc[30] = ["wind3", "solar3", ""]
        dataframe.loc[31] = [hof[0][79], hof[0][80], ""]
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dataframe.to_csv(f"C://Users/Jawk/PycharmProjects/EEE/EEE/data/result_{self.name}_3_1_{timestamp}.csv")


    def final(self):
        """
        Calculates the cost of power consumption for the factory with a battery.
        Attention: When recalling this function, the wind_power_list and solar_power_list should in standard format.
        :return: Total cost in yuan
        """

        def objective(individual):
            result = []
            for month in range(12):
                temp = 0
                for i in range(24):
                    if 7 <= i <= 22:
                        temp += individual[24 * month * 3 + 3 * i + 1] * power_demand_price
                    else:
                        temp += individual[24 * month * 3 + 3 * i + 1] * 0.4
                if month in [1, 3, 5, 7, 8, 10, 12]:
                    temp *= 31
                elif month in [4, 6, 9, 11]:
                    temp *= 30
                else:
                    temp *= 28
                result.append(temp)
            result.append(individual[864] * battery_power_price / 10)
            result.append(individual[865] * battery_capacity_price / 10)
            # Cost of constructing new wind and solar power plants
            result.append(individual[867] * wind_capacity_price / 5)
            result.append(individual[868] * solar_capacity_price / 5)
            return sum(result),

        constraints = []
        nonlinear_eq_constraints = []
        var_ranges = []

        for month in range(12):
            for i in range(24):

                if 8 <= i <= 16:
                    var_ranges.append((-self.battery_power, 0))
                else:
                    var_ranges.append((0, self.battery_power))

                def battery_power_upper_bound(individual, i=i):
                    return individual[24 * month * 3 + 3 * i] - individual[864]
                constraints.append(battery_power_upper_bound)

                def battery_power_lower_bound(individual, i=i):
                    return -individual[864] - individual[24 * month * 3 + 3 * i]
                constraints.append(battery_power_lower_bound)

                var_ranges.append((0, math.inf))

                def power_grid_lower_bound(individual, i=i):
                    return -individual[24 * month * 3 + 3 * i + 1]
                constraints.append(power_grid_lower_bound)

                var_ranges.append((0.1 * self.battery_capacity, 0.9 * self.battery_capacity))

                def SOC_upper_bound(individual, i=i):
                    return individual[24 * month * 3 + 3 * i + 2] - 0.9 * individual[865]
                constraints.append(SOC_upper_bound)

                def SOC_lower_bound(individual, i=i):
                    return 0.1 * individual[865] - individual[24 * month * 3 + 3 * i + 2]
                constraints.append(SOC_lower_bound)

                if i == 23:
                    def final_SOC_upper_bound(individual, i=i):
                        return individual[24 * month * 3 + 3 * i + 2] - individual[24 * month * 3 + 3 * i] - (individual[866]*1.01) * individual[865]

                    constraints.append(final_SOC_upper_bound)

                    def final_SOC_lower_bound(individual, i=i):
                        return (individual[866]*0.99) * individual[865] - (individual[24 * month * 3 + 3 * i + 2] - individual[24 * month * 3 + 3 * i])

                    constraints.append(final_SOC_lower_bound)

                def power_balance(individual, i=i):
                    wind_power = self.wind_power_list24[month][i] * individual[867]
                    solar_power = self.solar_power_list24[month][i] * individual[868]
                    shortage = self.power_demand_list[i] - wind_power - solar_power
                    if shortage < 0 < individual[24 * month * 3 + 3 * i]:
                        individual[24 * month * 3 + 3 * i] = 0
                    if individual[24 * month * 3 + 3 * i + 1] > 0 > individual[24 * month * 3 + 3 * i]:
                        individual[24 * month * 3 + 3 * i] = 0
                    if individual[24 * month * 3 + 3 * i] > 0:
                        temp = self.power_demand_list[i] - wind_power - solar_power - \
                               individual[24 * month * 3 + 3 * i] * battery_efficiency
                    else:
                        temp = self.power_demand_list[i] - wind_power - solar_power - \
                               individual[24 * month * 3 + 3 * i] / battery_efficiency
                    if temp < 0:
                        individual[24 * month * 3 + 3 * i + 1] = 0
                    else:
                        individual[24 * month * 3 + 3 * i + 1] = temp
                nonlinear_eq_constraints.append(power_balance)

                def SOC_balance(individual, i=i):
                    if i == 0:
                        individual[24 * month * 3 + 3 * i + 2] = individual[866] * individual[865]
                    else:
                        individual[24 * month * 3 + 3 * i + 2] = individual[24 * month * 3 + 3 * (i - 1) + 2] - individual[24 * month * 3 + 3 * (i - 1)]
                nonlinear_eq_constraints.append(SOC_balance)

        var_ranges.append((0.5 * self.battery_power, 2 * self.battery_power))
        var_ranges.append((0.5 * self.battery_capacity, 2 * self.battery_capacity))
        var_ranges.append((0.1, 0.9))
        var_ranges.append((500, 1500))

        def wind_power_lower_bound(individual):
            return -individual[867]
        var_ranges.append((500, 1500))

        def solar_power_lower_bound(individual):
            return -individual[868]
        constraints.append(wind_power_lower_bound)
        constraints.append(solar_power_lower_bound)

        var_ranges.append((500, 1500))

        ga = GeneticAlgorithm(objective_func=objective, constraints=constraints,
                               nonlinear_eq_constraints=nonlinear_eq_constraints, n_vars=24 * 12 * 3 + 5, var_ranges=var_ranges, ngen=1000, pop_size=100, cxpb=0.8, mutpb=0.3, penalty_factor=1e6)
        pop, log, hof = ga.run()
        dataframe = pd.DataFrame(columns=['battery_power', 'power_grid', 'SOC'])
        for month in range(12):
            for i in range(24):
                dataframe.loc[month * 24 + i] = [hof[0][24 * month * 3 + 3 * i], hof[0][24 * month * 3 + 3 * i + 1], hof[0][24 * month * 3 + 3 * i + 2]]
        dataframe.loc[24 * 12] = ["Power", "Capacity", "init_SOC"]
        dataframe.loc[24 * 12 + 1] = [hof[0][864], hof[0][865], hof[0][866]]
        result = hof[0].fitness.values[0]
        dataframe.loc[24 * 12 + 2] = ["result", "wind", "solar"]
        dataframe.loc[24 * 12 + 3] = [result, hof[0][867], hof[0][868]]
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        dataframe.to_csv(f"C://Users/Jawk/PycharmProjects/EEE/EEE/data/result_{self.name}_3_1_{timestamp}.csv")


