from gurobipy import Model, GRB
import math


def solve_linear_program(obj_coeffs, constraints, var_bounds, maximize=True):
    """
    Solves a linear programming problem using Gurobi.

    Args:
        obj_coeffs (dict): Dictionary of coefficients for the objective function, e.g., {'x': 3, 'y': 4}.
        constraints (list): List of constraints, each represented as a tuple (coeff_dict, sense, rhs).
                            e.g., [({'x': 1, 'y': 2}, '<=', 14), ({'x': 3, 'y': -1}, '<=', 18)]
        var_bounds (dict): Dictionary of variable bounds, e.g., {'x': (0, math.inf), 'y': (0, math.inf)}.
        maximize (bool): True if maximizing, False if minimizing.

    Returns:
        tuple: Optimal value of the objective function and values of variables if solution is found.
               None if no solution is found.
    """
    # Create a new model
    model = Model("lp_model")

    # Create variables
    variables = {}
    for var_name, bounds in var_bounds.items():
        variables[var_name] = model.addVar(lb=bounds[0], ub=bounds[1], name=var_name)

    # Set objective
    objective = sum(coeff * variables[var] for var, coeff in obj_coeffs.items())
    if maximize:
        model.setObjective(objective, GRB.MAXIMIZE)
    else:
        model.setObjective(objective, GRB.MINIMIZE)

    # Add constraints
    for coeff_dict, sense, rhs in constraints:
        if sense == '<=':
            model.addConstr(sum(coeff_dict[var] * variables[var] for var in coeff_dict) <= rhs)
        elif sense == '>=':
            model.addConstr(sum(coeff_dict[var] * variables[var] for var in coeff_dict) >= rhs)
        else:
            model.addConstr(sum(coeff_dict[var] * variables[var] for var in coeff_dict) == rhs)

    # Optimize the model
    model.optimize()

    # Check and return results
    if model.status == GRB.OPTIMAL:
        return (model.objVal, {v.varName: v.x for v in model.getVars()})
    else:
        return None


if __name__ == '__main__':
    obj_coeffs = {'x': 3, 'y': 4}
    constraints = [({'x': 1, 'y': 2}, '<=', 14), ({'x': 3, 'y': -1}, '<=', 18)]
    var_bounds = {'x': (0, math.inf), 'y': (0, math.inf)}
    result = solve_linear_program(obj_coeffs, constraints, var_bounds)
    print(result)
