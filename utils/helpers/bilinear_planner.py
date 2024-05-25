from gurobipy import Model, GRB, QuadExpr
import math


def solve_bilinear_program_only_target_bilinearized(obj_coeffs_vars, constraints, var_bounds, maximize=True):
    """
    Solves a bilinear programming problem using Gurobi.

    Args:
        obj_coeffs_vars (dict): Dictionary where keys are tuples of variable names representing bilinear terms,
                                e.g., {('x', 'z'): 1, ('y', 'z'): 2}.
        constraints (list): List of constraints, each represented as a tuple (coeff_dict, sense, rhs).
                            e.g., [({'x': 1, 'y': 2}, '<=', 14), ({'x': 3, 'y': -1}, '<=', 18)]
        var_bounds (dict): Dictionary of variable bounds, e.g., {'x': (0, math.inf), 'y': (0, math.inf), 'z': (0, math.inf)}.
        maximize (bool): True if maximizing, False if minimizing.

    Returns:
        tuple: Optimal value of the objective function and values of variables if solution is found.
               None if no solution is found.
    """
    # Create a new model
    model = Model("bilinear_model")

    # Create variables
    variables = {}
    for var_name, bounds in var_bounds.items():
        variables[var_name] = model.addVar(lb=bounds[0], ub=bounds[1], name=var_name)

    # Set bilinear objective
    objective = QuadExpr()
    for (var1, var2), coeff in obj_coeffs_vars.items():
        objective += coeff * variables[var1] * variables[var2]

    if maximize:
        model.setObjective(objective, GRB.MAXIMIZE)
    else:
        model.setObjective(objective, GRB.MINIMIZE)


    # Add constraints
    for coeff_dict, sense, rhs in constraints:
        linear_expr = sum(coeff_dict[var] * variables[var] for var in coeff_dict)
        if sense == '<=':
            model.addConstr(linear_expr <= rhs)
        elif sense == '>=':
            model.addConstr(linear_expr >= rhs)
        else:
            model.addConstr(linear_expr == rhs)

    # Optimize the model
    model.optimize()

    # Check and return results
    if model.status == GRB.OPTIMAL:
        return (model.objVal, {v.varName: v.x for v in model.getVars()})
    else:
        return None


from gurobipy import Model, GRB, QuadExpr

def solve_linear_program_with_bilinear_constraints(obj_coeffs, constraints, bilinear_constraints, var_bounds, maximize=True):
    """
    Solves a linear programming problem with bilinear constraints using Gurobi.

    Args:
        obj_coeffs (dict): Dictionary of coefficients for the linear objective function, e.g., {'x': 3, 'y': 4}.
        constraints (list): List of constraints, each represented as a tuple (coeff_dict, sense, rhs).
                            e.g., [({'x': 1, 'y': 2}, '<=', 14), ({'x': 3, 'y': -1}, '<=', 18)]
        bilinear_constraints (list): List of bilinear constraints, each represented as a tuple (var1, var2, coeff, sense, rhs).
                                     e.g., [('x', 'z', 1, '<=', 10), ('y', 'z', 1, '<=', 20)]
        var_bounds (dict): Dictionary of variable bounds, e.g., {'x': (0, math.inf), 'y': (0, math.inf), 'z': (0, math.inf)}.
        maximize (bool): True if maximizing, False if minimizing.

    Returns:
        tuple: Optimal value of the objective function and values of variables if solution is found.
               None if no solution is found.
    """
    # Create a new model
    model = Model("lp_model_with_bilinear_constraints")

    # Create variables
    variables = {}
    for var_name, bounds in var_bounds.items():
        variables[var_name] = model.addVar(lb=bounds[0], ub=bounds[1], name=var_name)

    # Set linear objective
    objective = sum(coeff * variables[var] for var, coeff in obj_coeffs.items())
    if maximize:
        model.setObjective(objective, GRB.MAXIMIZE)
    else:
        model.setObjective(objective, GRB.MINIMIZE)

    # Add linear constraints
    for coeff_dict, sense, rhs in constraints:
        linear_expr = sum(coeff_dict[var] * variables[var] for var in coeff_dict)
        if sense == '<=':
            model.addConstr(linear_expr <= rhs)
        elif sense == '>=':
            model.addConstr(linear_expr >= rhs)
        else:
            model.addConstr(linear_expr == rhs)

    # Add bilinear constraints
    for var1, var2, coeff, sense, rhs in bilinear_constraints:
        bilinear_expr = coeff * variables[var1] * variables[var2]
        if sense == '<=':
            model.addQConstr(bilinear_expr <= rhs)
        elif sense == '>=':
            model.addQConstr(bilinear_expr >= rhs)
        else:
            model.addQConstr(bilinear_expr == rhs)

    # Optimize the model
    model.optimize()

    # Check and return results
    if model.status == GRB.OPTIMAL:
        return (model.objVal, {v.varName: v.x for v in model.getVars()})
    else:
        return None


if __name__ == '__main__':
    obj_coeffs = {'x': 3, 'y': 4}
    constraints = [
        ({'x': 1, 'y': 2}, '<=', 14),
        ({'x': 3, 'y': -1}, '<=', 18)
    ]
    bilinear_constraints = [
        ('x', 'z', 1, '<=', 10),
        ('y', 'z', 1, '<=', 20)
    ]
    var_bounds = {'x': (0, math.inf), 'y': (0, math.inf), 'z': (0, math.inf)}

    result = solve_linear_program_with_bilinear_constraints(obj_coeffs, constraints, bilinear_constraints, var_bounds,
                                                            maximize=True)

    if result:
        objective_value, var_values = result
        print(f"Optimal objective value: {objective_value}")
        for var, value in var_values.items():
            print(f"{var}: {value}")
    else:
        print("No optimal solution found.")
