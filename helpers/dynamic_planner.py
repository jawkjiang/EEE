from gurobipy import Model, GRB


def solve_dynamic_programming(stages, states, decisions, transitions, costs, initial_state, objective='minimize'):
    """
    Solves a dynamic programming problem using Gurobi.

    Args:
        stages (int): Number of stages in the dynamic program.
        states (list): List of possible states at each stage.
        decisions (list): List of possible decisions at each stage.
        transitions (function): Function to determine the next state given the current state and decision.
        costs (function): Function to determine the cost given the current state, decision, and stage.
        initial_state (any): Initial state at stage 0.
        objective (str): 'minimize' to minimize cost or 'maximize' to maximize reward.

    Returns:
        dict: Optimal policy (decision) at each stage and state.
    """
    model = Model("Dynamic_Programming")

    # Decision variables
    decision_vars = {}
    for t in range(stages):
        for s in states[t]:
            for d in decisions[t]:
                decision_vars[(t, s, d)] = model.addVar(vtype=GRB.BINARY, name=f"d_{t}_{s}_{d}")

    # Add constraints
    # Initial state constraint
    model.addConstr(sum(decision_vars[(0, initial_state, d)] for d in decisions[0]) == 1)

    # Transition constraints
    for t in range(stages - 1):
        for s in states[t]:
            for d in decisions[t]:
                next_state = transitions(t, s, d)
                if next_state in states[t + 1]:
                    model.addConstr(sum(decision_vars[(t + 1, next_state, d_next)] for d_next in decisions[t + 1]) >= decision_vars[(t, s, d)])

    # Objective function
    objective_expr = sum(costs(t, s, d) * decision_vars[(t, s, d)] for t in range(stages) for s in states[t] for d in decisions[t])
    if objective == 'minimize':
        model.setObjective(objective_expr, GRB.MINIMIZE)
    else:
        model.setObjective(objective_expr, GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Check and return results
    if model.status == GRB.OPTIMAL:
        policy = {}
        for t in range(stages):
            for s in states[t]:
                for d in decisions[t]:
                    if decision_vars[(t, s, d)].X > 0.5:
                        if (t, s) not in policy:
                            policy[(t, s)] = d
        return policy
    else:
        return None


# Example usage
if __name__ == '__main__':
    def transitions(t, s, d):
        if t == 0:
            return s + d
        else:
            return s + d - 1

    def costs(t, s, d):
        return s * d + t

    stages = 3
    states = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    decisions = [[0, 1], [0, 1], [0, 1]]
    initial_state = 0
    policy = solve_dynamic_programming(stages, states, decisions, transitions, costs, initial_state, objective='minimize')
    print(policy)
