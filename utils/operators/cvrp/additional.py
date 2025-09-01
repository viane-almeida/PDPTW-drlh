
from copy import deepcopy
from DRLH.utils.utils import cvrp_objective_function, cvrp_check_best_position_change


def cvrp_remove_nothing(_, solution):
    return solution, [None]


def cvrp_insert_single_best(cvrp, solution):
    min_cost = float('inf')
    best_change = (None, None, None, None)
    for i_demand in range(1, cvrp.instance.n_demands+1):
        cvrp_copy = deepcopy(cvrp)
        v_from = cvrp_copy.d_loc[i_demand]
        temp_solution = deepcopy(solution)
        temp_solution[v_from] = [c for c in temp_solution[v_from] if c != i_demand]
        cost = cvrp_objective_function(cvrp_copy, temp_solution, changed={v_from})  # pdptw.sum_costs is correct now
        for v_to in range(cvrp.instance.n_vehicles):
            ij, add_cost = cvrp_check_best_position_change(cvrp_copy, temp_solution, i_demand, v_to)  # fix add_cost to be correct. FIXED
            if cost+add_cost < min_cost:
                min_cost = cost+add_cost
                best_change = (v_from, v_to, i_demand, ij)

    best_v_from, best_v_to, best_c, best_i = best_change
    solution[best_v_from] = [c for c in solution[best_v_from] if c != best_c]
    solution[best_v_to].insert(best_i, best_c)
    cost = cvrp_objective_function(cvrp, solution, changed={best_v_from, best_v_to})  # fixed mistake here
    return solution, cost