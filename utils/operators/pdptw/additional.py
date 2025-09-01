
from copy import copy, deepcopy
from utils.utils.utils import pdptw_objective_function, pdptw_check_best_position_change


def ptptw_remove_nothing(pdptw, solution):
    return solution, [None]


def pdptw_insert_single_best(pdptw, solution, _=None):
    min_cost = float('inf')
    best_change = (None, None, None, None)
    for i_call in range(pdptw.n_calls):
        pdptw_copy = deepcopy(pdptw)
        v_from = pdptw_copy.call_loc[i_call]
        temp_solution = deepcopy(solution)
        temp_solution[v_from] = [c for c in temp_solution[v_from] if c != i_call]
        cost = pdptw_objective_function(pdptw_copy, temp_solution, changed={v_from})  # pdptw.sum_costs is correct now
        for v_to in pdptw_copy.calls_compatibility[i_call]:
            ij, add_cost = pdptw_check_best_position_change(pdptw_copy, temp_solution, i_call, v_to)  # fix add_cost to be correct. FIXED
            if cost+add_cost < min_cost:
                min_cost = cost+add_cost
                best_change = (v_from, v_to, i_call, ij[0], ij[1])

    best_v_from, best_v_to, best_c, best_i, best_j = best_change
    solution[best_v_from] = [c for c in solution[best_v_from] if c != best_c]
    solution[best_v_to].insert(best_i, best_c)
    solution[best_v_to].insert(best_j, best_c)
    cost = pdptw_objective_function(pdptw, solution, changed={best_v_from, best_v_to})  # fixed mistake here
    pdptw.calls_counter[best_c] += 1
    return solution, cost
