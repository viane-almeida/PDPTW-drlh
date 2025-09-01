import copy
import random
import numpy as np
from copy import deepcopy
from DRLH.utils.utils import cvrp_objective_function, cvrp_check_first_position_change, \
    cvrp_beam_check_best_position_change, cvrp_check_best_position_change


def cvrp_insert_first(cvrp, solution, removed_demands):
    for i_demand in removed_demands:
        for v in sorted(range(cvrp.instance.n_vehicles), key=lambda _: random.random()):
            ij, _ = cvrp_check_first_position_change(cvrp, solution, i_demand, v)
            if ij is not None:
                solution[v].insert(ij, i_demand)
                cost = cvrp_objective_function(cvrp, solution, changed={v})
                break
    return solution, cost


def cvrp_insert_greedy(cvrp, solution, removed_demands):
    for i_demand in removed_demands:
        results = []
        for v in range(cvrp.instance.n_vehicles):
            ij, cost = cvrp_check_best_position_change(cvrp, solution, i_demand, v)
            results.append((v, ij, cost))
        v, ij, cost = min(results, key=lambda x: x[2])
        v_solution = solution[v]
        v_solution.insert(ij, i_demand)
        cost = cvrp_objective_function(cvrp, solution, changed={v})
    return solution, cost


def cvrp_insert_beam_search(cvrp, solution, removed_demands, beam_width=10, search_width=5):
    beam = [(solution, cvrp)]

    for i_demand in removed_demands:
        next_beam = []
        for sol, beam_cvrp in beam:
            v_i_cost_list = []
            for v in range(cvrp.instance.n_vehicles):
                v_i_cost_list += cvrp_beam_check_best_position_change(beam_cvrp, sol, i_demand, v)
            v_i_cost_list = sorted(v_i_cost_list, key=lambda x: (x[2], random.random()))[:search_width]
            for v, i, add_cost in v_i_cost_list:
                new_sol = deepcopy(sol)
                new_beam_cvrp = deepcopy(beam_cvrp)
                new_sol[v].insert(i, i_demand)
                cost = cvrp_objective_function(new_beam_cvrp, new_sol, changed={v})  # for updating costs/lpat/cum_weights/cum_leave_times/node_types
                next_beam.append((new_sol, new_beam_cvrp))
        next_beam = sorted(next_beam, key=lambda x: (x[1].sum_costs, random.random()))[:beam_width]  # beam width
        beam = next_beam
    best_solution, best_cvrp = beam[0]
    cvrp.update_from_other(best_cvrp)
    return best_solution, cvrp.sum_costs


def cvrp_insert_by_variance(cvrp, solution, removed_demands):  # 2N --> O(N)
    all_results = []
    for i_demand in removed_demands:
        results_per_demand = []
        for v in range(cvrp.instance.n_vehicles):
            _, add_cost = cvrp_check_best_position_change(cvrp, solution, i_demand, v)
            results_per_demand.append(add_cost)
        results_per_demand = sorted(r for r in results_per_demand if r<float('inf'))
        var = np.var(results_per_demand[:10])  # get variance of best 10 positions. Higher variance means that the call should have priority in insertion order
        all_results.append((i_demand, var))

    all_results = sorted(all_results, key=lambda x: (-x[1], random.random()))
    for i_demand, _ in all_results:
        results = []
        for v in range(cvrp.instance.n_vehicles):
            ij, cost = cvrp_check_best_position_change(cvrp, solution, i_demand, v)
            results.append((v, ij, cost))
        v, ij, cost = min(results, key=lambda x: (x[2], random.random()))
        v_solution = solution[v]
        v_solution.insert(ij, i_demand)
        cost = cvrp_objective_function(cvrp, solution, changed={v})
    return solution, cost


