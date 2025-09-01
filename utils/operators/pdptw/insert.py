
import random
import numpy as np
from copy import copy, deepcopy
from collections import Counter
from utils.utils.utils import pdptw_objective_function, pdptw_check_best_position_change, pdptw_beam_check_best_position_change, \
    pdptw_check_first_position_change, pdptw_objective_function_rearange, pdptw_beam_check_best_position_change_rearange


# Insert operators
# ==================================================================================================================

def pdptw_insert_first(pdptw, solution, removed_calls):
    for i_call in removed_calls:
        compatible_vehicles = list(pdptw.calls_compatibility[i_call])
        random.shuffle(compatible_vehicles)
        for v in compatible_vehicles:
            ij, _ = pdptw_check_first_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                break
    return solution, cost


def pdptw_insert_first2(pdptw, solution, removed_calls):
    for i_call in removed_calls:
        for v in pdptw.calls_compatibility[i_call]:
            ij, _ = pdptw_check_first_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                break
    return solution, cost


def pdptw_insert_greedy(pdptw, solution, removed_calls):
    for i_call in removed_calls:
        results = []
        for v in pdptw.calls_compatibility[i_call]:
            ij, cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            results.append((v, ij, cost))
        v, ij, cost = min(results, key=lambda x: x[2])
        v_solution = solution[v]
        v_solution.insert(ij[0], i_call)
        v_solution.insert(ij[1], i_call)
        cost = pdptw_objective_function(pdptw, solution, changed={v})
    return solution, cost


def pdptw_insert_beam_search(pdptw, solution, removed_calls, beam_width=10, search_width=5):
    beam = [(solution, pdptw)]  # Can I deepcopy the pdptw object and put it in the beam?

    for i_call in removed_calls:
        next_beam = []
        for sol, beam_pdptw in beam:

            v_i_j_cost_list = []
            for v in beam_pdptw.calls_compatibility[i_call]:
                v_i_j_cost_list += pdptw_beam_check_best_position_change(beam_pdptw, sol, i_call, v)
            v_i_j_cost_list = sorted(v_i_j_cost_list, key=lambda x: x[3])[:search_width]

            for v, i, j, add_cost in v_i_j_cost_list:
                new_sol = deepcopy(sol)
                new_beam_pdptw = deepcopy(beam_pdptw)
                new_sol[v].insert(i, i_call)
                new_sol[v].insert(j, i_call)
                cost = pdptw_objective_function(new_beam_pdptw, new_sol, changed={v})  # for updating costs/lpat/cum_weights/cum_leave_times/node_types
                next_beam.append((new_sol, new_beam_pdptw))
        next_beam = sorted(next_beam, key=lambda x: (x[1].sum_costs, random.random()))[:beam_width]  # beam width
        beam = next_beam

    best_solution, best_pdptw = beam[0]
    pdptw.update_from_other(best_pdptw)
    return best_solution, pdptw.sum_costs


def pdptw_insert_by_variance(pdptw, solution, removed_calls):  # 2N --> O(N)
    all_results = []
    for i_call in removed_calls:
        results_per_call = []
        for v in pdptw.calls_compatibility[i_call]:
            _, add_cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            results_per_call.append(add_cost)
        results_per_call = sorted(r for r in results_per_call if r<float('inf'))
        var = np.var(results_per_call[:10])  # get variance of best 10 positions. Higher variance means that the call should have priority in insertion order
        all_results.append((i_call, var))

    all_results = sorted(all_results, key=lambda x: (-x[1], random.random()))
    for i_call, _ in all_results:
        results = []
        for v in pdptw.calls_compatibility[i_call]:
            ij, cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            results.append((v, ij, cost))
        v, ij, cost = min(results, key=lambda x: x[2])
        v_solution = solution[v]
        v_solution.insert(ij[0], i_call)
        v_solution.insert(ij[1], i_call)
        cost = pdptw_objective_function(pdptw, solution, changed={v})
    return solution, cost



def pdptw_insert_least_loaded_vehicle(pdptw, solution, removed_calls):
    v_load = []
    for v in range(pdptw.n_vehicles):
        v_load.append(-sum(cw-pdptw.vehicles[v][2] for cw in pdptw.cum_weights[v])/len(pdptw.cum_weights))
    v_load = sorted(list(enumerate(v_load)), key=lambda x: (-x[1], random.random())) + [(pdptw.n_vehicles, -float('inf'))]

    for i_call in removed_calls:
        for v, _ in v_load:
            if v not in pdptw.calls_compatibility[i_call]:
                continue
            ij, _ = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                break
    return solution, cost


def pdptw_insert_least_active_vehicle(pdptw, solution, removed_calls):
    active_times = []
    for v in range(pdptw.n_vehicles):
        current_node, active_time, _ = pdptw.vehicles[v]
        for i, c in enumerate(solution[v]):
            next_node = pdptw.calls[c][pdptw.node_types[v][i]]
            node_type = pdptw.node_types[v][i]
            drive_time = pdptw.dist_matrix[v][current_node][next_node]
            extra_time = pdptw.wait_times[v][c][node_type]
            active_time += (drive_time + extra_time)
            current_node = next_node
        active_times.append(active_time)
    v_active = sorted(list(enumerate(active_times)), key=lambda x: (x[1], random.random())) + [(pdptw.n_vehicles, float('inf'))]

    for i_call in removed_calls:
        for v, _ in v_active:
            if v not in pdptw.calls_compatibility[i_call]:
                continue
            ij, _ = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                break
    return solution, cost


def pdptw_insert_close_vehicle(pdptw, solution, removed_calls):
    for i_call in removed_calls:
        close_vehicles = []
        for v in pdptw.calls_compatibility[i_call]:
            if v == pdptw.n_vehicles:
                close_vehicles.append((v, float('inf')))
                continue
            dist_pickup = pdptw.dist_matrix[v][pdptw.calls[i_call][0]][pdptw.vehicles[v][0]]
            dist_delivery = pdptw.dist_matrix[v][pdptw.calls[i_call][1]][pdptw.vehicles[v][0]]
            close_vehicles.append((v, dist_pickup+dist_delivery))
        close_vehicles = sorted(close_vehicles, key=lambda x: (x[1], random.random()))
        for v, _ in close_vehicles:
            ij, cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                break
    return solution, cost


def pdptw_insert_dummy(pdptw, solution, removed_calls):
    dummy_vehicle = pdptw.n_vehicles
    for i_call in removed_calls:
        solution[dummy_vehicle].insert(0, i_call)
        solution[dummy_vehicle].insert(1, i_call)
        cost = pdptw_objective_function(pdptw, solution, changed={dummy_vehicle})
    return solution, cost


def pdptw_insert_group(pdptw, solution, removed_calls):
    comp = Counter()
    for i_call in removed_calls:
        comp.update(pdptw.calls_compatibility[i_call])
    comp.update({pdptw.n_vehicles: -len(removed_calls)+0.5})

    inserted = set()
    for v, _ in comp.most_common():
        for i_call in removed_calls:
            if i_call in inserted:
                continue
            if v not in pdptw.calls_compatibility[i_call]:
                continue
            ij, cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                inserted.add(i_call)
        if len(inserted) == len(removed_calls):
            break
    return solution, cost


def pdptw_insert_by_difficulty(pdptw, solution, removed_calls):
    removed_calls = sorted(removed_calls, key=lambda x: (pdptw.call_difficulty[x], random.random()))
    for i_call in removed_calls:
        results = []
        for v in pdptw.calls_compatibility[i_call]:
            ij, cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            results.append((v, ij, cost))
        v, ij, cost = min(results, key=lambda x: x[2])
        v_solution = solution[v]
        v_solution.insert(ij[0], i_call)
        v_solution.insert(ij[1], i_call)
        cost = pdptw_objective_function(pdptw, solution, changed={v})
    return solution, cost


def pdptw_insert_best_fit(pdptw, solution, removed_calls):
    comp = Counter()
    for i_call in removed_calls:
        comp.update(pdptw.calls_compatibility[i_call])
    comp.update({pdptw.n_vehicles: 1})  # need to make it so that dummy is the LAST option

    for i_call in removed_calls:
        for v, _ in comp.most_common()[::-1]:
            if v not in pdptw.calls_compatibility[i_call]:
                continue
            ij, cost = pdptw_check_best_position_change(pdptw, solution, i_call, v)
            if ij is not None:
                solution[v].insert(ij[0], i_call)
                solution[v].insert(ij[1], i_call)
                cost = pdptw_objective_function(pdptw, solution, changed={v})
                break
    return solution, cost


def pdptw_rearange_vehicles(pdptw, solution, _=None, beam_width=10, search_width=5):
    for v in range(pdptw.n_vehicles):
        r_calls_set = set(solution[v])
        for r in r_calls_set:
            pdptw.calls_counter[r] += 1
        temp_solution = deepcopy(solution)
        temp_solution[v] = []
        temp_pdptw = deepcopy(pdptw)
        _ = pdptw_objective_function(temp_pdptw, temp_solution, changed={v})  # pdptw.sum_costs is correct now
        removed_calls = list(r_calls_set)
        random.shuffle(removed_calls)
        beam = [(temp_solution, temp_pdptw)]  # Can I deepcopy the pdptw object and put it in the beam?
        for i_call in removed_calls:
            next_beam = []
            for sol, beam_pdptw in beam:
                v_i_j_cost_list = pdptw_beam_check_best_position_change(beam_pdptw, sol, i_call, v)
                v_i_j_cost_list = sorted(v_i_j_cost_list, key=lambda x: x[3])[:search_width]
                for v, i, j, add_cost in v_i_j_cost_list:
                    new_sol = deepcopy(sol)
                    new_beam_pdptw = deepcopy(beam_pdptw)
                    new_sol[v].insert(i, i_call)
                    new_sol[v].insert(j, i_call)
                    cost = pdptw_objective_function(new_beam_pdptw, new_sol, changed={v})  # for updating costs/lpat/cum_weights/cum_leave_times/node_types
                    next_beam.append((new_sol, new_beam_pdptw))
            next_beam = sorted(next_beam, key=lambda x: x[1].sum_costs)[:beam_width]  # beam width
            beam = next_beam
        beam.append((solution, pdptw))
        beam = sorted(beam, key=lambda x: x[1].sum_costs)
        best_solution, best_pdptw = beam[0]
        pdptw.update_from_other(best_pdptw)
    return best_solution, pdptw.sum_costs


def pdptw_rearange_vehicles_fast(pdptw, solution, _=None, beam_width=10, search_width=10):
    for v in range(pdptw.n_vehicles):
        if len(solution[v]) <= 2:
            continue
        removed_calls = list(set(solution[v]))
        random.shuffle(removed_calls)
        for r in removed_calls:
            pdptw.calls_counter[r] += 1
        v_solution = []
        v_pdptw = pdptw.create_v_pdptw(v=v)
        _ = pdptw_objective_function_rearange(pdptw, v_pdptw, v_solution, v)
        beam = [(v_solution, v_pdptw)]
        for i_call in removed_calls:
            next_beam = []
            for v_sol, beam_v_pdptw in beam:
                v_i_j_cost_list = pdptw_beam_check_best_position_change_rearange(pdptw, beam_v_pdptw, v_sol, i_call, v)
                v_i_j_cost_list = sorted(v_i_j_cost_list, key=lambda x: x[3])[:search_width]
                for v, i, j, add_cost in v_i_j_cost_list:
                    new_v_sol = copy(v_sol)
                    new_beam_v_pdptw = deepcopy(beam_v_pdptw)
                    new_v_sol.insert(i, i_call)
                    new_v_sol.insert(j, i_call)
                    cost = pdptw_objective_function_rearange(pdptw, new_beam_v_pdptw, new_v_sol, v)
                    next_beam.append((new_v_sol, new_beam_v_pdptw))
            next_beam = sorted(next_beam, key=lambda x: x[1].cost)[:beam_width]
            beam = next_beam
        beam = sorted(beam, key=lambda x: x[1].cost)
        if len(beam) > 0:
            best_v_solution, best_v_pdptw = beam[0]
            if best_v_pdptw.cost < pdptw.costs[v]:
                pdptw.update_from_v_pdptw(best_v_pdptw, v)
                solution[v] = best_v_solution
    return solution, pdptw.sum_costs

