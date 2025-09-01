
import random
import numpy as np
from copy import copy, deepcopy
from collections import Counter
from utils.utils.utils import pdptw_objective_function, pdptw_calculate_tour_deviation


# Remove operators
# ==================================================================================================================

def pdptw_remove_calls(pdptw, solution, r_calls):
    r_calls_set = set(r_calls)
    r_vehicles = set()
    for r in r_calls_set:
        pdptw.calls_counter[r] += 1
        v = pdptw.call_loc[r]
        r_vehicles.add(v)
    for v in r_vehicles:
        solution[v] = [e for e in solution[v] if e not in r_calls_set]
    _ = pdptw_objective_function(pdptw, solution, changed=r_vehicles)  # pdptw.sum_costs is correct now
    return solution, r_calls

def ptptw_remove_nothing(pdptw, solution):
    return solution, [None]

def pdptw_remove_single_best(pdptw, solution):
    return solution, [None]

def pdptw_remove_longest_tour_deviation_s(pdptw, solution):
    tour_deviation = pdptw_calculate_tour_deviation(pdptw, solution)
    n_remove = random.randint(min(pdptw.n_calls, 5), min(pdptw.n_calls, 10))
    r_calls = list(dict.fromkeys([c for (c, _) in tour_deviation[:n_remove]]))
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)

def pdptw_remove_longest_tour_deviation_l(pdptw, solution):
    tour_deviation = pdptw_calculate_tour_deviation(pdptw, solution)
    n_remove = random.randint(min(pdptw.n_calls, 20), min(pdptw.n_calls, 30))
    r_calls = list(dict.fromkeys([c for (c, _) in tour_deviation[:n_remove]]))
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_tour_neighbors(pdptw, solution):
    non_empty_vehicles = [v for (v, v_sol) in enumerate(solution) if len(v_sol) > 0]
    v = random.sample(non_empty_vehicles, 1)[0]
    v_solution = solution[v]
    len_v_solution = len(v_solution)
    len_remove_segment = random.randint(min(len_v_solution, 2), min(len_v_solution, 5))
    start_segment = random.randint(0, len_v_solution-len_remove_segment)
    end_segment = start_segment+len_remove_segment
    remove_segment = v_solution[start_segment:end_segment]
    r_calls = list(dict.fromkeys([r for r in remove_segment]))
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_xs(pdptw, solution):
    r_calls = random.sample(range(pdptw.n_calls), random.randint(min(pdptw.n_calls, 2), min(pdptw.n_calls, 5)))
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_s(pdptw, solution):
    r_calls = random.sample(range(pdptw.n_calls), random.randint(min(pdptw.n_calls, 5), min(pdptw.n_calls, 10)))
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_m(pdptw, solution):
    r_calls = random.sample(range(pdptw.n_calls), random.randint(min(pdptw.n_calls, 10), min(pdptw.n_calls, 20)))
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_l(pdptw, solution):
    r_calls = random.sample(range(pdptw.n_calls), random.randint(min(pdptw.n_calls, 20), min(pdptw.n_calls, 30)))
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_xl(pdptw, solution):
    r_calls = random.sample(range(pdptw.n_calls), random.randint(min(pdptw.n_calls, 30), min(pdptw.n_calls, 40)))
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_xxl(pdptw, solution):
    r_calls = random.sample(range(pdptw.n_calls), random.randint(min(pdptw.n_calls, 80), min(pdptw.n_calls, 100)))
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_least_frequent_s(pdptw, solution):
    r_calls = sorted(list(enumerate(pdptw.calls_counter)), key=lambda x: (x[1], random.random()))[:10]
    r_calls = list(list(zip(*r_calls))[0])
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_least_frequent_m(pdptw, solution):
    r_calls = sorted(list(enumerate(pdptw.calls_counter)), key=lambda x: (x[1], random.random()))[:20]
    r_calls = list(list(zip(*r_calls))[0])
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)

def pdptw_remove_least_frequent_xl(pdptw, solution):
    r_calls = sorted(list(enumerate(pdptw.calls_counter)), key=lambda x: (x[1], random.random()))[:40]
    r_calls = list(list(zip(*r_calls))[0])
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_one_vehicle(pdptw, solution):
    non_empty_vehicles = [v for (v, v_sol) in enumerate(solution) if len(v_sol) > 0]
    v = random.sample(non_empty_vehicles, 1)[0]
    r_calls = list(set([r for r in solution[v]]))
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)


def pdptw_remove_two_vehicles(pdptw, solution):
    non_empty_vehicles = [v for (v, v_sol) in enumerate(solution) if len(v_sol) > 0]
    v = random.sample(non_empty_vehicles, min(len(non_empty_vehicles), 2))
    #r_calls = list(set([r for r in solution[v[0]]] + [r for r in solution[v[1]]]))
    r_calls = list(set(sum((solution[vv] for vv in v), [])))
    random.shuffle(r_calls)
    return pdptw_remove_calls(pdptw, solution, r_calls)
