from copy import deepcopy
import random
from DRLH.utils.utils import cvrp_objective_function, cvrp_calculate_tour_deviation

def cvrp_remove_insert(cvrp, solution, op, prev_accepted=True):
    cvrp.accepted(accepted=prev_accepted)  # accepting/rejecting current/prev variable values based on prev_accepted
    solution = deepcopy(solution)
    op, op2 = op
    solution, removed_demands = op(cvrp, solution)
    new_solution, new_cost = op2(cvrp, solution, removed_demands)
    return new_solution, new_cost

def cvrp_remove_demands(cvrp, solution, r_demands):
    r_demands_set = set(r_demands)
    r_vehicles = set()
    for r in r_demands_set:
        v = cvrp.d_loc[r]
        r_vehicles.add(v)
    for v in r_vehicles:
        solution[v] = [e for e in solution[v] if e not in r_demands_set]
    _ = cvrp_objective_function(cvrp, solution, changed=r_vehicles)  # pdptw.sum_costs is correct now
    return solution, r_demands


def cvrp_remove_nothing(cvrp, solution):
    return solution, [None]


def cvrp_remove_longest_tour_deviation_s(cvrp, solution):
    tour_deviation = cvrp_calculate_tour_deviation(cvrp, solution)
    n_remove = random.randint(min(cvrp.instance.n_demands, 5), min(cvrp.instance.n_demands, 10))
    r_demands = list(dict.fromkeys([c for (c, _) in tour_deviation[:n_remove]]))
    random.shuffle(r_demands)
    return cvrp_remove_demands(cvrp, solution, r_demands)


def cvrp_remove_tour_neighbors(cvrp, solution):
    non_empty_vehicles = [v for (v, v_sol) in enumerate(solution) if len(v_sol) > 0]
    v = random.sample(non_empty_vehicles, 1)[0]
    v_solution = solution[v]
    len_v_solution = len(v_solution)
    len_remove_segment = random.randint(min(len_v_solution, 2), min(len_v_solution, 5))
    start_segment = random.randint(0, len_v_solution-len_remove_segment)
    end_segment = start_segment+len_remove_segment
    remove_segment = v_solution[start_segment:end_segment]
    r_demands = list(dict.fromkeys([r for r in remove_segment]))
    random.shuffle(r_demands)
    return cvrp_remove_demands(cvrp, solution, r_demands)


def cvrp_remove_xs(cvrp, solution):
    r_demands = random.sample(range(1, cvrp.instance.n_demands+1), random.randint(min(cvrp.instance.n_demands, 2), min(cvrp.instance.n_demands, 5)))
    return cvrp_remove_demands(cvrp, solution, r_demands)


def cvrp_remove_s(cvrp, solution):
    r_demands = random.sample(range(1, cvrp.instance.n_demands+1), random.randint(min(cvrp.instance.n_demands, 5), min(cvrp.instance.n_demands, 10)))
    return cvrp_remove_demands(cvrp, solution, r_demands)


def cvrp_remove_m(cvrp, solution):
    r_demands = random.sample(range(1, cvrp.instance.n_demands+1), random.randint(min(cvrp.instance.n_demands, 10), min(cvrp.instance.n_demands, 20)))
    return cvrp_remove_demands(cvrp, solution, r_demands)


def cvrp_remove_l(cvrp, solution):
    r_demands = random.sample(range(1, cvrp.instance.n_demands+1), random.randint(min(cvrp.instance.n_demands, 20), min(cvrp.instance.n_demands, 30)))
    return cvrp_remove_demands(cvrp, solution, r_demands)


def cvrp_remove_xl(cvrp, solution):
    r_demands = random.sample(range(1, cvrp.instance.n_demands+1), random.randint(min(cvrp.instance.n_demands, 30), min(cvrp.instance.n_demands, 40)))
    return cvrp_remove_demands(cvrp, solution, r_demands)