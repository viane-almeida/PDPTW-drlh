import os
import numpy as np
import random
from random import shuffle, sample
from collections import defaultdict, Counter
from bisect import bisect


EPSILON = 1e-5


# euclidean distance
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def pdp_get_weights(pdp, solution):
    current_weight = 0
    weights = [0]
    for e in solution:
        current_weight += pdp.capacities[e-1]
        weights.append(current_weight)
    weights.append(0)
    return weights


def pdp_embed_solution(pdp, solution):
    embedded_solution = np.zeros((len(solution), 11))
    weights = pdp_get_weights(pdp, solution)
    solution = [0] + solution[:] + [solution[-1]]
    for index in range(1, pdp.size):
        customer = solution[index]
        embedded_input = []
        embedded_input.append(pdp.capacities[customer-1])  # INSTANCE : capacity
        embedded_input.extend(pdp.locations[customer])  # INSTANCE : location
        embedded_input.append(1 - weights[index])  # SOLUTION : current free capacity
        embedded_input.extend(pdp.locations[solution[index - 1]])  # SOLUTION : location of prev
        embedded_input.extend(pdp.locations[solution[index + 1]])  # SOLUTION : location of next
        embedded_input.append(pdp.dist_matrix[solution[index - 1], customer])  # SOLUTION : dist from prev to node
        embedded_input.append(pdp.dist_matrix[customer, solution[index + 1]])  # SOLUTION : dist from node to next
        embedded_input.append(pdp.dist_matrix[solution[index - 1], solution[index + 1]])  # SOLUTION : dist if current node removed
        for embedded_input_index in range(len(embedded_input)):
            embedded_solution[customer - 1, embedded_input_index] = embedded_input[embedded_input_index]
    return embedded_solution


# Calculates cost and feasibility of the solution:
def pdp_objective_function(pdp, solution):
    cost = 0
    current_weight = 0
    current_node = 0
    visited = [False] * (pdp.size+1)
    for e in solution:
        if e % 2 == 0 and not visited[e-1]:
            return float('inf')  # infeasible due to delivery before pickup
        visited[e] = True
        cost += pdp.dist_matrix[current_node, e]
        current_weight += pdp.capacities[e-1]
        current_node = e
        if current_weight > 1 + EPSILON:
            return float('inf')  # infeasible due to capacity constraint
    return cost


def pdp_check_best_position_change(pdp, solution, i_call):
    len_solution = len(solution)
    i_call_weight = pdp.capacities[int(i_call*2)-2]

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]
    best_ij = None
    best_cost = float('inf')
    for i in range(len_solution+1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1 + EPSILON:
            continue  # if inserting i_call here already makes the vehicle overloaded

        for j in range(i+1, len_solution+2):

            # SIZE CHECK:
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1 + EPSILON:
                break  # Go to next i

            if i + 1 == j:
                if i == 0 and j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                elif j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A1 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len_solution + 1:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j-2], solution[j-1]]
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j-1]]
                elif j == len_solution + 1:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j-2], solution[j-1]]
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j-1]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            # CHECK IF NEW BEST POSITION:
            if add_cost < best_cost:
                best_ij = (i, j)
                best_cost = add_cost

    return best_ij, best_cost


def pdp_check_first_position_change(pdp, solution, i_call):
    len_solution = len(solution)
    if len_solution == 0:
        return (0, 1), None
    i_call_weight = pdp.capacities[int(i_call*2)-2]

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]

    first = list(range(len_solution+1))
    shuffle(first)
    potentials = None
    for i in first:
        if potentials:
            return sample(potentials, 1)[0], None
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1 + EPSILON:  # SIZE CHECK:
            continue
        potentials = []
        for j in range(i+1, len_solution+2):
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1 + EPSILON:  # SIZE CHECK:
                break
            potentials.append((i, j))


def pdp_check_best_tour_spot(pdp, solution, temp_sol):
    len_solution = len(solution)
    a, b = temp_sol[0], temp_sol[-1]
    if len_solution == 0:
        return 0

    temp_sol_cumulative_weight = [0] + [None] * len(temp_sol)
    for i, e in enumerate(temp_sol, 1):
        temp_sol_cumulative_weight[i] = temp_sol_cumulative_weight[i - 1] + pdp.capacities[e - 1]
    max_weight = max(temp_sol_cumulative_weight)

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len_solution
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]

    best_ind = None
    best_cost = float('inf')

    for i in range(len_solution+1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + max_weight > 1 + EPSILON:
            continue  # if inserting i_call here already makes the vehicle overloaded

        if i == 0:
            R1 = pdp.dist_matrix[0, solution[i]]
            A1 = pdp.dist_matrix[0, a]
            A2 = pdp.dist_matrix[b, solution[i]]
        elif i == len_solution:
            R1 = 0
            A1 = pdp.dist_matrix[solution[i-1], a]
            A2 = 0
        else:
            R1 = pdp.dist_matrix[solution[i-1], solution[i]]
            A1 = pdp.dist_matrix[solution[i-1], a]
            A2 = pdp.dist_matrix[b, solution[i]]
        add_cost = -R1 + A1 + A2

        # CHECK IF NEW BEST POSITION:
        if add_cost < best_cost:
            best_ind = i
            best_cost = add_cost

    return best_ind


def pdp_beam_check_best_position_change(pdp, solution, i_call, search_width):
    len_solution = len(solution)
    i_call_weight = pdp.capacities[int(i_call * 2) - 2]

    if len_solution > 0:
        cumulative_weight = [0] + [None] * len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i - 1] + pdp.capacities[e - 1]

    i_j_cost_list = []

    for i in range(len_solution + 1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1 + EPSILON:
            continue  # if inserting i_call here already makes the vehicle overloaded

        for j in range(i + 1, len_solution + 2):

            # SIZE CHECK:
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1 + EPSILON:
                break  # Go to next i

            if i + 1 == j:
                if i == 0 and j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                elif j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i - 1], solution[i]]
                    A1 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len_solution + 1:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j - 2], solution[j - 1]]
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j - 1]]
                elif j == len_solution + 1:
                    R1 = pdp.dist_matrix[solution[i - 1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i - 1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j - 2], solution[j - 1]]
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j - 1]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            i_j_cost_list.append((i, j, add_cost))

    return sorted(i_j_cost_list, key=lambda x: x[2])[:search_width]


################CVRP##########

def cvrp_calculate_tour_deviation(cvrp, solution):
    tour_deviation = [(None, 0)] * (cvrp.instance.n_demands+1)
    for i, vehicle in enumerate(solution):  # vehicles
        prev_node = 0
        for j, d in enumerate(vehicle):  # considered demand
            next_node = vehicle[j+1] if j < len(vehicle)-1 else 0
            R1 = cvrp.instance.dist_matrix[prev_node, d]
            R2 = cvrp.instance.dist_matrix[d, next_node]
            A1 = cvrp.instance.dist_matrix[prev_node, next_node]
            tour_deviation[d] = (d, (R1 + R2 - A1))
            prev_node = d
    return sorted(tour_deviation, key=lambda x: (-x[1], random.random()))


def cvrp_objective_function(cvrp, solution, prev_costs=False, changed=None):
    if changed is None:
        changed = {v for v in range(cvrp.instance.n_vehicles)}
    for v in changed:
        cvrp.costs[v] = 0
        max_weight = 0
        cost = 0
        prev_node = 0
        for i, d in enumerate(solution[v]):
            max_weight += cvrp.instance.demand[d]
            if max_weight > cvrp.instance.max_capacity:
                cvrp.costs[v] = float('inf')
            cost += cvrp.instance.dist_matrix[prev_node, d]
            prev_node = d
            cvrp.d_loc[d] = v
        cost += cvrp.instance.dist_matrix[prev_node, 0]  # back to depot
        cvrp.costs[v] += cost  # += because it could be float('inf') or 0 already.
        cvrp.max_weights[v] = max_weight
    if changed:
        cvrp.sum_costs = sum(cvrp.costs)
    if not prev_costs:
        return cvrp.sum_costs
    return cvrp.sum_costs, cvrp.costs


def cvrp_check_best_position_change(cvrp, solution, i_demand, changed_v):
    vehicle = solution[changed_v]
    if cvrp.max_weights[changed_v] + cvrp.instance.demand[i_demand] > cvrp.instance.max_capacity:
        return None, float('inf')
    best_i = None
    best_cost = float('inf')
    prev_node = 0
    for i in range(len(vehicle)+1):
        next_node = vehicle[i] if i < len(vehicle) else 0
        A1 = cvrp.instance.dist_matrix[prev_node, i_demand]
        A2 = cvrp.instance.dist_matrix[i_demand, next_node]
        R1 = cvrp.instance.dist_matrix[prev_node, next_node]
        add_cost = (A1 + A2 - R1)
        prev_node = next_node
        if add_cost < best_cost:
            best_i = i
            best_cost = add_cost
    return best_i, best_cost


def cvrp_check_first_position_change(cvrp, solution, i_demand, changed_v):
    vehicle = solution[changed_v]
    if cvrp.max_weights[changed_v] + cvrp.instance.demand[i_demand] > cvrp.instance.max_capacity:
        return None, float('inf')
    return random.randint(0, len(vehicle)), None


def cvrp_beam_check_best_position_change(cvrp, solution, i_demand, changed_v, search_width=10):
    vehicle = solution[changed_v]
    if cvrp.max_weights[changed_v] + cvrp.instance.demand[i_demand] > cvrp.instance.max_capacity:
        return [(changed_v, None, float('inf'))]
    i_cost_list = []
    prev_node = 0
    for i in range(len(vehicle)+1):
        next_node = vehicle[i] if i < len(vehicle) else 0
        A1 = cvrp.instance.dist_matrix[prev_node, i_demand]
        A2 = cvrp.instance.dist_matrix[i_demand, next_node]
        R1 = cvrp.instance.dist_matrix[prev_node, next_node]
        add_cost = (A1 + A2 - R1)
        prev_node = next_node
        i_cost_list.append((changed_v, i, add_cost))
    return sorted(i_cost_list, key=lambda x: x[2])[:search_width]

#############PDPTW######################



def pdptw_objective_function(pdptw, solution, prev_costs=False, changed=None):

    if changed is None:
        changed = {v for v in range(pdptw.n_vehicles+1)}

    if pdptw.n_vehicles in changed:
        pdptw.costs[pdptw.n_vehicles] = sum(pdptw.calls[v_call][3] for v_call in solution[pdptw.n_vehicles]) // 2  #fixed += to = now

    for v in changed:
        v_solution = solution[v]
        for v_call in v_solution:
            pdptw.call_loc[v_call] = v  # updating call_loc

    for v in changed - {pdptw.n_vehicles}:

        # Reset cost for the changed vehicle
        pdptw.costs[v] = 0

        # Useful variables
        v_solution = solution[v]
        v_dist_matrix = pdptw.dist_matrix[v]
        v_wait_times = pdptw.wait_times[v]
        v_toll_costs = pdptw.toll_costs[v]
        v_current_node, v_current_time, v_capacity = pdptw.vehicles[v]
        v_current_weight = 0
        v_history = set()

        # pre-calculate cum_weights, cum_leave_times and node_types, also verify size and time constraints for V
        v_cum_weights = [v_current_weight]
        v_cum_leave_times = [v_current_time]
        v_node_types = []
        for v_call in v_solution:

            if v_call not in pdptw.vehicles_compatibility[v]:  # Compatibility check
                pdptw.costs[v] = float('inf')

            if v_call in v_history:
                node_type = 1  # DELIVERY NODE
                v_current_weight -= pdptw.calls[v_call][2]
                _, v_next_node, _, _, _, _, vc_lb, vc_ub = pdptw.calls[v_call]
            else:
                node_type = 0  # PICKUP NODE
                v_history.add(v_call)
                v_current_weight += pdptw.calls[v_call][2]
                v_next_node, _, _, _, vc_lb, vc_ub, _, _ = pdptw.calls[v_call]
                if v_current_weight > v_capacity:
                    pdptw.costs[v] = float('inf')
            v_current_time = max(v_current_time + v_dist_matrix[v_current_node][v_next_node], vc_lb)
            if not vc_lb <= v_current_time <= vc_ub:
                pdptw.costs[v] = float('inf')
            v_current_time += v_wait_times[v_call][node_type]
            pdptw.costs[v] += pdptw.cost_matrix[v][v_current_node][v_next_node]
            pdptw.costs[v] += v_toll_costs[v_call][node_type]
            v_current_node = v_next_node
            v_cum_leave_times.append(v_current_time)
            v_cum_weights.append(v_current_weight)
            v_node_types.append(node_type)

        # pre-calculate lpat for V
        v_node_types_reversed = v_node_types[::-1]
        lpat = []
        prev_lpat = float('inf')
        v_prev_node = 1  # doesn't matter because it will be changed after first iteration and doesn't impact the first one
        for i, v_call in enumerate(v_solution[::-1]):
            node_type = v_node_types_reversed[i]
            if node_type == 0:  # PICKUP NODE
                v_current_node, _, _, _, vc_lb, vc_ub, _, _ = pdptw.calls[v_call]
            else:  # DELIVERY NODE
                _, v_current_node, _, _, _, _, vc_lb, vc_ub = pdptw.calls[v_call]
            current_lpat = min(vc_ub, prev_lpat - v_dist_matrix[v_current_node][v_prev_node] - v_wait_times[v_call][node_type])
            lpat.append(current_lpat)
            prev_lpat = current_lpat
            v_prev_node = v_current_node
        lpat = lpat[::-1]

        # UPDATE VARIBLES FOR PDPTW for vehicle V:
        pdptw.cum_weights[v] = v_cum_weights
        pdptw.cum_leave_times[v] = v_cum_leave_times
        pdptw.node_types[v] = v_node_types
        pdptw.lpat[v] = lpat

    if changed:
        pdptw.sum_costs = sum(pdptw.costs)
    if not prev_costs:
        return pdptw.sum_costs
    return pdptw.sum_costs, pdptw.costs


def pdptw_objective_function_rearange(pdptw, v_pdptw, v_solution, v):
    # reset cost
    v_pdptw.cost = 0

    # Useful variables
    v_dist_matrix = pdptw.dist_matrix[v]
    v_wait_times = pdptw.wait_times[v]
    v_toll_costs = pdptw.toll_costs[v]
    v_current_node, v_current_time, v_capacity = pdptw.vehicles[v]
    v_current_weight = 0
    v_history = set()

    # pre-calculate cum_weights, cum_leave_times and node_types, also verify size and time constraints for V
    v_cum_weights = [v_current_weight]
    v_cum_leave_times = [v_current_time]
    v_node_types = []
    for v_call in v_solution:

        # if v_call not in pdptw.vehicles_compatibility[v]:  # Compatibility check
        #     v_pdptw.cost = float('inf')

        if v_call in v_history:
            node_type = 1  # DELIVERY NODE
            v_current_weight -= pdptw.calls[v_call][2]
            _, v_next_node, _, _, _, _, vc_lb, vc_ub = pdptw.calls[v_call]
        else:
            node_type = 0  # PICKUP NODE
            v_history.add(v_call)
            v_current_weight += pdptw.calls[v_call][2]
            v_next_node, _, _, _, vc_lb, vc_ub, _, _ = pdptw.calls[v_call]
            if v_current_weight > v_capacity:
                v_pdptw.cost = float('inf')
        v_current_time = max(v_current_time + v_dist_matrix[v_current_node][v_next_node], vc_lb)
        if not vc_lb <= v_current_time <= vc_ub:
            v_pdptw.cost = float('inf')
        v_current_time += v_wait_times[v_call][node_type]
        v_pdptw.cost += pdptw.cost_matrix[v][v_current_node][v_next_node]
        v_pdptw.cost += v_toll_costs[v_call][node_type]
        v_current_node = v_next_node
        v_cum_leave_times.append(v_current_time)
        v_cum_weights.append(v_current_weight)
        v_node_types.append(node_type)

    # pre-calculate lpat for V
    v_node_types_reversed = v_node_types[::-1]
    lpat = []
    prev_lpat = float('inf')
    v_prev_node = 1  # doesn't matter because it will be changed after first iteration and doesn't impact the first one
    for i, v_call in enumerate(v_solution[::-1]):
        node_type = v_node_types_reversed[i]
        if node_type == 0:  # PICKUP NODE
            v_current_node, _, _, _, vc_lb, vc_ub, _, _ = pdptw.calls[v_call]
        else:  # DELIVERY NODE
            _, v_current_node, _, _, _, _, vc_lb, vc_ub = pdptw.calls[v_call]
        current_lpat = min(vc_ub, prev_lpat - v_dist_matrix[v_current_node][v_prev_node] - v_wait_times[v_call][node_type])
        lpat.append(current_lpat)
        prev_lpat = current_lpat
        v_prev_node = v_current_node
    lpat = lpat[::-1]

    # UPDATE VARIBLES FOR V_PDPTW:
    v_pdptw.cum_weights = v_cum_weights
    v_pdptw.cum_leave_times = v_cum_leave_times
    v_pdptw.node_types = v_node_types
    v_pdptw.lpat = lpat
    return v_pdptw.cost


def pdptw_calculate_call_difficulty(pdptw):
    compatibilities = []
    sizes = []
    time_window_gaps = []
    time_window_min = []
    time_window_max = []
    distances = []
    wait_times = []
    difficulty_features = (compatibilities, sizes, time_window_gaps, time_window_min, time_window_max, distances, wait_times)
    difficulty_quartiles = defaultdict(list)
    call_difficulty = []  # high --> easy, low --> difficult

    for i_call in range(pdptw.n_calls):
        pickup_node, delivery_node = pdptw.calls[i_call][0], pdptw.calls[i_call][1]
        compatibilities.append(len(pdptw.calls_compatibility[i_call]))  # pos = good
        sizes.append(pdptw.calls[i_call][2])  # pos = bad
        time_window_gaps.append((pdptw.calls[i_call][5] - pdptw.calls[i_call][4]) + (pdptw.calls[i_call][7] - pdptw.calls[i_call][6]))  # pos = good
        time_window_min.append(pdptw.calls[i_call][6] - pdptw.calls[i_call][4])  # pos = bad
        time_window_max.append(pdptw.calls[i_call][7] - pdptw.calls[i_call][4])  # pos = good
        distances.append(sum(pdptw.dist_matrix[v][pickup_node][delivery_node] for v in pdptw.calls_compatibility[i_call]-{pdptw.n_vehicles}) / (len(pdptw.calls_compatibility[i_call])-1))  # pos = bad (needs average over V)
        wait_times.append(sum(pdptw.wait_times[v][i_call][0] + pdptw.wait_times[v][i_call][1] for v in pdptw.calls_compatibility[i_call]-{pdptw.n_vehicles}) / (len(pdptw.calls_compatibility[i_call])-1))  # pos = bad (needs average over V)

    for name, feat in zip(("compatibility", "size", "tw_gap", "tw_min", "tw_max", "distance", "wait_times"), difficulty_features):
        for p in (.25, .50, .75):  # Q1, Median, Q3
            difficulty_quartiles[name].append(np.quantile(feat, p))

    for (compatibility, size, tw_gap, tw_min, tw_max, distance, wait_time) in zip(*difficulty_features):
        a = bisect(difficulty_quartiles["compatibility"], compatibility)
        b = bisect(difficulty_quartiles["size"], size)
        c = bisect(difficulty_quartiles["tw_gap"], tw_gap)
        d = bisect(difficulty_quartiles["tw_min"], tw_min)
        e = bisect(difficulty_quartiles["tw_max"], tw_max)
        f = bisect(difficulty_quartiles["distance"], distance)
        g = bisect(difficulty_quartiles["wait_times"], wait_time)
        total = (a - b + c - d + e - f - g)
        call_difficulty.append(total)
    return call_difficulty

def pdptw_calculate_tour_deviation(pdptw, solution):
    tour_deviation = [0] * pdptw.n_calls

    for v in range(pdptw.n_vehicles):  # all vehicles except dummy
        v_solution = solution[v]
        v_cost_matrix = pdptw.cost_matrix[v]
        i = 0
        while i < len(v_solution):
            v_call = v_solution[i]
            v_node = pdptw.calls[v_call][pdptw.node_types[v][i]]

            if i == len(v_solution) - 1:  # single delivery call at the end
                v_prev_node = pdptw.calls[v_solution[i - 1]][pdptw.node_types[v][i - 1]]
                R1 = v_cost_matrix[v_prev_node][v_node]
                tour_deviation[v_call] += R1
                break

            v_next_node = pdptw.calls[v_solution[i + 1]][pdptw.node_types[v][i + 1]]  # we not v_call is not last
            if i == 0:  # first
                v_prev_node = pdptw.vehicles[v][0]
            else:  # not first
                v_prev_node = pdptw.calls[v_solution[i - 1]][pdptw.node_types[v][i - 1]]

            R1 = v_cost_matrix[v_prev_node][v_node]  # prev to node
            R2 = v_cost_matrix[v_node][v_next_node]  # node to next
            R3 = 0
            A1 = 0

            if v_call == v_solution[i + 1]:  # neighbor call
                if (i + 1) == len(v_solution) - 1:  # (i+1) is the last delivery node in v_solution
                    pass
                else:  # continued end
                    v_next_next_node = pdptw.calls[v_solution[i + 2]][pdptw.node_types[v][i + 2]]
                    R3 = v_cost_matrix[v_next_node][v_next_next_node]  # next to next_next
                    A1 = v_cost_matrix[v_prev_node][v_next_next_node]  # prev to next_next
                i += 2
            else:  # not neighbors
                A1 = v_cost_matrix[v_prev_node][v_next_node]  # prev to next
                i += 1

            tour_deviation[v_call] += (R1 + R2 + R3 - A1)

    for v_call in set(solution[pdptw.n_vehicles]):  # dummy vehicle
        tour_deviation[v_call] += pdptw.calls[v_call][3]

    tour_deviation = list(zip(range(pdptw.n_calls), tour_deviation))
    tour_deviation = sorted(tour_deviation, key=lambda x: -x[1])
    return tour_deviation


def pdptw_check_best_position_change(pdptw, solution, i_call, changed_v):

    if changed_v == pdptw.n_vehicles:
        return (0, 1), pdptw.calls[i_call][3]  # Inserting into dummy is simple

    v_solution = solution[changed_v]
    v_dist_matrix = pdptw.dist_matrix[changed_v]
    v_cost_matrix = pdptw.cost_matrix[changed_v]
    v_wait_times = pdptw.wait_times[changed_v]
    v_toll_costs = pdptw.toll_costs[changed_v]

    v_lpat = pdptw.lpat[changed_v]
    v_cum_weights = pdptw.cum_weights[changed_v]
    v_cum_leave_times = pdptw.cum_leave_times[changed_v]
    v_node_types = pdptw.node_types[changed_v]

    v_start_node, _, v_capacity = pdptw.vehicles[changed_v]
    ii_call = pdptw.calls[i_call]

    best_ij = None
    best_cost = float('inf')
    for i in range(len(v_solution) + 1):

        # SIZE CHECK:
        if v_cum_weights[i] + ii_call[2] > v_capacity:
            continue  # if inserting i_call here already makes the vehicle overloaded

        if i == 0:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[v_start_node][ii_call[0]]
        else:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]

        if current_arrival_time > ii_call[5]:
            continue  # if i_call was not able to be inserted here due to time window violation
        current_arrival_time = max(current_arrival_time, ii_call[4])

        prev_leave_time = current_arrival_time + v_wait_times[i_call][0]

        # LPAT CHECK:
        if i != len(v_solution) and prev_leave_time + v_dist_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]] > v_lpat[i]:
            continue  # if inserting i_call here caused the next call in the order to violate its lpat

        for j in range(i + 1, len(v_solution) + 2):

            # COMPUTE INSERTION COST:
            if i+1 == j:
                if i == 0 and j == len(v_solution)+1:
                    R1 = 0
                    A1 = v_cost_matrix[v_start_node][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = 0
                elif i == 0:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A1 = v_cost_matrix[v_start_node][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                elif j == len(v_solution)+1:
                    R1 = 0
                    A1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = 0
                else:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len(v_solution)+1:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[v_start_node][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = 0
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = 0
                elif i == 0:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[v_start_node][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                elif j == len(v_solution) + 1:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]]  [ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = 0
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = 0
                else:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            # CHECK COMPATIBILITY:
            if j == len(v_solution) + 1:
                lpat_check = False
            else:
                lpat_check = True
                next_drive_time = v_dist_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]

            if i+1 == j:
                prev_prev_call = i_call
                prev_prev_node = pdptw.calls[prev_prev_call][0]
                # SIZE CHECK IS NOT NEEDED:

                # DONT UPDATE LEAVE TIME
                current_arrival_time = prev_leave_time + v_dist_matrix[ii_call[0]][ii_call[1]]
                current_arrival_time = max(current_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if current_arrival_time > ii_call[7]:
                    continue

                # LPAT CHECK:
                if lpat_check and current_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            else:
                # SIZE CHECK:
                if v_cum_weights[j-1] + ii_call[2] > v_capacity:
                    break  # Go to next i
                prev_call = v_solution[j - 2]
                prev_node = pdptw.calls[prev_call][v_node_types[j-2]]

                # UPDATE TIME FROM PREV_PREV --> PREV
                current_arrival_time = prev_leave_time + v_dist_matrix[prev_prev_node][prev_node]
                current_arrival_time = max(current_arrival_time, pdptw.calls[prev_call][v_node_types[j-2]*2+4])
                prev_leave_time = current_arrival_time + v_wait_times[prev_call][v_node_types[j-2]]
                prev_prev_node = prev_node

                # CHECK IF INSERTION IS OKAY BASED ON PREV_CALLS's TIME WINDOWS
                if current_arrival_time > pdptw.calls[prev_call][v_node_types[j-2]*2+5]:
                    break  # Go to next i

                i_call_dest_arrival_time = prev_leave_time + v_dist_matrix[prev_node][ii_call[1]]
                i_call_dest_arrival_time = max(i_call_dest_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if i_call_dest_arrival_time > ii_call[7]:
                    break  # Go to next i

                # LPAT CHECK:
                if lpat_check and i_call_dest_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            # CHECK IF NEW BEST POSITION:
            if add_cost < best_cost:
                best_ij = (i, j)
                best_cost = add_cost

    A13 = v_toll_costs[i_call][0]  # delivery cost for i_call origin
    A23 = v_toll_costs[i_call][1]  # delivery cost for i_call destination
    return best_ij, best_cost+A13+A23


def pdptw_check_first_position_change(pdptw, solution, i_call, changed_v):

    if changed_v == pdptw.n_vehicles:
        return (0, 1), None  # Inserting into dummy is simple

    v_solution = solution[changed_v]
    v_dist_matrix = pdptw.dist_matrix[changed_v]
    v_wait_times = pdptw.wait_times[changed_v]

    v_lpat = pdptw.lpat[changed_v]
    v_cum_weights = pdptw.cum_weights[changed_v]
    v_cum_leave_times = pdptw.cum_leave_times[changed_v]
    v_node_types = pdptw.node_types[changed_v]

    v_start_node, _, v_capacity = pdptw.vehicles[changed_v]
    ii_call = pdptw.calls[i_call]

    for i in range(len(v_solution) + 1):

        # SIZE CHECK:
        if v_cum_weights[i] + ii_call[2] > v_capacity:
            continue  # if inserting i_call here already makes the vehicle overloaded

        if i == 0:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[v_start_node][ii_call[0]]
        else:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]

        if current_arrival_time > ii_call[5]:
            continue  # if i_call was not able to be inserted here due to time window violation
        current_arrival_time = max(current_arrival_time, ii_call[4])

        prev_leave_time = current_arrival_time + v_wait_times[i_call][0]

        # LPAT CHECK:
        if i != len(v_solution) and prev_leave_time + v_dist_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]] > v_lpat[i]:
            continue  # if inserting i_call here caused the next call in the order to violate its lpat

        for j in range(i + 1, len(v_solution) + 2):

            # CHECK COMPATIBILITY:
            if j == len(v_solution) + 1:
                lpat_check = False
            else:
                lpat_check = True
                next_drive_time = v_dist_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]

            if i+1 == j:
                prev_prev_call = i_call
                prev_prev_node = pdptw.calls[prev_prev_call][0]
                # SIZE CHECK IS NOT NEEDED:

                # DONT UPDATE LEAVE TIME
                current_arrival_time = prev_leave_time + v_dist_matrix[ii_call[0]][ii_call[1]]
                current_arrival_time = max(current_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if current_arrival_time > ii_call[7]:
                    continue

                # LPAT CHECK:
                if lpat_check and current_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            else:
                # SIZE CHECK:
                if v_cum_weights[j-1] + ii_call[2] > v_capacity:
                    break  # Go to next i
                prev_call = v_solution[j - 2]
                prev_node = pdptw.calls[prev_call][v_node_types[j-2]]

                # UPDATE TIME FROM PREV_PREV --> PREV
                current_arrival_time = prev_leave_time + v_dist_matrix[prev_prev_node][prev_node]
                current_arrival_time = max(current_arrival_time, pdptw.calls[prev_call][v_node_types[j-2]*2+4])
                prev_leave_time = current_arrival_time + v_wait_times[prev_call][v_node_types[j-2]]
                prev_prev_node = prev_node

                # CHECK IF INSERTION IS OKAY BASED ON PREV_CALLS's TIME WINDOWS
                if current_arrival_time > pdptw.calls[prev_call][v_node_types[j-2]*2+5]:
                    break  # Go to next i

                i_call_dest_arrival_time = prev_leave_time + v_dist_matrix[prev_node][ii_call[1]]
                i_call_dest_arrival_time = max(i_call_dest_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if i_call_dest_arrival_time > ii_call[7]:
                    break  # Go to next i

                # LPAT CHECK:
                if lpat_check and i_call_dest_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            # Getting to this point means that (i, j) is a feasible position, so we return (i, j) immediately
            return (i, j), None
    return None, None




def pdptw_beam_check_best_position_change(pdptw, solution, i_call, changed_v):

    if changed_v == pdptw.n_vehicles:
        return [(changed_v, 0, 1, pdptw.calls[i_call][3])]  # Inserting into dummy is simple

    v_solution = solution[changed_v]
    v_dist_matrix = pdptw.dist_matrix[changed_v]
    v_cost_matrix = pdptw.cost_matrix[changed_v]
    v_wait_times = pdptw.wait_times[changed_v]
    v_toll_costs = pdptw.toll_costs[changed_v]

    v_lpat = pdptw.lpat[changed_v]
    v_cum_weights = pdptw.cum_weights[changed_v]
    v_cum_leave_times = pdptw.cum_leave_times[changed_v]
    v_node_types = pdptw.node_types[changed_v]

    v_start_node, _, v_capacity = pdptw.vehicles[changed_v]
    ii_call = pdptw.calls[i_call]

    i_j_cost_list = []
    A13 = v_toll_costs[i_call][0]  # delivery cost for i_call origin
    A23 = v_toll_costs[i_call][1]  # delivery cost for i_call destination
    #best_ij = None
    #best_cost = float('inf')
    for i in range(len(v_solution) + 1):

        # SIZE CHECK:
        if v_cum_weights[i] + ii_call[2] > v_capacity:
            continue  # if inserting i_call here already makes the vehicle overloaded

        if i == 0:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[v_start_node][ii_call[0]]
        else:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]

        if current_arrival_time > ii_call[5]:
            continue  # if i_call was not able to be inserted here due to time window violation
        current_arrival_time = max(current_arrival_time, ii_call[4])

        prev_leave_time = current_arrival_time + v_wait_times[i_call][0]

        # LPAT CHECK:
        if i != len(v_solution) and prev_leave_time + v_dist_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]] > v_lpat[i]:
            continue  # if inserting i_call here caused the next call in the order to violate its lpat

        for j in range(i + 1, len(v_solution) + 2):

            # COMPUTE INSERTION COST:
            if i+1 == j:
                if i == 0 and j == len(v_solution)+1:
                    R1 = 0
                    A1 = v_cost_matrix[v_start_node][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = 0
                elif i == 0:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A1 = v_cost_matrix[v_start_node][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                elif j == len(v_solution)+1:
                    R1 = 0
                    A1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = 0
                else:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len(v_solution)+1:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[v_start_node][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = 0
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = 0
                elif i == 0:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[v_start_node][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                elif j == len(v_solution) + 1:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]]  [ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = 0
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = 0
                else:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            # CHECK COMPATIBILITY:
            if j == len(v_solution) + 1:
                lpat_check = False
            else:
                lpat_check = True
                next_drive_time = v_dist_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]

            if i+1 == j:
                prev_prev_call = i_call
                prev_prev_node = pdptw.calls[prev_prev_call][0]
                # SIZE CHECK IS NOT NEEDED:

                # DONT UPDATE LEAVE TIME
                current_arrival_time = prev_leave_time + v_dist_matrix[ii_call[0]][ii_call[1]]
                current_arrival_time = max(current_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if current_arrival_time > ii_call[7]:
                    continue

                # LPAT CHECK:
                if lpat_check and current_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            else:
                # SIZE CHECK:
                if v_cum_weights[j-1] + ii_call[2] > v_capacity:
                    break  # Go to next i
                prev_call = v_solution[j - 2]
                prev_node = pdptw.calls[prev_call][v_node_types[j-2]]

                # UPDATE TIME FROM PREV_PREV --> PREV
                current_arrival_time = prev_leave_time + v_dist_matrix[prev_prev_node][prev_node]
                current_arrival_time = max(current_arrival_time, pdptw.calls[prev_call][v_node_types[j-2]*2+4])
                prev_leave_time = current_arrival_time + v_wait_times[prev_call][v_node_types[j-2]]
                prev_prev_node = prev_node

                # CHECK IF INSERTION IS OKAY BASED ON PREV_CALLS's TIME WINDOWS
                if current_arrival_time > pdptw.calls[prev_call][v_node_types[j-2]*2+5]:
                    break  # Go to next i

                i_call_dest_arrival_time = prev_leave_time + v_dist_matrix[prev_node][ii_call[1]]
                i_call_dest_arrival_time = max(i_call_dest_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if i_call_dest_arrival_time > ii_call[7]:
                    break  # Go to next i

                # LPAT CHECK:
                if lpat_check and i_call_dest_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            i_j_cost_list.append((changed_v, i, j, add_cost+A13+A23))

    return sorted(i_j_cost_list, key=lambda x: x[3])



def pdptw_beam_check_best_position_change_rearange(pdptw, v_pdptw, v_solution, i_call, changed_v):
    # useful fixed
    v_dist_matrix = pdptw.dist_matrix[changed_v]
    v_cost_matrix = pdptw.cost_matrix[changed_v]
    v_wait_times = pdptw.wait_times[changed_v]
    v_toll_costs = pdptw.toll_costs[changed_v]

    # useful variables
    v_lpat = v_pdptw.lpat
    v_cum_weights = v_pdptw.cum_weights
    v_cum_leave_times = v_pdptw.cum_leave_times
    v_node_types = v_pdptw.node_types

    v_start_node, _, v_capacity = pdptw.vehicles[changed_v]
    ii_call = pdptw.calls[i_call]

    i_j_cost_list = []
    A13 = v_toll_costs[i_call][0]  # delivery cost for i_call origin
    A23 = v_toll_costs[i_call][1]  # delivery cost for i_call destination
    #best_ij = None
    #best_cost = float('inf')
    for i in range(len(v_solution) + 1):

        # SIZE CHECK:
        if v_cum_weights[i] + ii_call[2] > v_capacity:
            continue  # if inserting i_call here already makes the vehicle overloaded

        if i == 0:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[v_start_node][ii_call[0]]
        else:
            current_arrival_time = v_cum_leave_times[i] + v_dist_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]

        if current_arrival_time > ii_call[5]:
            continue  # if i_call was not able to be inserted here due to time window violation
        current_arrival_time = max(current_arrival_time, ii_call[4])

        prev_leave_time = current_arrival_time + v_wait_times[i_call][0]

        # LPAT CHECK:
        if i != len(v_solution) and prev_leave_time + v_dist_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]] > v_lpat[i]:
            continue  # if inserting i_call here caused the next call in the order to violate its lpat

        for j in range(i + 1, len(v_solution) + 2):

            # COMPUTE INSERTION COST:
            if i+1 == j:
                if i == 0 and j == len(v_solution)+1:
                    R1 = 0
                    A1 = v_cost_matrix[v_start_node][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = 0
                elif i == 0:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A1 = v_cost_matrix[v_start_node][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                elif j == len(v_solution)+1:
                    R1 = 0
                    A1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = 0
                else:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A2 = v_cost_matrix[ii_call[0]][ii_call[1]]
                    A3 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len(v_solution)+1:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[v_start_node][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = 0
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = 0
                elif i == 0:
                    R1 = v_cost_matrix[v_start_node][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[v_start_node][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                elif j == len(v_solution) + 1:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]]  [ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = 0
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = 0
                else:
                    R1 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    A11 = v_cost_matrix[pdptw.calls[v_solution[i-1]][v_node_types[i-1]]][ii_call[0]]
                    A12 = v_cost_matrix[ii_call[0]][pdptw.calls[v_solution[i]][v_node_types[i]]]
                    R2 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                    A21 = v_cost_matrix[pdptw.calls[v_solution[j-2]][v_node_types[j-2]]][ii_call[1]]
                    A22 = v_cost_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            # CHECK COMPATIBILITY:
            if j == len(v_solution) + 1:
                lpat_check = False
            else:
                lpat_check = True
                next_drive_time = v_dist_matrix[ii_call[1]][pdptw.calls[v_solution[j-1]][v_node_types[j-1]]]

            if i+1 == j:
                prev_prev_call = i_call
                prev_prev_node = pdptw.calls[prev_prev_call][0]
                # SIZE CHECK IS NOT NEEDED:

                # DONT UPDATE LEAVE TIME
                current_arrival_time = prev_leave_time + v_dist_matrix[ii_call[0]][ii_call[1]]
                current_arrival_time = max(current_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if current_arrival_time > ii_call[7]:
                    continue

                # LPAT CHECK:
                if lpat_check and current_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            else:
                # SIZE CHECK:
                if v_cum_weights[j-1] + ii_call[2] > v_capacity:
                    break  # Go to next i
                prev_call = v_solution[j - 2]
                prev_node = pdptw.calls[prev_call][v_node_types[j-2]]

                # UPDATE TIME FROM PREV_PREV --> PREV
                current_arrival_time = prev_leave_time + v_dist_matrix[prev_prev_node][prev_node]
                current_arrival_time = max(current_arrival_time, pdptw.calls[prev_call][v_node_types[j-2]*2+4])
                prev_leave_time = current_arrival_time + v_wait_times[prev_call][v_node_types[j-2]]
                prev_prev_node = prev_node

                # CHECK IF INSERTION IS OKAY BASED ON PREV_CALLS's TIME WINDOWS
                if current_arrival_time > pdptw.calls[prev_call][v_node_types[j-2]*2+5]:
                    break  # Go to next i

                i_call_dest_arrival_time = prev_leave_time + v_dist_matrix[prev_node][ii_call[1]]
                i_call_dest_arrival_time = max(i_call_dest_arrival_time, ii_call[6])

                # CHECK IF INSERTION IS OKAY BASED ON I_CALL's TIME WINDOWS
                if i_call_dest_arrival_time > ii_call[7]:
                    break  # Go to next i

                # LPAT CHECK:
                if lpat_check and i_call_dest_arrival_time + v_wait_times[i_call][1] + next_drive_time > v_lpat[j-1]:
                    continue

            i_j_cost_list.append((changed_v, i, j, add_cost+A13+A23))

    return sorted(i_j_cost_list, key=lambda x: x[3])



def find_last_model_index(path):
    for root, dirs, files in os.walk(path):
        models = sorted([subs for subs in files if 'checkpoint' in subs])
        return int(models[-1].split('_')[-1].split('.')[0])

