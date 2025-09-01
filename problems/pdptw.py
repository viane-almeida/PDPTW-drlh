import numpy as np
import os
import pickle
from copy import copy, deepcopy
from collections import namedtuple
from problems import Problem
from utils.utils.utils import pdptw_objective_function, distance, pdptw_calculate_call_difficulty
from utils.operators.pdptw import pdptw_remove_longest_tour_deviation_s, pdptw_remove_tour_neighbors,\
    pdptw_remove_xs, pdptw_remove_s, pdptw_remove_m, pdptw_remove_l, pdptw_remove_xl, pdptw_insert_greedy,\
    pdptw_insert_beam_search, pdptw_insert_by_variance, pdptw_insert_first, pdptw_insert_single_best, pdptw_remove_longest_tour_deviation_l, pdptw_remove_xxl, pdptw_remove_least_frequent_s, pdptw_remove_least_frequent_m, pdptw_remove_least_frequent_xl, pdptw_remove_one_vehicle, pdptw_remove_two_vehicles, pdptw_insert_least_loaded_vehicle, pdptw_insert_least_active_vehicle, pdptw_insert_close_vehicle, pdptw_insert_group, pdptw_insert_by_difficulty, pdptw_insert_best_fit, pdptw_rearange_vehicles_fast


class PDPTW(Problem):
    def __init__(self, 
        n_nodes, n_calls, n_vehicles, calls, vehicles, vehicles_compatibility, calls_compatibility, dist_matrix, cost_matrix, wait_times, toll_costs,
        T_f=None, **kwargs):
        super(PDPTW, self).__init__(name='PDPTW', heuristics=self._get_heus(), T_f=T_f, size=n_vehicles*n_calls, **kwargs)

        #
        # constants
        self.instance_info = namedtuple(self.name,
                                        ['n_nodes',
                                         'n_calls',
                                         'n_vehicles',
                                         'calls',
                                         'vehicles',
                                         'vehicles_compatibility',
                                         'calls_compatibility',
                                         'dist_matrix',
                                         'cost_matrix',
                                         'wait_times',
                                         'toll_costs'])
        self.instance = self.instance_info

        # INPUT (FIXED)
        self.n_nodes = n_nodes  # |N|
        self.n_calls = n_calls  # |C|
        self.n_vehicles = n_vehicles  # |V|
        self.calls = calls  # origin node, destination node, size, cost of not transporting, lbtw pickup, ubtw pickup, lbtw delivery, ubtw delivery
        self.vehicles = vehicles  # home_node, starting_node, capacity
        self.vehicles_compatibility = vehicles_compatibility
        self.calls_compatibility = calls_compatibility
        self.dist_matrix = dist_matrix  # |V| * |N| * |N|
        self.cost_matrix = cost_matrix  # |V| * |N| * |N|
        self.wait_times = wait_times  # |V| * |C|
        self.toll_costs = toll_costs  # |V| * |C|
        # EXTRA (FIXED)
        # NB! MOVED TO LOAD_INSTANCE()
        #self.call_difficulty = pdptw_calculate_call_difficulty(self)
        #
        # EXTRA (VARIABLES)
        self.costs = [0]*(self.n_vehicles+1)  # all prev costs are 0. REMANED prev_costs to costs
        self.sum_costs = None
        self.call_loc = [self.n_vehicles]*self.n_calls  # all calls start in dummy vehicle
        self.lpat = [None]*self.n_vehicles
        self.cum_weights = [None]*self.n_vehicles
        self.cum_leave_times = [None] * self.n_vehicles
        self.node_types = [None]*self.n_vehicles
        self.calls_counter = [0]*self.n_calls  # No need to maintain multiple copies of this for any reason
        #
        self.prev_costs = [0]*(self.n_vehicles+1)  # all prev costs are 0
        self.prev_sum_costs = None
        self.prev_call_loc = [self.n_vehicles]*self.n_calls  # all calls start in dummy vehicle
        self.prev_lpat = [None]*self.n_vehicles
        self.prev_cum_weights = [None]*self.n_vehicles
        self.prev_cum_leave_times = [None] * self.n_vehicles
        self.prev_node_types = [None]*self.n_vehicles


    def objective_function(self, solution, changed=None):
        return pdptw_objective_function(self, solution, changed)


    def generate_initial_solution(self):
        return [[] for _ in range(self.n_vehicles)] + [[i//2 for i in range(self.n_calls*2)]]


    # TO DO: FIX THIS!!
    def generate_instance(self, size=None, index_sample=None, T_f=None):  # not implemented
        pass


    def load_instance(self, file_path, T_f=None):

        def int_or_float(x):
            try:
                return int(x)
            except:
                return float(x)

        if T_f is not None:
            self.T_f = T_f
        
        with open(file_path, 'r', encoding='windows-1252') as f:

            # Information about instance generation
            for _ in range(13):
                f.readline()

            # Number of nodes:
            f.readline()
            n_nodes = int(f.readline())

            # Number of vehicles:
            f.readline()
            n_vehicles = int(f.readline())

            # Vehicles:
            f.readline()
            vehicles = [None] * n_vehicles
            for i in range(n_vehicles):
                nums = list(map(int_or_float, f.readline().split(',')))
                nums[1] -= 1  # home node
                vehicles[i] = nums[1:]

            # Number of calls:
            f.readline()
            n_calls = int(f.readline())

            # Compatibility
            f.readline()
            vehicles_compatibility = [None] * n_vehicles
            vehicles_compatibility.append(set(range(n_calls)))  # dummy
            calls_compatibility = [{n_vehicles} for _ in range(
                n_calls)]  # dummy, MISTAKE here was fixed here 20.05.21. Used to be [{n_vehicles}]*n_calls
            for i in range(n_vehicles):
                nums = list(map(int_or_float, f.readline().split(',')))
                nums = [n - 1 for n in nums]
                for n in nums:
                    calls_compatibility[n].add(i)
                vehicles_compatibility[i] = set(nums[1:])

            # Calls:
            f.readline()
            calls = [None] * n_calls
            for i in range(n_calls):
                nums = list(map(int_or_float, f.readline().split(',')))
                nums[1] -= 1
                nums[2] -= 1
                calls[i] = nums[1:]

            # Travel times and and costs:
            f.readline()
            dist_matrix = [[[None for _ in range(n_nodes)] for _ in range(n_nodes)] for _ in
                           range(n_vehicles)]  # |V| * |N| * |N|
            cost_matrix = [[[None for _ in range(n_nodes)] for _ in range(n_nodes)] for _ in
                           range(n_vehicles)]  # |V| * |N| * |N|
            for _ in range(n_vehicles * n_nodes * n_nodes):
                v, origin, destination, dist, cost = map(int_or_float, f.readline().split(','))
                dist_matrix[v - 1][origin - 1][destination - 1] = dist
                cost_matrix[v - 1][origin - 1][destination - 1] = cost

            # Node times and costs:
            f.readline()
            wait_times = [[[None, None] for _ in range(n_calls)] for _ in range(n_vehicles)]  # |V| * |C|
            toll_costs = [[[None, None] for _ in range(n_calls)] for _ in range(n_vehicles)]  # |V| * |C|
            for _ in range(n_vehicles * n_calls):
                v, c, origin_time, origin_cost, destination_time, destination_cost = map(int_or_float, f.readline().split(','))
                wait_times[v - 1][c - 1] = [origin_time, destination_time]
                toll_costs[v - 1][c - 1] = [origin_cost, destination_cost]

            # EOF:
            f.readline()

        return self.instance_info(n_nodes, n_calls, n_vehicles, calls, vehicles, vehicles_compatibility, calls_compatibility,
                   dist_matrix, cost_matrix, wait_times, toll_costs)


    def apply_heus(self, solution, heu, prev_accepted=True):
        self.criteria_accepted(accepted=prev_accepted)  # accepting/rejecting current/prev variable values based on prev_accepted
        if len(heu) > 1:
            solution = deepcopy(solution)
            op, op2 = heu
            solution, removed_objcts = op(self, solution)
            new_solution, new_cost = op2(self, solution, removed_objcts)

        else:
            new_solution, new_cost = heu[0](self, solution)
        return new_solution, new_cost

    def criteria_accepted(self, accepted=None):
        if accepted:
            self.prev_costs = copy(self.costs)
            self.prev_sum_costs = copy(self.sum_costs)
            self.prev_call_loc = deepcopy(self.call_loc)
            self.prev_lpat = deepcopy(self.lpat)
            self.prev_cum_weights = deepcopy(self.cum_weights)
            self.prev_cum_leave_times = deepcopy(self.cum_leave_times)
            self.prev_node_types = deepcopy(self.node_types)
        else:
            self.costs = copy(self.prev_costs)
            self.sum_costs = copy(self.prev_sum_costs)
            self.call_loc = deepcopy(self.prev_call_loc)
            self.lpat = deepcopy(self.prev_lpat)
            self.cum_weights = deepcopy(self.prev_cum_weights)
            self.cum_leave_times = deepcopy(self.prev_cum_leave_times)
            self.node_types = deepcopy(self.prev_node_types)

    def update_from_other(self, other):  # I could do deepcopy here, but other is not going to be used anyway
        self.costs = other.costs
        self.sum_costs = other.sum_costs
        self.call_loc = other.call_loc
        self.lpat = other.lpat
        self.cum_weights = other.cum_weights
        self.cum_leave_times = other.cum_leave_times
        self.node_types = other.node_types

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)  # shallow copy of every attribute

        # deepcopy of the variables:
        new.costs = deepcopy(self.costs, memodict)
        new.sum_costs = deepcopy(self.sum_costs, memodict)
        new.call_loc = deepcopy(self.call_loc, memodict)
        new.lpat = deepcopy(self.lpat, memodict)
        new.cum_weights = deepcopy(self.cum_weights, memodict)
        new.cum_leave_times = deepcopy(self.cum_leave_times, memodict)
        new.node_types = deepcopy(self.node_types, memodict)
        return new

    def deepcopy_rearange(self, v):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)  # shallow copy of every attribute

        # copy of the variables for vehicle v:
        new.costs[v] = copy(self.costs[v])
        new.sum_costs = copy(self.sum_costs)
        new.lpat[v] = copy(self.lpat[v])
        new.cum_weights[v] = copy(self.cum_weights[v])
        new.cum_leave_times[v] = copy(self.cum_leave_times[v])
        new.node_types[v] = copy(self.node_types[v])
        return new

    def update_from_v_pdptw(self, other, v):
        self.costs[v] = other.cost
        self.sum_costs = sum(self.costs)
        self.lpat[v] = other.lpat
        self.cum_weights[v] = other.cum_weights
        self.cum_leave_times[v] = other.cum_leave_times
        self.node_types[v] = other.node_types

    def create_v_pdptw(self, v):
        return V_PDPTW(pdptw=self, v=v)


    def _get_heus(self):
        remove_operators = [pdptw_remove_longest_tour_deviation_s, pdptw_remove_tour_neighbors,
                            pdptw_remove_xs, pdptw_remove_s, pdptw_remove_m, pdptw_remove_l, pdptw_remove_xl]
        insert_operators = [pdptw_insert_greedy, pdptw_insert_beam_search, pdptw_insert_by_variance, pdptw_insert_first]
        additional_operators = [pdptw_insert_single_best]
        len_r_op = len(remove_operators)
        len_i_op = len(insert_operators)
        len_operators = len_r_op * len_i_op
        heuristics = [additional_operators] + \
                    [(remove_operators[i // len_i_op], insert_operators[i % len_i_op]) for i in range(len_operators)]
        heuristic_names = [
             '{:03d}_'.format(i+1) + '_and_'.join([str(n).split()[1] for n in name])
            for i, name in enumerate(heuristics)]

        return dict(zip(heuristic_names, heuristics))


    # TO DO: IT LOOKS LIKE THIS WAS NOT USED IN THE ORIGINAL CODE; EXPERIMENT WITH IT IF THE RESULTS ARE NOT GOOD
    def _get_heus_all(self):
        remove_operators = [pdptw_remove_longest_tour_deviation_s, pdptw_remove_longest_tour_deviation_l,
                            pdptw_remove_tour_neighbors, pdptw_remove_xs, pdptw_remove_s, pdptw_remove_m, pdptw_remove_l, pdptw_remove_xl, pdptw_remove_xxl,
                            pdptw_remove_least_frequent_s, pdptw_remove_least_frequent_m, pdptw_remove_least_frequent_xl, pdptw_remove_one_vehicle,
                            pdptw_remove_two_vehicles]
        insert_operators = [pdptw_insert_first, pdptw_insert_greedy, pdptw_insert_beam_search, pdptw_insert_by_variance,
                            pdptw_insert_least_loaded_vehicle, pdptw_insert_least_active_vehicle, pdptw_insert_close_vehicle, pdptw_insert_group,
                            pdptw_insert_by_difficulty, pdptw_insert_best_fit]  # I removed "insert_dummy" because I am afraid that RL will exploit this operator.
        additional_operators = [pdptw_insert_single_best, pdptw_rearange_vehicles_fast]
        len_r_op = len(remove_operators)
        len_i_op = len(insert_operators)
        len_operators = len_r_op * len_i_op
        heuristics = [additional_operators] + \
                    [(remove_operators[i // len_i_op], insert_operators[i % len_i_op]) for i in range(len_operators)]
        heuristic_names = [
             '{:03d}_'.format(i+1) + '_and_'.join([str(n).split()[1] for n in name])
            for i, name in enumerate(heuristics)]

        return dict(zip(heuristic_names, heuristics))


class V_PDPTW:
    def __init__(self, cost=None, lpat=None, cum_weights=None, cum_leave_times=None, node_types=None, pdptw=None, v=None):
        if pdptw is not None and v is not None:
            self.cost = pdptw.costs[v]
            self.lpat = copy(pdptw.lpat[v])
            self.cum_weights = copy(pdptw.cum_weights[v])
            self.cum_leave_times = copy(pdptw.cum_leave_times[v])
            self.node_types = copy(pdptw.node_types[v])
        else:
            self.cost = cost
            self.lpat = lpat
            self.cum_weights = cum_weights
            self.cum_leave_times = cum_leave_times
            self.node_types = node_types

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        new = cls.__new__(cls)
        new.cost = self.cost
        new.lpat = copy(self.lpat)
        new.cum_weights = copy(self.cum_weights)
        new.cum_leave_times = copy(self.cum_leave_times)
        new.node_types = copy(self.node_types)
        return new
