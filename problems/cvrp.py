import json
import numpy as np
from copy import copy
from copy import deepcopy
from collections import namedtuple
from DRLH.problems import Problem
from DRLH.utils.utils import cvrp_objective_function, distance
from DRLH.utils.operators.cvrp import cvrp_remove_longest_tour_deviation_s, cvrp_remove_tour_neighbors,\
    cvrp_remove_xs, cvrp_remove_s, cvrp_remove_m, cvrp_remove_l, cvrp_remove_xl, cvrp_insert_greedy,\
    cvrp_insert_beam_search, cvrp_insert_by_variance, cvrp_insert_first, cvrp_insert_single_best, cvrp_remove_nothing


class CVRP(Problem):
    def __init__(self, size,  T_f=None, **kwargs):
        super(CVRP, self).__init__(name='CVRP', heuristics=self._get_heus(), T_f=T_f, size=size, **kwargs)

        #
        # constants
        self.instance_info = namedtuple(self.name,
                              ['n_vehicles',
                              'n_demands',
                               'locations',
                               'demand',
                               'max_capacity',
                               'dist_matrix'])
        self.instance = self.instance_info

        # variables
        self.costs = [None] * size
        self.sum_costs = [None] * size
        self.max_weights = [None] * size
        self.d_loc = [None] * (size + 1)
        #
        self.prev_costs = [None] * size
        self.prev_sum_costs = [None] * size
        self.prev_max_weights = [None] * size
        self.prev_d_loc = [None] * (size + 1)


    def objective_function(self, solution, changed=None):
        return cvrp_objective_function(self, solution, changed)



    def generate_initial_solution(self):
        return [[i] for i in range(1, self.instance.n_vehicles+1)]

    def generate_instance(self, size=None, index_sample=None, T_f=None):
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.,
            200: 50.,
            500: 50.,
        }
        locations = np.random.uniform(size=(size + 1, 2))  # Node locations. 0 is depot
        demand = np.random.randint(1, 10, size=(size + 1))  # Demand, uniform integer 1 ... 9, Index 0 is NOT in use.
        max_capacity = CAPACITIES[size]
        dist_matrix = np.empty((size + 1, size + 1), dtype=np.float)
        for i in range(size + 1):
            for j in range(i, size + 1):
                d = distance(locations[i], locations[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return self.instance_info(size, size, locations, demand, max_capacity, dist_matrix)

    def load_instance(self, file_path, T_f=None):
        with open(file_path, "r") as f:
            data = json.loads(f.read())
        if "T_f" in data:
            self.T_f = data["T_f"]
        data["locations"] = np.asarray(data["locations"], dtype=np.float)
        data["demand"] = np.asarray(data["demand"], dtype=np.float)
        size = data["size"]
        dist_matrix = np.empty((size + 1, size + 1), dtype=np.float)
        for i in range(size + 1):
            for j in range(i, size + 1):
                d = distance(data["locations"][i], data["locations"][j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return self.instance_info(data["size"], data["size"], data["locations"], data["demand"], data['max_capacity'],
                                  dist_matrix)

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

    def criteria_accepted(self, accepted):
        if accepted:
            self.prev_costs = copy(self.costs)
            self.prev_sum_costs = copy(self.sum_costs)
            self.prev_max_weights = copy(self.max_weights)
            self.prev_d_loc = copy(self.d_loc)
        else:
            self.costs = copy(self.prev_costs)
            self.sum_costs = copy(self.prev_sum_costs)
            self.max_weights = copy(self.prev_max_weights)
            self.d_loc = copy(self.prev_d_loc)

    def update_from_other(self, other):  # I could do deepcopy here, but other is not going to be used anyway
        self.costs = other.costs
        self.sum_costs = other.sum_costs
        self.max_weights = other.max_weights
        self.d_loc = other.d_loc

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)  # shallow copy of every attribute, Is this enough?
        new.costs = copy(self.costs)
        new.sum_costs = copy(self.sum_costs)
        new.max_weights = copy(self.max_weights)
        new.d_loc = copy(self.d_loc)
        return new

    def _get_heus(self):
        remove_operators = [cvrp_remove_longest_tour_deviation_s, cvrp_remove_tour_neighbors,
                            cvrp_remove_xs, cvrp_remove_s, cvrp_remove_m, cvrp_remove_l, cvrp_remove_xl]
        insert_operators = [cvrp_insert_greedy, cvrp_insert_beam_search, cvrp_insert_by_variance, cvrp_insert_first]
        additional_operators = [cvrp_insert_single_best]
        len_r_op = len(remove_operators)
        len_i_op = len(insert_operators)
        len_operators = len_r_op * len_i_op
        heuristics = [additional_operators] + \
                    [(remove_operators[i // len_i_op], insert_operators[i % len_i_op]) for i in range(len_operators)]
        heuristic_names = [
             '{:03d}_'.format(i+1) + '_and_'.join([str(n).split()[1] for n in name])
            for i, name in enumerate(heuristics)]

        return dict(zip(heuristic_names, heuristics))