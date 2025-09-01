import os
import os.path as osp
import numpy as np
import copy
from copy import deepcopy
import random
import ast
import gym
from gym import spaces
from DRLH.utils.reward import *
from DRLH.utils.utils import *




class Problem(gym.Env):

    def __init__(self, name: str, heuristics: dict, size=100, T_0=None, cs=None, max_steps=1000, warmup_steps=None
                 , dataset=None,
                 state_rep='reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen'
                 , reward_func='5310', acceptance_func="simulated_annealing_ac", T_f=None, pos_deltas_target=None,
                 initial_solutions=None, n_iterations_per_instance=1, num_cold_start_solves=1):

        self.size = size
        self.num_cold_start_solves = num_cold_start_solves

        #TODO
        # Should be a default operators list in here applicabale to all problems

        self.heuristics_list, self.heuristics_names = list(heuristics.values()), list(heuristics.keys())
        self.n_actions = len(self.heuristics_list)
        self.state_rep = {
            "reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen":
                state_rep_reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen,
            "reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen":
                state_rep_reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen,
            "reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen":
                state_rep_reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen,
            "reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen":
                state_rep_reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen
        }.get(state_rep, "not valid state representation")
        self.reward_func = {
            "5310": reward_func_5310,
            "10310": reward_func_10310,
            "pm": reward_func_pm,
            "pzm": reward_func_pzm,
            "delta_change": reward_func_delta_change,
            "delta_change_scaled": reward_func_delta_change_scaled,
            "new_best": reward_func_new_best,
            "new_best_p1": reward_func_new_best_p1,
            "min_distance": reward_func_min_distance,
        }.get(reward_func, "not valid reward function")
        self.n_iterations_per_instance = n_iterations_per_instance
        self.name = name
        self.observation_space = spaces.Box(low=-100, high=100,
                                            shape=(len(state_rep.split('___')) + self.n_actions + 1,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.size = size
        self.max_steps = max_steps
        self.np_random = None

        self.acceptance_func = {
            "record_to_record_ac": record_to_record_ac,
            "simulated_annealing_ac": simulated_annealing_ac,
        }.get(acceptance_func, "not valid acceptance function")
        self.T_0 = T_0  # If T_0, then T_f or cs must also be specified
        self.T_f = T_f
        if self.T_f is not None and self.T_0 is not None:
            self.cs = np.power((self.T_f / self.T_0), (1 / self.max_steps))
        else:
            self.cs = cs
        self.d_E_history = None
        self.pos_deltas_target = pos_deltas_target if pos_deltas_target is not None else -1
        self.warmup_steps = warmup_steps if warmup_steps is not None else -1

        # variables
        self.warmup_phase = None
        self.num_wasted_actions = None  # visualization only
        self.index_sample = -1  # int(input("index_sample-1")) #-1  #int(input("index_sample-1"))#-1
        self.n_resets = 0
        self.T = None
        self.solution = None
        self.best_solution = None
        self.distance = None
        self.min_distance = None
        self.prev_min_distance = None
        self.start_distance = None
        self.index_step = None
        self.min_step = None
        self.state = None
        self.seen_solutions = None
        self.no_improvement = None
        self.num_improvements = None
        self.num_best_improvements = None
        self.action_counter = None
        #
        self.next_solution = None
        self.next_distance = None
        self.unseen = None
        self.was_changed = None
        self.d_E = None
        self.accepted = None
        self.new_best = None
        self.action = None
        self.extra_params_to_log = None

        self.info = {  # <-- a dictionary containing information used by reward_func, state_rep, logging and debugging
            "n_actions": self.n_actions,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "T_start": None,
            "start_distance": None,
            "T": None,
            "cs": None,
            "solution": None,
            "best_solution": None,
            "distance": None,
            "min_distance": None,
            "prev_min_distance": None,
            "index_step": None,
            "min_step": None,
            "num_seen_solutions": None,
            "no_improvement": None,
            "num_improvements": None,
            "num_best_improvements": None,
            "action_counter": None,
            "next_solution": None,
            "next_distance": None,
            "unseen": None,
            "was_changed": None,
            "d_E": None,
            "dist_from_min": None,
            "accepted": None,
            "new_best": None,
            "action": None,
            "acceptance_prob": None,
            "reduced_dist": None,
            "len_d_E_history": None,
            "num_wasted_actions": None,
            "warmup_phase": None,
            "warmup_phase_end": None,
            "extra_params_to_log": None
        }
        file_path = __file__

        print(dataset)
        self.dataset = dataset
        if self.dataset is not None:
            self.dataset_path = osp.abspath(dataset)
            self.dataset_files = sorted(os.listdir(self.dataset_path), key=lambda f: int(
                f.split('_')[-1][:-4]))  # instance files should end in _NUM.txt where NUM a number.



        self.initial_solutions = initial_solutions
        if self.initial_solutions is not None:
            initial_solutions_path = osp.abspath(initial_solutions)
            self.initial_solutions = []
            with open(initial_solutions_path, "r") as file:
                for line in file:
                    self.initial_solutions.append(ast.literal_eval(line.strip()))

    def step(self, action):
        if self.warmup_phase and (self.index_step == self.warmup_steps or len(self.d_E_history) == self.pos_deltas_target):
            self.info["len_d_E_history"] = len(self.d_E_history)  # only updated once, here.
            self.d_E_history = [d_E for d_E in self.d_E_history if d_E < float('inf')]
            if len(self.d_E_history) == 0:
                self.d_E_history.append(self.T_f)  # emergency add in case there are no positive deltas. CHANGED from 0.1 to T_f, should not affect previous results
            avg_delta = np.mean(self.d_E_history)
            self.T = (-avg_delta / np.log(0.8))  # computed so that T starts at 0.8 acceptance probability given avg_delta
            T_f = self.T_f  # tune this hyper-parameter
            self.cs = np.power((T_f / self.T), (1 / self.max_steps))
            catalyst = "warmup_steps reached" if self.index_step == self.warmup_steps else f"pos_deltas_target reached, len_d_E_histor={len(self.d_E_history)}"
            print(f"{catalyst}")
            print(f"Computed a TEMP and CS based on the warmup steps and avg_delta={avg_delta}")
            print(f"TEMP={self.T},    CS={self.cs}")
            self.info["T_start"] = self.T
            self.warmup_phase = False  # to prevent this from triggering multiple times
            self.info["warmup_phase_end"] = self.index_step  # for logging

        # update num_wasted_actions (ONLY for visualization purposes)
        if self.action == action == 0 and self.was_changed == 0:  # If action 0 is attempted twice even though the first time did not work, then it's wasted.
            self.num_wasted_actions += 1

        # apply action to solution
        self.action = action
        op = self.heuristics_list[action]
        self.next_solution, self.next_distance = self.apply_heus(self.solution, op, prev_accepted=self.info["accepted"])

        # useful variables
        str_next_solution = str(self.next_solution)
        self.unseen = int(str_next_solution not in self.seen_solutions)
        self.was_changed = int(str(self.solution) != str_next_solution)
        self.d_E = self.next_distance - self.distance
        self.update_info()  # maybe?

        # update variables and decide on acceptance of new solution
        self.accepted = 1
        self.new_best = 0
        if self.d_E < 0:
            self.solution = self.next_solution
            self.distance = self.next_distance
            self.no_improvement = 0
            self.num_improvements += 1
            if self.next_distance < self.min_distance:
                self.prev_min_distance = self.min_distance
                self.min_distance = self.next_distance
                self.min_step = self.index_step
                self.best_solution = copy.deepcopy(self.next_solution)
                self.num_best_improvements += 1
                self.new_best = 1
        elif self.acceptance_func(self.info):
            self.solution = self.next_solution
            self.distance = self.next_distance
            self.no_improvement += 1
        else:
            self.accepted = 0
            self.no_improvement += 1
            self.was_changed = 0
            self.unseen = 0

        if self.warmup_phase and self.d_E > 0:
            self.d_E_history.append(self.d_E)

        #self.index_step += 1
        self.seen_solutions.add(str_next_solution)
        self.action_counter[self.heuristics_names[action]] += 1
        self.update_info()
        self.update_extra_info()
        if self.T is not None and self.cs is not None:
            self.T *= self.cs
        reward = self.reward_func(self.info)  # get reward
        self.state = self.state_rep(self.info)  # get new state
        self.index_step += 1  # moved this down here. Hopefully doesn't break anything.
        return self.state, reward, self.index_step == self.max_steps, self.info

    def reset(self):
        if self.n_resets % self.n_iterations_per_instance == 0:
            self.index_sample += 1
            if self.dataset is not None:
                self.index_sample %= len(self.dataset_files)

        self.n_resets += 1

        if self.dataset is not None:
            file_path = osp.join(self.dataset_path, self.dataset_files[self.index_sample])
            print(f"loading instance from file={file_path}")
            self.instance = self.load_instance(file_path=file_path, T_f=self.T_f)
        else:
            print(f"generating instance of size={self.size}")
            self.instance = self.generate_instance(size=self.size, index_sample=self.index_sample, T_f=self.T_f)

        if self.initial_solutions is not None:
            self.solution = self.initial_solutions[self.index_sample]
            print(f"initial solution used: {self.solution}")
        else:
            self.solution = self.generate_initial_solution()
            print(f"generating random solution: {self.solution}")


        if self.T_0 is None:  # dynamic start temp and alpha
            self.d_E_history = []
            self.T = None
            self.cs = None
            self.warmup_phase = True
        else:  # static start temp and cs
            self.T = self.T_0
            self.warmup_phase = False

        self.num_wasted_actions = 0
        self.seen_solutions = set()
        self.index_step = 0
        self.min_step = 0
        self.best_solution = copy.deepcopy(self.solution)
        self.min_distance = self.objective_function(self.solution)
        self.prev_min_distance = self.min_distance
        self.distance = self.min_distance
        self.start_distance = self.min_distance
        self.no_improvement = 0
        self.num_improvements = 0
        self.num_best_improvements = 0
        self.action_counter = {h_name: 0 for h_name in self.heuristics_names}
        #
        self.next_distance = self.distance
        self.unseen = 1
        self.was_changed = 1
        self.d_E = -1
        self.accepted = 1
        self.new_best = 1
        self.action = 0  # random
        #
        self.update_info()
        self.state = self.state_rep(self.info)
        return self.state

    def update_info(self):
        self.info.update({
            "start_distance": self.start_distance,
            "T": self.T,
            "cs": self.cs,
            "solution": str(self.solution),
            "best_solution": str(self.best_solution),
            "distance": self.distance,
            "min_distance": self.min_distance,
            "prev_min_distance": self.prev_min_distance,
            "index_step": self.index_step,
            "min_step": self.min_step,
            "num_seen_solutions": len(self.seen_solutions),
            "no_improvement": self.no_improvement,
            "num_improvements": self.num_improvements,
            "num_best_improvements": self.num_best_improvements,
            "action_counter": self.action_counter,
            "next_solution": str(self.next_solution),
            "next_distance": self.next_distance,
            "unseen": self.unseen,
            "was_changed": self.was_changed,
            "d_E": self.d_E,
            "dist_from_min": self.distance - self.min_distance,
            "accepted": self.accepted,
            "new_best": self.new_best,
            "action": self.action,
            "acceptance_prob": 1 if self.T is None else np.e ** (-self.d_E / self.T) if self.d_E > 0 else None,
            "reduced_dist": self.accepted * self.d_E,
            "num_wasted_actions": self.num_wasted_actions,
            "warmup_phase": self.warmup_phase,
            "extra_params_to_log": self.extra_params_to_log
        })

    def __len__(self):

        if self.dataset is not None:
            if isinstance(self.dataset, str):
                return len(self.dataset_files)
            elif isinstance(self.dataset, list):
                return len(self.dataset)
        else:
            return self.num_cold_start_solves

    def apply_heus(self, solution, heu, prev_accepted=True):
        if len(heu) > 1:
            solution = deepcopy(solution)
            op, op2 = heu
            solution, removed_objcts = op(self, solution)
            new_solution, new_cost = op2(self, solution, removed_objcts)

        else:
            new_solution, new_cost = heu(self, solution)
        return new_solution, new_cost

    def objective_function(self, solution):
        raise NotImplementedError

    def generate_initial_solution(self):
        raise NotImplementedError

    @classmethod
    def generate_instance(cls, size=None, index_sample=None, T_f=None):
        raise NotImplementedError

    @classmethod
    def load_instance(cls, file_path, T_f=None):
        raise NotImplementedError

    def _remove_reinsert(self, solution, heu, prev_accepted=True):
        pass

    def accepted(self):
        pass

    def update_extra_info(self):
        pass


