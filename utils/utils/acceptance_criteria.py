import numpy as np


def record_to_record_ac(info):
    return info["next_distance"] < float('inf') and info["unseen"] and info["next_distance"] < info["min_distance"] + \
           0.2*info["min_distance"]*((info["max_steps"] - info["index_step"]) / info["max_steps"])


def simulated_annealing_ac(info):
    return info["next_distance"] < float('inf') and info["unseen"] and \
           (info["warmup_phase"] or np.random.random() < np.e ** (-info["d_E"] / info["T"]))