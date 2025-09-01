

def reward_func_5310(info):  # encourages new best solutions and diversifying
    if info['d_E'] < 0:
        if info['new_best']:
            reward = 5
        else:
            reward = 3
    elif info['accepted']:
        reward = 1
    else:
        reward = 0
    return reward


def reward_func_10310(info):  # encourages new best solutions (extra) and diversifying
    if info['d_E'] < 0:
        if info['new_best']:
            reward = 10
        else:
            reward = 3
    elif info['accepted']:
        reward = 1
    else:
        reward = 0
    return reward


def reward_func_pm(info):  # simple +-1
    if info['d_E'] < 0:
        reward = 1
    else:
        reward = -1
    return reward


def reward_func_pzm(info):  # simple +-1, neutral when diversifying
    if info['d_E'] < 0:
        reward = 1
    elif info['accepted']:
        reward = 0
    else:
        reward = -1
    return reward


def reward_func_delta_change(info):  # rewards according to improvement
    reward = -info['reduced_dist']
    return reward


def reward_func_delta_change_scaled(info):  # rewards improvement, but does not punish diversification so much
    if info['reduced_dist'] < 0:
        reward = -info['reduced_dist']
    else:
        reward = -info['reduced_dist'] // 2
    return reward


def reward_func_new_best(info):
    if info['new_best']:
        reward = info["prev_min_distance"] - info["min_distance"]
    else:
        reward = 0
    return reward


def reward_func_new_best_p1(info):
    if info['new_best']:
        reward = 1
    else:
        reward = 0
    return reward


def reward_func_min_distance(info):  # used together with self-baseline to encourage finding good min_solutions quickly.
    reward = info['min_distance']
    return reward