def state_rep_reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen(info):
    next_state = [info['reduced_dist'], info['dist_from_min'], info['no_improvement'], info['index_step'],
                  info['was_changed'], info['unseen']]
    next_state.extend(get_last_action(info))
    return next_state


def state_rep_reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen(info):
    next_state = [info['reduced_dist'], info['dist_from_min'], info['distance'], info['min_distance'],
                  info['no_improvement'], info['index_step'], info['was_changed'], info['unseen']]
    next_state.extend(get_last_action(info))
    return next_state


def state_rep_reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen(info):
    next_state = [info['reduced_dist'], info['dist_from_min'], info['T'] if info['T'] else 0, info['cs'] if info['cs'] else 0, # T used to be 50, cs used to be 1. I changed these to 0
                  info['no_improvement'], info['index_step'], info['was_changed'], info['unseen']]
    next_state.extend(get_last_action(info))
    return next_state


def state_rep_reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen(info):
    next_state = [info['reduced_dist'], info['dist_from_min'], info['distance'], info['min_distance'],
                  info['T'] if info['T'] else 0, info['cs'] if info['cs'] else 0, info['no_improvement'],  # T used to be 50, cs used to be 1. I changed these to 0
                  info['index_step'], info['was_changed'], info['unseen']]
    next_state.extend(get_last_action(info))
    return next_state


def get_last_action(info):
    next_state = []
    if info['reduced_dist'] < 0:
        delta_sign = -1.0
    # elif info['reduced_dist'] == 0:
    #     delta_sign = 0
    else:
        delta_sign = 1.0
    next_state.append(delta_sign)
    action_1_hot = [0] * info['n_actions']
    action_1_hot[info['action']] = 1
    next_state.extend(action_1_hot)
    return next_state