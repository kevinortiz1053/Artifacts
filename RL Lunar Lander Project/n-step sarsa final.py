import gymnasium as gym
import random
import matplotlib.pyplot as plt


def initialize_state_and_actions(q_of_s_dict, state_as_key):
    # Assumes input is an empty dictionary, given the state as a key
    q_of_s_dict[state_as_key] = {
        0: 0,  # do nothing
        1: 0,  # fire left orientation engine
        2: 0,  # fire main engine
        3: 0  # fire right orientation engine
    }
    return q_of_s_dict


def scale_state_values(state_vector):
    # Input is entire 8-tuple state vector

    # Scales x_position and y_position to the first decimal place
    x_position = round(state_vector[0])
    state_vector[0] = x_position

    y_position = round(state_vector[1])
    state_vector[1] = y_position

    # Scales velocity variables and angle to the second decimal place
    v_x = round(state_vector[2], 1)
    state_vector[2] = v_x

    v_y = round(state_vector[3], 1)
    state_vector[3] = v_y

    angle = round(state_vector[4], 1)
    state_vector[4] = angle

    angular_v = round(state_vector[5])
    state_vector[5] = angular_v

    # Last 2 boolean values are left untouched
    return state_vector


def q_value_check(q, current_state):
    value = q[current_state][0]
    for action in q[current_state]:
        if q[current_state][action] != value:
            return False
    return True


def return_e_greedy_action(q_of_s_a, state):
    epsilon = 0.1
    r = random.random()
    if r <= epsilon or q_value_check(q_of_s_a, state):
        random_action = random.randint(0, 3)
        return random_action

    max_value = -10000
    max_action = None
    for action in q_of_s_a[state]:
        if q_of_s_a[state][action] > max_value:
            max_value = q_of_s_a[state][action]
            max_action = action
    return max_action


def q_update(q, little_t, g_value, state_record, action_record, alpha):
    current_state_key = state_record[little_t]
    current_action = action_record[little_t]
    current_q = q[current_state_key][current_action]

    update = current_q + alpha * (g_value - current_q)
    q[current_state_key][current_action] = update
    return q


def check_if_state_exists_in_dictionary(dictionary, dict_key):
    initialize_criteria = dictionary.setdefault(dict_key)
    if initialize_criteria is None:
        dictionary = initialize_state_and_actions(dictionary, dict_key)
    return dictionary


def return_g_value(little_t, n_step, big_t, reward_record, lamda):
    g = 0
    for sum_i in range(little_t + 1, min(little_t + n_step, big_t) + 1):
        g = g + (lamda ** (sum_i - little_t - 1)) * reward_record[sum_i]
    return g


def g_update(lamda, n_step, state_record, action_record, little_t, previous_g, q_of_s_a):
    state = state_record[little_t + n_step]
    action = action_record[little_t + n_step]
    new_g = previous_g + (lamda**n_step) * q_of_s_a[state][action]
    return new_g


def average_arrays(arrays_to_be_averaged):
    number_of_arrays_to_be_averaged = len(arrays_to_be_averaged)
    average_array = []
    for i in range(0, len(arrays_to_be_averaged[0])):
        sum1 = 0
        for array in arrays_to_be_averaged:
            sum1 = sum1 + array[i]
        average = sum1 / number_of_arrays_to_be_averaged
        average_array.append(average)
    return average_array


def n_sarsa(q, episodes, n_step, alpha):
    lamda = 0.9
    successes = 0
    success_counter = []
    episode_counter = []

    for episode in range(episodes):
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=2
        )

        start_state = env.reset(seed=123)[0]
        current_state = scale_state_values(start_state)
        current_state_key = str(current_state)
        q = check_if_state_exists_in_dictionary(q, current_state_key)
        action = return_e_greedy_action(q, current_state_key)

        reward_record = {}
        state_record = {}
        actions_record = {}

        step_counter = 0
        state_record[step_counter] = current_state_key
        actions_record[step_counter] = action

        big_t = 100000
        tau = -10
        reward = 0
        cumulative_reward = 0

        while tau != big_t - 1:
            if step_counter < big_t:
                action = actions_record[step_counter]
                res = env.step(action)

                next_state = res[0]
                next_state = scale_state_values(next_state)
                next_state_key = str(next_state)

                reward = res[1]
                cumulative_reward = cumulative_reward + reward

                reward_record[step_counter + 1] = reward
                state_record[step_counter + 1] = next_state_key
                done = res[2]

                if done is True:  # If S of t+1 is terminal
                    big_t = step_counter + 1
                else:
                    q = check_if_state_exists_in_dictionary(q, next_state_key)
                    next_action = return_e_greedy_action(q, next_state_key)
                    actions_record[step_counter + 1] = next_action

            tau = step_counter - n_step + 1
            if tau >= 0:
                g = return_g_value(tau, n_step, big_t, reward_record, lamda)
                if (tau + n_step) < big_t:
                    g = g_update(lamda, n_step, state_record, actions_record, tau, g, q)
                q = q_update(q, tau, g, state_record, actions_record, alpha)

            step_counter = step_counter + 1

            if reward == 100:
                successes = successes + 1
                reward = 0

        success_counter.append(successes)
        episode_counter.append(episode + 1)
    print('Number of times landed: ' + str(successes))
    plt.plot(episode_counter, success_counter)
    plt.title("n-step Sarsa, n = " + str(n_step) + ', alpha = ' + str(alpha))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Landings')
    plt.show()
    return q, success_counter, episode_counter


def averaged_run():
    alpha = 0.1
    n_step = 5
    success_array = []
    for k in range(10):
        optimal_policy, num_of_successes_array, episode_count_array = n_sarsa({}, 10000, n_step, alpha)
        print('k value is: ' + str(k))
        success_array.append(num_of_successes_array)
    averaged_success_array = average_arrays(success_array)
    plt.plot(episode_count_array, averaged_success_array)
    plt.title("(seed) n-step Sarsa (baseline), n = " + str(n_step) + ', alpha = ' + str(alpha))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Averaged Number of Landings')
    plt.show()


averaged_run()
