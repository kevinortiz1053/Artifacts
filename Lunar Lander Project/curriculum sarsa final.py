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


def q_update(q, current_state_key, current_action, next_state_key, next_action, reward, alpha):
    lamda = 0.9
    current_q = q[current_state_key][current_action]
    next_q = q[next_state_key][next_action]
    update = current_q + alpha * (reward + lamda * next_q - current_q)
    q[current_state_key][current_action] = update
    return q


def check_if_state_exists_in_dictionary(dictionary, dict_key):
    initialize_criteria = dictionary.setdefault(dict_key)
    if initialize_criteria is None:
        dictionary = initialize_state_and_actions(dictionary, dict_key)
    return dictionary


def sarsa_c1(q, episodes, alpha):
    successes = 0
    success_counter = []
    for episode in range(episodes):
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=0.0,
            turbulence_power=0
        )

        start_state = env.reset(seed=123)[0]
        current_state = scale_state_values(start_state)
        current_state_key = str(current_state)

        done = False

        while not done:
            q = check_if_state_exists_in_dictionary(q, current_state_key)

            action = return_e_greedy_action(q, current_state_key)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]
            if reward == 100:
                successes = successes + 1

            next_state = scale_state_values(next_state)
            next_state_key = str(next_state)

            q = check_if_state_exists_in_dictionary(q, next_state_key)

            next_action = return_e_greedy_action(q, next_state_key)
            q = q_update(q, current_state_key, action, next_state_key, next_action, reward, alpha)

            current_state_key = next_state_key
        success_counter.append(successes)
    print('Number of times landed in C1: ' + str(successes))
    return q, success_counter


def sarsa_c2(q, episodes, alpha, last_num_of_successes):
    successes = last_num_of_successes
    success_counter = []
    for episode in range(episodes):
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=5.0,
            turbulence_power=0
        )

        start_state = env.reset(seed=123)[0]
        current_state = scale_state_values(start_state)
        current_state_key = str(current_state)

        done = False

        while not done:
            q = check_if_state_exists_in_dictionary(q, current_state_key)

            action = return_e_greedy_action(q, current_state_key)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]
            if reward == 100:
                successes = successes + 1

            next_state = scale_state_values(next_state)
            next_state_key = str(next_state)

            q = check_if_state_exists_in_dictionary(q, next_state_key)

            next_action = return_e_greedy_action(q, next_state_key)
            q = q_update(q, current_state_key, action, next_state_key, next_action, reward, alpha)

            current_state_key = next_state_key
        success_counter.append(successes)
    print('Number of times landed in C2: ' + str(successes))
    return q, success_counter


def sarsa_c3(q, episodes, alpha, last_num_of_successes):
    successes = last_num_of_successes
    success_counter = []
    for episode in range(episodes):
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=10.0,
            turbulence_power=0
        )

        start_state = env.reset(seed=123)[0]
        current_state = scale_state_values(start_state)
        current_state_key = str(current_state)

        done = False

        while not done:
            q = check_if_state_exists_in_dictionary(q, current_state_key)

            action = return_e_greedy_action(q, current_state_key)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]
            if reward == 100:
                successes = successes + 1

            next_state = scale_state_values(next_state)
            next_state_key = str(next_state)

            q = check_if_state_exists_in_dictionary(q, next_state_key)

            next_action = return_e_greedy_action(q, next_state_key)
            q = q_update(q, current_state_key, action, next_state_key, next_action, reward, alpha)

            current_state_key = next_state_key
        success_counter.append(successes)
    print('Number of times landed in C4: ' + str(successes))
    return q, success_counter


def sarsa_c4(q, episodes, alpha, last_num_of_successes):
    successes = last_num_of_successes
    success_counter = []
    for episode in range(episodes):
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=1
        )

        start_state = env.reset(seed=123)[0]
        current_state = scale_state_values(start_state)
        current_state_key = str(current_state)

        done = False

        while not done:
            q = check_if_state_exists_in_dictionary(q, current_state_key)

            action = return_e_greedy_action(q, current_state_key)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]
            if reward == 100:
                successes = successes + 1

            next_state = scale_state_values(next_state)
            next_state_key = str(next_state)

            q = check_if_state_exists_in_dictionary(q, next_state_key)

            next_action = return_e_greedy_action(q, next_state_key)
            q = q_update(q, current_state_key, action, next_state_key, next_action, reward, alpha)

            current_state_key = next_state_key
        success_counter.append(successes)
    print('Number of times landed in C4: ' + str(successes))
    return q, success_counter


def sarsa(q, episodes, alpha, last_num_of_successes):
    successes = last_num_of_successes
    episode_counter = []
    success_counter = []
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

        done = False

        while not done:
            initialize_criteria = q.setdefault(current_state_key)
            if initialize_criteria is None:
                q = initialize_state_and_actions(q, current_state_key)

            action = return_e_greedy_action(q, current_state_key)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]

            next_state = scale_state_values(next_state)
            next_state_key = str(next_state)

            initialize_criteria = q.setdefault(next_state_key)
            if initialize_criteria is None:
                q = initialize_state_and_actions(q, next_state_key)

            next_action = return_e_greedy_action(q, next_state_key)
            q = q_update(q, current_state_key, action, next_state_key, next_action, reward, alpha)

            current_state_key = next_state_key
            if reward == 100:
                successes = successes + 1
        success_counter.append(successes)
        episode_counter.append(episode + 1)
    print('Number of times landed: ' + str(successes))
    plt.plot(episode_counter, success_counter)
    plt.title('Sarsa (Curriculum 3), alpha = ' + str(alpha))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Number of Landings')
    plt.show()
    return q, success_counter


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


def x_axis_array():
    x = []
    for i in range(10000):
        x.append(i + 1)
    return x


def curriculum_learning(num_episodes1, alpha1):
    success_array = []
    num_episodes = int(num_episodes1 / 5)

    policy, num_of_successes_array = sarsa_c1({}, num_episodes, alpha1)
    success_array = success_array + num_of_successes_array
    last_success_number = num_of_successes_array[len(num_of_successes_array) - 1]

    policy, num_of_successes_array = sarsa_c2(policy, num_episodes, alpha1, last_success_number)
    success_array = success_array + num_of_successes_array
    last_success_number = num_of_successes_array[len(num_of_successes_array) - 1]

    policy, num_of_successes_array = sarsa_c3(policy, num_episodes, alpha1, last_success_number)
    success_array = success_array + num_of_successes_array
    last_success_number = num_of_successes_array[len(num_of_successes_array) - 1]

    policy, num_of_successes_array = sarsa_c4(policy, num_episodes, alpha1, last_success_number)
    success_array = success_array + num_of_successes_array
    last_success_number = num_of_successes_array[len(num_of_successes_array) - 1]

    policy, num_of_successes_array = sarsa(policy, num_episodes, alpha1, last_success_number)
    success_array = success_array + num_of_successes_array

    return success_array


def averaged_run():
    alpha = 0.5
    success_array = []
    for k in range(10):
        num_of_successes_array = curriculum_learning(10000, alpha)
        print('k value is: ' + str(k))
        success_array.append(num_of_successes_array)
    averaged_success_array = average_arrays(success_array)
    episode_count_array = x_axis_array()
    plt.plot(episode_count_array, averaged_success_array)
    plt.title("(seed) Curriculum 3 Sarsa, " + 'alpha = ' + str(alpha))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Averaged Number of Landings')
    plt.show()


averaged_run()
