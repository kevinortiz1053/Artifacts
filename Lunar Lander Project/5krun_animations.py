import gymnasium as gym
import random
import pickle
import matplotlib.pyplot as plt


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


def initialize_state_and_actions(q_of_s_dict, state_as_key):
    # Assumes input is an empty dictionary, given the state as a key
    q_of_s_dict[state_as_key] = {
        0: 0,  # do nothing
        1: 0,  # fire left orientation engine
        2: 0,  # fire main engine
        3: 0  # fire right orientation engine
    }
    return q_of_s_dict


def check_if_state_exists_in_dictionary(dictionary, dict_key):
    initialize_criteria = dictionary.setdefault(dict_key)
    if initialize_criteria is None:
        dictionary = initialize_state_and_actions(dictionary, dict_key)
    return dictionary


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


with open('curr_sarsa_baseline.pkl', 'rb') as fp:
    optimal_policy = pickle.load(fp)
print("Length of optimal policy: " + str(len(optimal_policy)))
print(optimal_policy)


def one_run(optimal_policy1):
    done = False
    env = gym.make(
        'LunarLander-v2',
        continuous=False,
        gravity=-10.0,
        enable_wind=True,
        wind_power=15.0,
        turbulence_power=2,
        render_mode="human"
    )
    start_state = env.reset(seed=123)[0]
    current_state = scale_state_values(start_state)
    current_state_key = str(current_state)
    while not done:
        optimal_policy1 = check_if_state_exists_in_dictionary(optimal_policy1, current_state_key)
        action = return_e_greedy_action(optimal_policy1, current_state_key)
        res = env.step(action)
        next_state = res[0]
        reward = res[1]
        done = res[2]
        if reward == 100:
            print("---------LANDED----------")
            print("landing state: " + str(next_state))
            print(" ")
        current_state_key = str(scale_state_values(next_state))


def for_animation(optimal_policy1):  # Function to get video of Lander landing
    for i in range(1000):
        if i % 100 == 0:
            print("Animation Number: " + str(i))
        one_run(optimal_policy1)


# for_animation(optimal_policy)


def x_axis_array():
    x = []
    for i in range(5000):
        x.append(i + 1)
    return x


def five_k_run(num_of_episodes, optimal_policy1):
    success_counter = []
    successes = 0
    for episode in range(num_of_episodes):
        done = False
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10.0,
            enable_wind=True,
            wind_power=15.0,
            turbulence_power=2,
        )
        start_state = env.reset(seed=123)[0]
        current_state = scale_state_values(start_state)
        current_state_key = str(current_state)
        while not done:
            optimal_policy1 = check_if_state_exists_in_dictionary(optimal_policy1, current_state_key)
            action = return_e_greedy_action(optimal_policy1, current_state_key)
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]
            if reward == 100:
                successes = successes + 1
            current_state_key = str(scale_state_values(next_state))
        success_counter.append(successes)
    x_vals = x_axis_array()
    plt.plot(x_vals, success_counter)
    plt.title("Curriculum Sarsa - Optimal Policy")
    plt.xlabel("Episode Number")
    plt.ylabel("Number of Landings")
    plt.show()


five_k_run(5000, optimal_policy)
