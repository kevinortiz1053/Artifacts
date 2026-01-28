# import packages
import gymnasium as gym
import pandas as pd
import numpy
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pickle

# set up the environment
env = gym.make(
    'LunarLander-v2',
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    # render_mode="human"
)


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


def check_if_state_exists_in_dictionary(dictionary, dict_key):
    initialize_criteria = dictionary.setdefault(dict_key)
    if initialize_criteria is None:
        dictionary = initialize_state_and_actions(dictionary, dict_key)
    return dictionary


def initialize_state_and_actions(q_of_s_dict, state_as_key):
    # Assumes input is an empty dictionary, given the state as a key
    q_of_s_dict[state_as_key] = {
        0: 0,  # do nothing
        1: 0,  # fire left orientation engine
        2: 0,  # fire main engine
        3: 0  # fire right orientation engine
    }
    return q_of_s_dict


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


def q_value_check(q, current_state):
    value = q[current_state][0]
    for action in q[current_state]:
        if q[current_state][action] != value:
            return False
    return True


# Initialize the Q-table
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# maybe I want to intialize Q as a dictionary
Q = defaultdict(lambda: [0] * env.action_space.n)


def q_learning(num_episodes, alpha, curriculum=True):
    list_dict = {}
    for j in range(10):
        gamma = 0.99  # discount factor
        epsilon = 0.1
        env = gym.make(
            'LunarLander-v2',
            continuous=False,
            gravity=-10,
            enable_wind=False,
            wind_power=0.0,
            turbulence_power=0.0
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        Q = defaultdict(lambda: [0] * env.action_space.n)

        # Train the agent for a fixed number of episodes
        episode_rewards_average = []
        episode_rewards_sum = []
        episodes = []
        count_list = []
        landed_list = []
        landings = 0

        for i in range(num_episodes):
            done = False
            reward_list = []
            episodes.append(i)

            if curriculum is True:
                # print("Running Curriculum Learning...")

                if i >= 2000 and i < 4000:
                    env = gym.make(
                        'LunarLander-v2',
                        continuous=False,
                        gravity=-10,
                        enable_wind=True,
                        wind_power=5.0,
                        turbulence_power=0.0,
                        # render_mode="human"
                    )
                elif i >= 4000 and i < 6000:
                    env = gym.make(
                        'LunarLander-v2',
                        continuous=False,
                        gravity=-10,
                        enable_wind=True,
                        wind_power=10.0,
                        turbulence_power=0.0
                    )

                elif i >= 6000 and i < 8000:
                    env = gym.make(
                        'LunarLander-v2',
                        continuous=False,
                        gravity=-10,
                        enable_wind=True,
                        wind_power=15.0,
                        turbulence_power=1.0
                    )

                elif i >= 8000:
                    env = gym.make(
                        'LunarLander-v2',
                        continuous=False,
                        gravity=-10,
                        enable_wind=True,
                        wind_power=15.0,
                        turbulence_power=2.0
                    )

                else:
                    env = gym.make(
                        'LunarLander-v2',
                        continuous=False,
                        gravity=-10,
                        enable_wind=False,
                        wind_power=0.0,
                        turbulence_power=0.0
                    )
            else:
                env = gym.make(
                    'LunarLander-v2',
                    continuous=False,
                    gravity=-10,
                    enable_wind=True,
                    wind_power=15.0,
                    turbulence_power=2.0
                )

            # Reduce the exploration rate at each episode and ensure we don't go below min epsilon
            # epsilon *= epsilon_decay
            # epsilon = max(epsilon, min_epsilon)
            state = env.reset(seed=223)[0]
            state = scale_state_values(state)
            counter = 0
            while not done:
                # Choose an action using epsilon-greedy exploration
                state = scale_state_values(state)
                state_tuple = tuple(state)  # initialize the state tuple.
                if state_tuple not in Q:
                    Q[state_tuple] = np.zeros(num_actions)

                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()  # take a random action

                else:
                    action = np.argmax(Q[state_tuple])

                # env.render()

                # Take action and observe reward and next state
                res = env.step(action)
                next_state = res[0]
                reward = res[1]
                reward_list.append(reward)
                done = res[2]

                # Update Q-value for the state-action pair
                next_state_tuple = tuple(next_state)

                # if the next state is not in Q
                if next_state_tuple not in Q:
                    Q[next_state_tuple] = np.zeros(num_actions)

                Q[state_tuple][action] = alpha * (reward + gamma * np.max(Q[next_state_tuple]) - \
                                                  Q[state_tuple][action])
                counter += 1
                if reward == 100:
                    print(f"episode: {i} ----------> Landed Successfully <------------------")
                    landings += 1
                    # landed_list.append(landings)
                    print(landings)
                # elif res[2] and reward != 100:
                # landed_list.append(0)
                done = res[2]
                # Update state
                state = next_state
            landed_list.append(landings)
            count_list.append(counter)
            ave_reward = statistics.mean(reward_list)
            episode_rewards_average.append(ave_reward)
            sum_rewards = sum(reward_list)
            episode_rewards_sum.append(sum_rewards)

        list_dict[j] = landed_list

    combined_lists = zip(list_dict[0], list_dict[1], list_dict[2], list_dict[3], list_dict[4], list_dict[5], \
                         list_dict[6], list_dict[7], list_dict[8], list_dict[9])

    average_list = [sum(values) / len(values) for values in combined_lists]
    return average_list, Q
    # return episodes, episode_rewards_average, episode_rewards_sum, count_list, landed_list

    # Print total reward for episode
    # print("Episode {}: Total reward = {}".format(i, reward))


def q_learning_single(num_episodes, alpha, curriculum=True):
    list_dict = {}
    gamma = 0.99  # discount factor
    epsilon = 0.1
    env = gym.make(
        'LunarLander-v2',
        continuous=False,
        gravity=-10,
        enable_wind=False,
        wind_power=0.0,
        turbulence_power=0.0
    )

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    Q = defaultdict(lambda: [0] * env.action_space.n)

    # Train the agent for a fixed number of episodes
    episode_rewards_average = []
    episode_rewards_sum = []
    episodes = []
    count_list = []
    landed_list = []
    landings = 0

    for i in range(num_episodes):
        done = False
        reward_list = []
        episodes.append(i)

        if curriculum is True:
            # print("Running Curriculum Learning...")

            if i >= 2000 and i < 4000:
                env = gym.make(
                    'LunarLander-v2',
                    continuous=False,
                    gravity=-10,
                    enable_wind=True,
                    wind_power=5.0,
                    turbulence_power=0.0,
                    # render_mode="human"
                )
            elif i >= 4000 and i < 6000:
                env = gym.make(
                    'LunarLander-v2',
                    continuous=False,
                    gravity=-10,
                    enable_wind=True,
                    wind_power=10.0,
                    turbulence_power=0.0
                )

            elif i >= 6000 and i < 8000:
                env = gym.make(
                    'LunarLander-v2',
                    continuous=False,
                    gravity=-10,
                    enable_wind=True,
                    wind_power=15.0,
                    turbulence_power=1.0
                )

            elif i >= 8000:
                env = gym.make(
                    'LunarLander-v2',
                    continuous=False,
                    gravity=-10,
                    enable_wind=True,
                    wind_power=15.0,
                    turbulence_power=2.0
                )

            else:
                env = gym.make(
                    'LunarLander-v2',
                    continuous=False,
                    gravity=-10,
                    enable_wind=False,
                    wind_power=0.0,
                    turbulence_power=0.0
                )
        else:
            env = gym.make(
                'LunarLander-v2',
                continuous=False,
                gravity=-10,
                enable_wind=True,
                wind_power=15.0,
                turbulence_power=2.0
            )

        # Reduce the exploration rate at each episode and ensure we don't go below min epsilon
        # epsilon *= epsilon_decay
        # epsilon = max(epsilon, min_epsilon)
        state = env.reset(seed=223)[0]
        state = scale_state_values(state)
        counter = 0
        while not done:
            # Choose an action using epsilon-greedy exploration
            state = scale_state_values(state)
            state_tuple = tuple(state)  # initialize the state tuple.
            if state_tuple not in Q:
                Q[state_tuple] = np.zeros(num_actions)

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # take a random action

            else:
                action = np.argmax(Q[state_tuple])

            # env.render()

            # Take action and observe reward and next state
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            reward_list.append(reward)
            done = res[2]

            # Update Q-value for the state-action pair
            next_state_tuple = tuple(next_state)

            # if the next state is not in Q
            if next_state_tuple not in Q:
                Q[next_state_tuple] = np.zeros(num_actions)

            Q[state_tuple][action] = alpha * (reward + gamma * np.max(Q[next_state_tuple]) -
                                              Q[state_tuple][action])
            counter += 1
            if reward == 100:
                print(f"episode: {i} ----------> Landed Successfully <------------------")
                landings += 1
                # landed_list.append(landings)
                print(landings)
            # elif res[2] and reward != 100:
            # landed_list.append(0)
            done = res[2]
            # Update state
            state = next_state
        landed_list.append(landings)
        count_list.append(counter)
        ave_reward = statistics.mean(reward_list)
        episode_rewards_average.append(ave_reward)
        sum_rewards = sum(reward_list)
        episode_rewards_sum.append(sum_rewards)

    # list_dict[j] = landed_list

    # combined_lists = zip(list_dict[0], list_dict[1], list_dict[2], list_dict[3], list_dict[4], list_dict[5], \
    # list_dict[6], list_dict[7], list_dict[8], list_dict[9])

    # average_list = [sum(values) / len(values) for values in combined_lists]
    return Q


# 5K run -Q-learning
def x_axis_array():
    x = []
    for i in range(5000):
        x.append(i + 1)
    return x


def five_k_run2(num_of_episodes, optimal_policy1):
    epsilon = 0.1
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

        state = env.reset(seed=223)[0]
        state = scale_state_values(state)
        while not done:
            state = scale_state_values(state)
            state_tuple = tuple(state)  # initialize the state tuple.

            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # take a random action

            else:
                action = np.argmax(Q[state_tuple])
            # action = np.argmax(Q[state_tuple])

            # Take action and observe reward and next state
            res = env.step(action)
            next_state = res[0]
            reward = res[1]
            done = res[2]
            # Update state
            # state = next_state
            if reward == 100:
                successes = successes + 1
            state = scale_state_values(next_state)
            # print(current_state_key)
        success_counter.append(successes)
    x_vals = x_axis_array()
    plt.plot(x_vals, success_counter)
    plt.title("Curriculum Q-learning - Optimal Policy")
    plt.xlabel("Episode Number")
    plt.ylabel("Number of Landings")
    plt.show()


def plot_results(episodes, average_list_baseline, average_list_curr, curriculum):
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(episodes, average_list_baseline, label='baseline')
    ax.plot(episodes, average_list_curr, label='curriculum')

    ax.set_xlabel('Epsiodes', fontsize=25)
    ax.set_ylabel('Average Number of Landings', fontsize=25)
    ax.legend(fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(f'Q-Learning - Curriculum {curriculum}', fontsize=30)
    ax.legend(prop={'size': 25})

    # calculate equation for trendline
    # z = np.polyfit(episodes, average_list_alpha5, 1)
    # p = np.poly1d(z)

    # add trendline to plot
    # ax.plot(episodes, p(episodes), "r--",linewidth=3.0)
    # ax.text(1, 1,"y=%.6fx+(%.6f)"%(z[0],z[1]), horizontalalignment='center',
    # verticalalignment='center',
    # transform=ax.transAxes, fontsize = 30)

    plt.show()
