import numpy as np
import random as rand
import matplotlib.pyplot as plt


def increment(action_values_sample, action_values_step):
    for i in range(len(action_values_sample)):
        delta = np.random.normal(loc=0.0, scale=.01)
        action_values_sample[i] = action_values_sample[i] + delta
        action_values_step[i] = action_values_step[i] + delta
    return action_values_sample, action_values_step


def initialize_q():
    initial_q = np.random.normal(loc=0.0, scale=1.0)
    print(initial_q)
    q_values_sample1 = [initial_q, initial_q, initial_q, initial_q, initial_q, initial_q, initial_q, initial_q,
                       initial_q, initial_q]
    q_values_step1 = [initial_q, initial_q, initial_q, initial_q, initial_q, initial_q, initial_q, initial_q,
                     initial_q, initial_q]
    return q_values_sample1, q_values_step1


def choose_action_index(action_values_sample, action_values_step):
    max_ind_sample = action_values_sample.index(max(action_values_sample))
    max_ind_step = action_values_step.index(max(action_values_step))
    r = rand.random()
    if r < .9:
        arm_ind_sample = max_ind_sample
        arm_ind_step = max_ind_step
    else:
        arm_ind_sample = rand.randint(0, 9)
        arm_ind_step = arm_ind_sample
    return arm_ind_sample, arm_ind_step


def average(lst):
    return sum(lst) / len(lst)


def get_rewards(sample, sample_ind, step, step_ind):
    reward_sample1 = np.random.normal(loc=sample[sample_ind], scale=1)
    reward_step1 = np.random.normal(loc=step[step_ind], scale=1)
    return reward_sample1, reward_step1


# Epsilon = 0.1
run_array_sample = []
run_array_step = []
init_q_sample, init_q_step = initialize_q()
for k in range(2000):  # Number of runs
    original_q_sample = init_q_sample.copy()
    original_q_step = init_q_step.copy()
    q_values_sample = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    q_values_step = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    reward_array_sample = []
    reward_array_step = []
    t = 0
    while t < 10000:  # Number of steps per run
        action_index_sample, action_index_step = choose_action_index(q_values_sample, q_values_step)
        original_q_sample, original_q_step = increment(original_q_sample, original_q_step)
        reward_sample, reward_step = get_rewards(original_q_sample, action_index_sample, original_q_step, action_index_step)
        n_count[action_index_sample] = n_count[action_index_sample] + 1
        q_values_sample[action_index_sample] = q_values_sample[action_index_sample] + (1/n_count[action_index_sample]) *\
                                 (reward_sample - q_values_sample[action_index_sample])
        q_values_step[action_index_step] = q_values_step[action_index_step] + .1 * (reward_step - q_values_step[action_index_step])
        t = t + 1
        reward_array_sample.append(q_values_sample[action_index_sample])
        reward_array_step.append(q_values_step[action_index_step])
    run_array_sample.append(reward_array_sample)
    run_array_step.append(reward_array_step)


def avg_run_reward(run_array):  # Returns y-values
    y_values = []
    for j in range(len(run_array[0])):
        temp_arr = []
        for z in run_array:
            temp_arr.append(z[j])
        avg = average(temp_arr)
        y_values.append(avg)
    return y_values


step_counter = []
for i in range(10000):  # Making values for x-axis
    step_counter.append(i+1)
sample_y_values = avg_run_reward(run_array_sample)
step_y_values = avg_run_reward(run_array_step)

plt.plot(step_counter, sample_y_values, 'b', label='Sample Average')
plt.plot(step_counter, step_y_values, 'r', label='Constant Step-size')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.title('Non-Stationary')
plt.show()
