import random
import numpy as np
import matplotlib.pyplot as plt


def return_grid_world1():
    grid = [
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3]
    ]
    return grid


def return_grid_world2():
    grid = [
        [0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3]
    ]
    return grid


def initialize_q_values():
    q_values = {}
    for i in range(1, 7):
        for j in range(1, 10):
            coordinate = int(str(i) + str(j))
            q_values[coordinate] = {
                'up': 0,
                'down': 0,
                'right': 0,
                'left': 0
            }
    return q_values


def initialize_model():
    model = {}
    for i in range(1, 7):
        for j in range(1, 10):
            coordinate = str(i) + str(j) + 'up'
            model[coordinate] = []
    return model


def tie_check(max_action_array):
    r = random.randint(1, len(max_action_array))
    max_action = max_action_array[r - 1]
    return max_action


def return_e_greedy_action(current_state1, q_values1):
    max_value = -1000
    for action in q_values1[current_state1]:
        if q_values1[current_state1][action] > max_value:
            max_value = q_values1[current_state1][action]
    max_action_array = []
    for action in q_values1[current_state1]:
        if q_values1[current_state1][action] == max_value:
            max_action_array.append(action)
    max_action = tie_check(max_action_array)

    r = random.random()
    epsilon = 0.1
    if r < epsilon:
        r = random.randint(1, 4)
        if r == 1:
            max_action = 'up'
        elif r == 2:
            max_action = 'down'
        elif r == 3:
            max_action = 'right'
        elif r == 4:
            max_action = 'left'
    return max_action


def boundary_check(current_state1, action1):
    coordinate = str(current_state1)
    x_val = int(coordinate[1])
    y_val = int(coordinate[0])
    if action1 == 'up':
        y_val = y_val + 1
    if action1 == 'down':
        y_val = y_val - 1
    if action1 == 'right':
        x_val = x_val + 1
    if action1 == 'left':
        x_val = x_val - 1
    if y_val > 6 or y_val < 1:
        return True
    if x_val > 9 or x_val < 1:
        return True
    return False


def wall_check(current_state1, action1, grid1):
    coordinate = str(current_state1)
    x_val = int(coordinate[1])
    y_val = int(coordinate[0])
    if action1 == 'up':
        y_val = y_val + 1
    if action1 == 'down':
        y_val = y_val - 1
    if action1 == 'right':
        x_val = x_val + 1
    if action1 == 'left':
        x_val = x_val - 1
    if grid1[y_val - 1][x_val - 1] == 1:
        return True
    return False


def return_next_state(current_state1, action1, grid):
    b_check = boundary_check(current_state1, action1)
    if b_check:
        return current_state1
    w_check = wall_check(current_state1, action1, grid)
    if w_check:
        return current_state1
    coordinate = str(current_state1)
    x_val = int(coordinate[1])
    y_val = int(coordinate[0])
    if action1 == 'up':
        y_val = y_val + 1
        next_state = int(str(y_val) + str(x_val))
        return next_state
    if action1 == 'down':
        y_val = y_val - 1
        next_state = int(str(y_val) + str(x_val))
        return next_state
    if action1 == 'right':
        x_val = x_val + 1
        next_state = int(str(y_val) + str(x_val))
        return next_state
    if action1 == 'left':
        x_val = x_val - 1
        next_state = int(str(y_val) + str(x_val))
        return next_state


def return_reward_and_next_state(current_state1, action1, grid1, q_values):
    reward = 0
    next_state = return_next_state(current_state1, action1, grid1)
    if next_state == current_state1:
        action1 = return_e_greedy_action(current_state1, q_values)
        reward, next_state = return_reward_and_next_state(current_state1, action1, grid1, q_values)
    if next_state == 69:
        reward = 1
    return reward, next_state


def greedy_q_value(q_values, current_state):
    max_value = -1000
    for action in q_values[current_state]:
        if q_values[current_state][action] > max_value:
            max_value = q_values[current_state][action]
    return max_value


def q_update(q_values, reward, action, current_state, state_prime):
    q_value = q_values[current_state][action]
    q_value_state_prime = greedy_q_value(q_values, state_prime)
    alpha = .1
    lamda = 0.95
    updated_q_value = q_value + alpha * (reward + lamda * q_value_state_prime - q_value)
    q_values[current_state][action] = updated_q_value
    return q_values


def return_random_model_state(model):
    r = random.randint(1, len(model))
    counter = 0
    for state in model:
        if counter == r - 1:
            return state
        else:
            counter = counter + 1


def tabular_dyna_q(n_steps):
    grid1 = return_grid_world1()
    grid2 = return_grid_world2()
    q_vals = initialize_q_values()
    model = {}
    current_state = 14
    cumulative_reward = 0
    y_values = []
    x_values = []
    for i in range(3000):
        action = return_e_greedy_action(current_state, q_vals)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid1, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        model[model_state_action] = [reward, state_prime]
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        for planning in range(n_steps):
            random_model_state_action = return_random_model_state(model)
            random_state = int(random_model_state_action[0:2])
            random_action = random_model_state_action[2:]
            rew = model[random_model_state_action][0]
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    for i in range(3000, 6000):
        action = return_e_greedy_action(current_state, q_vals)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid2, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        model[model_state_action] = [reward, state_prime]
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        for planning in range(n_steps):
            random_model_state_action = return_random_model_state(model)
            random_state = int(random_model_state_action[0:2])
            random_action = random_model_state_action[2:]
            rew = model[random_model_state_action][0]
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    return x_values, y_values


def tabular_dyna_q_plus(n_steps):
    grid1 = return_grid_world1()
    grid2 = return_grid_world2()
    q_vals = initialize_q_values()
    model = {}
    time_step_counter = {}
    current_state = 14
    cumulative_reward = 0
    y_values = []
    x_values = []
    for i in range(3000):
        action = return_e_greedy_action(current_state, q_vals)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid1, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        time_step_for_bonus = time_step_counter.setdefault(model_state_action)
        if time_step_for_bonus is None:
            time_step_counter[model_state_action] = i
            model[model_state_action] = [reward, state_prime, 0]
        else:
            if i - time_step_for_bonus > 10:
                model[model_state_action] = [reward, state_prime, i - time_step_for_bonus]
                time_step_counter[model_state_action] = i
            else:
                model[model_state_action] = [reward, state_prime, 0]

        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        for planning in range(n_steps):
            random_model_state_action = return_random_model_state(model)
            random_state = int(random_model_state_action[0:2])
            random_action = random_model_state_action[2:]
            rew = model[random_model_state_action][0]
            k = 0.01
            bonus = k * (model[random_model_state_action][2])**0.5
            rew = rew + bonus
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    for i in range(3000, 6000):
        action = return_e_greedy_action(current_state, q_vals)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid2, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        time_step_for_bonus = time_step_counter.setdefault(model_state_action)
        if time_step_for_bonus is None:
            model[model_state_action] = [reward, state_prime, 0]
            time_step_counter[model_state_action] = i
        else:
            if i - time_step_for_bonus > 10:
                model[model_state_action] = [reward, state_prime, i - time_step_for_bonus]
                time_step_counter[model_state_action] = i
            else:
                model[model_state_action] = [reward, state_prime, 0]
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        for planning in range(n_steps):
            random_model_state_action = return_random_model_state(model)
            random_state = int(random_model_state_action[0:2])
            random_action = random_model_state_action[2:]
            rew = model[random_model_state_action][0]
            k = 0.01
            bonus = k * (model[random_model_state_action][2])**0.5
            rew = rew + bonus
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    return x_values, y_values


def return_e_greedy_action_exp(current_state, q_values, time_step_counter, current_time_step):
    max_value = -1000
    bonus_dict = {}
    for action in q_values[current_state]:
        state_action_pair = str(current_state) + action
        time_step_for_bonus = time_step_counter.setdefault(state_action_pair)
        if time_step_for_bonus is None:
            time_step_counter[state_action_pair] = current_time_step
            bonus_value = 0
        else:
            k = 0.01
            bonus_value = k * (current_time_step - time_step_for_bonus)**0.5
            time_step_counter[state_action_pair] = current_time_step
        bonus_dict[action] = bonus_value
        if q_values[current_state][action] + bonus_value > max_value:
            max_value = q_values[current_state][action] + bonus_value
    max_action_array = []
    for action in q_values[current_state]:
        bonus_value = bonus_dict[action]
        if q_values[current_state][action] + bonus_value == max_value:
            max_action_array.append(action)
    max_action = tie_check(max_action_array)

    r = random.random()
    epsilon = 0.01
    if r < epsilon:
        r = random.randint(1, 4)
        if r == 1:
            max_action = 'up'
        elif r == 2:
            max_action = 'down'
        elif r == 3:
            max_action = 'right'
        elif r == 4:
            max_action = 'left'
    return max_action, time_step_counter


def tabular_dyna_q_plus_exp(n_steps):
    grid1 = return_grid_world1()
    grid2 = return_grid_world2()
    q_vals = initialize_q_values()
    model = {}
    time_step_counter = {}
    current_state = 14
    cumulative_reward = 0
    y_values = []
    x_values = []
    for i in range(3000):
        action, time_step_counter = return_e_greedy_action_exp(current_state, q_vals, time_step_counter, i)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid1, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        model[model_state_action] = [reward, state_prime]
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        for planning in range(n_steps):
            random_model_state_action = return_random_model_state(model)
            random_state = int(random_model_state_action[0:2])
            random_action = random_model_state_action[2:]
            rew = model[random_model_state_action][0]
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    for i in range(3000, 6000):
        action, time_step_counter = return_e_greedy_action_exp(current_state, q_vals, time_step_counter, i)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid2, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        model[model_state_action] = [reward, state_prime]
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        for planning in range(n_steps):
            random_model_state_action = return_random_model_state(model)
            random_state = int(random_model_state_action[0:2])
            random_action = random_model_state_action[2:]
            rew = model[random_model_state_action][0]
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    return x_values, y_values


x, y = tabular_dyna_q(5)
x_plus, y_plus = tabular_dyna_q_plus(5)
x_plus_exp, y_plus_exp = tabular_dyna_q_plus_exp(5)
plt.plot(x, y)
plt.plot(x_plus, y_plus)
plt.plot(x_plus_exp, y_plus_exp)
plt.legend(['Dyna-Q', 'Dyna-Q+', 'Dyna-Q+prime'], loc="upper left")
plt.title('Dyna-Q vs Dyna-Q+ vs Dyna-Q+prime')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.show()

