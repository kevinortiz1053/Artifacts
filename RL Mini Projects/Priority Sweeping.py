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


def return_grid_world3():
    grid = [
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    ]
    return grid


def return_grid_world4():
    grid = [
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    ]
    return grid


def return_grid_world5():
    grid = [
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    ]
    return grid


def return_grid_world6():
    grid = [
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
    ]
    return grid


def initialize_q_values(grid):
    q_values = {}
    for i in range(1, len(grid) + 1):
        for j in range(1, len(grid[0]) + 1):
            coordinate = int(str(i) + str(j))
            q_values[coordinate] = {
                'up': 0,
                'down': 0,
                'right': 0,
                'left': 0
            }
    print(q_values)
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


def boundary_check(current_state1, action1, grid):
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
    print('final x: ' + str(x_val))
    print('final y: ' + str(y_val))
    if y_val > len(grid) or y_val < 1:
        return True
    if x_val > len(grid[0]) or x_val < 1:
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
    b_check = boundary_check(current_state1, action1, grid)
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
    if action1 == 'down':
        y_val = y_val - 1
        next_state = int(str(y_val) + str(x_val))
    if action1 == 'right':
        x_val = x_val + 1
        next_state = int(str(y_val) + str(x_val))
    if action1 == 'left':
        x_val = x_val - 1
        next_state = int(str(y_val) + str(x_val))

    print('next state: ' + str(next_state))
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


def q_update_sweep(q_values, reward, action, current_state, state_prime):
    q_value = q_values[current_state][action]
    q_value_state_prime = greedy_q_value(q_values, state_prime)
    lamda = 0.95
    update_size = reward + lamda * q_value_state_prime - q_value
    return update_size


def return_random_model_state(model):
    r = random.randint(1, len(model))
    counter = 0
    for state in model:
        if counter == r - 1:
            return state
        else:
            counter = counter + 1


def get_state_value(state_action_pair):
    state = ''
    for i in state_action_pair:
        if i.isdigit():
            state = state + i
    state = int(state)
    return state


def get_action_value(state_action_pair):
    action = ''
    for i in state_action_pair:
        if not i.isdigit():
            action = action + i
    return action


def tabular_dyna_q(n_steps):
    grid1 = return_grid_world1()
    grid2 = return_grid_world2()
    q_vals = initialize_q_values(grid1)
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
            random_state = get_state_value(random_model_state_action)
            random_action = get_action_value(random_model_state_action)
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
            random_state = get_state_value(random_model_state_action)
            random_action = get_action_value(random_model_state_action)
            rew = model[random_model_state_action][0]
            s_prime = model[random_model_state_action][1]
            q_vals = q_update(q_vals, rew, random_action, random_state, s_prime)
    return x_values, y_values


def boundary_check_sweep(current_state1, grid):
    coordinate = str(current_state1)
    x_val = int(coordinate[1])
    y_val = int(coordinate[0])
    if y_val > len(grid) or y_val < 1:
        return True
    if x_val > len(grid[0]) or x_val < 1:
        return True
    return False


def wall_check_sweep(current_state1, grid1):
    coordinate = str(current_state1)
    x_val = int(coordinate[1])
    y_val = int(coordinate[0])
    if grid1[y_val - 1][x_val - 1] == 1:
        return True
    return False


def model_check_sweep(model, state_action_pair):
    model1 = model.copy()
    if model1.setdefault(state_action_pair) is None:
        return True
    return False


def return_predecessor_states(current_state, grid, model):
    state_action_array = []
    y_value = int(str(current_state)[0])
    x_value = int(str(current_state)[1])

    c_state = int(str(y_value + 1) + str(x_value))  # Coming from top
    c_action = 'down'
    if boundary_check_sweep(c_state, grid) is False and wall_check_sweep(c_state, grid) is False and model_check_sweep(model, str(c_state) + c_action) is False:
        state_action_array.append(str(c_state) + c_action)

    if y_value - 1 != 0:
        c_state = int(str(y_value - 1) + str(x_value))  # Coming from bottom
        c_action = 'up'
        if boundary_check_sweep(c_state, grid) is False and wall_check_sweep(c_state, grid) is False and model_check_sweep(model, str(c_state) + c_action) is False:
            state_action_array.append(str(c_state) + c_action)

    c_state = int(str(y_value) + str(x_value + 1))  # Coming from the right
    c_action = 'left'
    if boundary_check_sweep(c_state, grid) is False and wall_check_sweep(c_state, grid) is False and model_check_sweep(model, str(c_state) + c_action) is False:
        state_action_array.append(str(c_state) + c_action)

    if x_value - 1 != 0:
        c_state = int(str(y_value) + str(x_value - 1))  # Coming from the left
        c_action = 'right'
        if boundary_check_sweep(c_state, grid) is False and wall_check_sweep(c_state, grid) is False and model_check_sweep(model, str(c_state) + c_action) is False:
            state_action_array.append(str(c_state) + c_action)

    return state_action_array


def prioritized_sweeping(n_steps):
    grid1 = return_grid_world1()
    grid2 = return_grid_world2()
    q_vals = initialize_q_values(grid1)
    model = {}
    P = {}
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
        update_size = q_update_sweep(q_vals, reward, action, current_state, state_prime)
        if update_size > 2:
            P[model_state_action] = update_size
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        counter = 0
        while counter < n_steps and len(P) > 0:
            max_key = max(P)
            P.pop(max_key)
            reward = model[max_key][0]
            state_prime = model[max_key][1]
            c_state = get_state_value(max_key)
            c_action = get_action_value(max_key)
            q_vals = q_update(q_vals, reward, c_action, c_state, state_prime)
            state_action_array = return_predecessor_states(c_state, grid1, model)
            for s in state_action_array:
                predicted_reward = model[s][0]
                predecessor_state = get_state_value(s)
                predecessor_action = get_action_value(s)
                update_size = q_update_sweep(q_vals, predicted_reward, predecessor_action, predecessor_state, c_state)
                if update_size > 2:
                    P[s] = update_size
            counter = counter + 1
    for i in range(3000, 6000):
        action = return_e_greedy_action(current_state, q_vals)
        reward, state_prime = return_reward_and_next_state(current_state, action, grid2, q_vals)
        if state_prime == 69:
            reward = reward + 1
        cumulative_reward = cumulative_reward + reward
        q_vals = q_update(q_vals, reward, action, current_state, state_prime)
        model_state_action = str(current_state) + action
        model[model_state_action] = [reward, state_prime]
        update_size = q_update_sweep(q_vals, reward, action, current_state, state_prime)
        if update_size > 2:
            P[model_state_action] = update_size
        current_state = state_prime
        if state_prime == 69:
            current_state = 14
        x_values.append(i + 1)
        y_values.append(cumulative_reward)
        counter = 0
        while counter < n_steps and len(P) > 0:
            max_key = max(P)
            P.pop(max_key)
            reward = model[max_key][0]
            state_prime = model[max_key][1]
            c_state = get_state_value(max_key)
            c_action = get_action_value(max_key)
            q_vals = q_update(q_vals, reward, c_action, c_state, state_prime)
            state_action_array = return_predecessor_states(c_state, grid1, model)
            for s in state_action_array:
                predicted_reward = model[s][0]
                predecessor_state = get_state_value(s)
                predecessor_action = get_action_value(s)
                update_size = q_update_sweep(q_vals, predicted_reward, predecessor_action, predecessor_state, c_state)
                if update_size > 2:
                    P[s] = update_size
            counter = counter + 1

    return x_values, y_values


x, y = tabular_dyna_q(5)
x_sweep, y_sweep = prioritized_sweeping(5)
plt.plot(x, y)
plt.plot(x_sweep, y_sweep)
plt.legend(['Dyna-Q', 'Prioritized Sweeping'], loc="upper left")
plt.title('Dyna-Q vs Prioritized Sweeping')
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.show()
