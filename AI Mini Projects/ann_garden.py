import numpy as np
import random
import math
import matplotlib.pyplot as plt


filename = 'ANN - Iris data.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=0, dtype=str)
for i in data:
    if i[4] == 'Iris-setosa':
        i[4] = 1
    elif i[4] == 'Iris-versicolor':
        i[4] = 2
    elif i[4] == 'Iris-virginica':
        i[4] = 3
data = data.astype('float64')
data = data.tolist()
for i in data:
    if i[4] == 1:
        i[4] = [1, 0, 0]
    elif i[4] == 2:
        i[4] = [0, 1, 0]
    elif i[4] == 3:
        i[4] = [0, 0, 1]
# Pre-processing done


def normalize(data_set):  # percentage normalization
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for sample in data_set:
        s1 = s1 + sample[0]
        s2 = s2 + sample[1]
        s3 = s3 + sample[2]
        s4 = s4 + sample[3]
    for sample in data_set:
        sample[0] = sample[0] / s1
        sample[1] = sample[1] / s2
        sample[2] = sample[2] / s3
        sample[3] = sample[3] / s4
    return data


def train_test_sets(data_set):  # 80%, 20%
    train1 = data_set[0:40] + data_set[50:90] + data_set[100:140]
    random.shuffle(train1)
    test1 = data_set[40:50] + data_set[90:100] + data_set[140:]
    random.shuffle(test1)
    return train1, test1


def forward_prop(weights, data_point):
    p = data_point[0]*weights[0] + data_point[1]*weights[1] + data_point[2]*weights[2]\
        + data_point[3]*weights[3]
    activation_output = 1 / (1 + math.e**(-p))  # Sigmoid activation function
    return activation_output


def fp_output(weights, data_point):
    n5 = forward_prop(weights[0], data_point)
    n6 = forward_prop(weights[1], data_point)
    n7 = forward_prop(weights[2], data_point)
    n8 = forward_prop(weights[3], data_point)
    hidden_data_point = [n5, n6, n7, n8]
    n9 = forward_prop(weights[4], hidden_data_point)
    n10 = forward_prop(weights[5], hidden_data_point)
    n11 = forward_prop(weights[6], hidden_data_point)
    return n5, n6, n7, n8, n9, n10, n11


def get_weights():
    w = [[random.random(), random.random(), random.random(), random.random()],
         [random.random(), random.random(), random.random(), random.random()],
         [random.random(), random.random(), random.random(), random.random()],
         [random.random(), random.random(), random.random(), random.random()],
         [random.random(), random.random(), random.random(), random.random()],
         [random.random(), random.random(), random.random(), random.random()],
         [random.random(), random.random(), random.random(), random.random()]
         ]
    return w


def back_prop(output_5, output_6, output_7, output_8, output_9, output_10, output_11, data_point, weights):
    learning_rate = 0.1
    e11 = output_11 * (1 - output_11) * (data_point[4][2] - output_11)
    e10 = output_10 * (1 - output_10) * (data_point[4][1] - output_10)
    e9 = output_9 * (1 - output_9) * (data_point[4][0] - output_9)

    e8 = output_8 * (1 - output_8) * (weights[4][3] * e9 + weights[5][3] * e10 + weights[6][3] * e11)
    e7 = output_7 * (1 - output_7) * (weights[4][2] * e9 + weights[5][2] * e10 + weights[6][2] * e11)
    e6 = output_6 * (1 - output_6) * (weights[4][1] * e9 + weights[5][1] * e10 + weights[6][1] * e11)
    e5 = output_5 * (1 - output_5) * (weights[4][0] * e9 + weights[5][0] * e10 + weights[6][0] * e11)
    #  calculating delta w for output nuerons
    d_w59 = learning_rate * output_5 * e9
    d_w69 = learning_rate * output_6 * e9
    d_w79 = learning_rate * output_7 * e9
    d_w89 = learning_rate * output_8 * e9

    d_w510 = learning_rate * output_5 * e10
    d_w610 = learning_rate * output_6 * e10
    d_w710 = learning_rate * output_7 * e10
    d_w810 = learning_rate * output_8 * e10

    d_w511 = learning_rate * output_5 * e11
    d_w611 = learning_rate * output_6 * e11
    d_w711 = learning_rate * output_7 * e11
    d_w811 = learning_rate * output_8 * e11

    #  calculating delta w for hidden layer
    d_w15 = learning_rate * data_point[0] * e5
    d_w25 = learning_rate * data_point[1] * e5
    d_w35 = learning_rate * data_point[2] * e5
    d_w45 = learning_rate * data_point[3] * e5

    d_w16 = learning_rate * data_point[0] * e6
    d_w26 = learning_rate * data_point[1] * e6
    d_w36 = learning_rate * data_point[2] * e6
    d_w46 = learning_rate * data_point[3] * e6

    d_w17 = learning_rate * data_point[0] * e7
    d_w27 = learning_rate * data_point[1] * e7
    d_w37 = learning_rate * data_point[2] * e7
    d_w47 = learning_rate * data_point[3] * e7

    d_w18 = learning_rate * data_point[0] * e8
    d_w28 = learning_rate * data_point[1] * e8
    d_w38 = learning_rate * data_point[2] * e8
    d_w48 = learning_rate * data_point[3] * e8

    weight_deltas = [[d_w15, d_w25, d_w35, d_w45],
                     [d_w16, d_w26, d_w36, d_w46],
                     [d_w17, d_w27, d_w37, d_w47],
                     [d_w18, d_w28, d_w38, d_w48],
                     [d_w59, d_w69, d_w79, d_w89],
                     [d_w510, d_w610, d_w710, d_w810],
                     [d_w511, d_w611, d_w711, d_w811]
                     ]
    return weight_deltas


def update_weights(change_in_weights, weights):
    for row in range(7):
        for col in range(4):
            weights[row][col] = weights[row][col] + change_in_weights[row][col]
    return weights


def train_model(training_data):
    weights = get_weights()
    count = 1
    xs = []
    ys = []
    for epoch in range(100):
        t_rmse = 0
        for sample in training_data:
            n5_out, n6_out, n7_out, n8_out, n9_out, n10_out, n11_out = fp_output(weights, sample)
            delta_weights = back_prop(n5_out, n6_out, n7_out, n8_out, n9_out, n10_out, n11_out, sample, weights)
            weights = update_weights(delta_weights, weights)
            rmse = math.sqrt(((n9_out - sample[4][0]) ** 2 + (n10_out - sample[4][1]) ** 2 + (n11_out - sample[4][2]) **
                              2) / 3)
            t_rmse = t_rmse + rmse
        t_rmse_avg = t_rmse / len(training_data)
        xs.append(count)
        ys.append(t_rmse_avg)
        count = count + 1
    plt.plot(xs, ys)
    plt.title('Training Error')
    plt.show()
    return weights


def test_model(test_data, trained_weights):
    t_rmse = 0
    for sample in test_data:
        n5_out, n6_out, n7_out, n8_out, n9_out, n10_out, n11_out = fp_output(trained_weights, sample)
        rmse = math.sqrt(
            ((n9_out - sample[4][0]) ** 2 + (n10_out - sample[4][1]) ** 2 + (n11_out - sample[4][2]) ** 2) / 3)
        t_rmse = t_rmse + rmse
        print([n9_out, n10_out, n11_out])
        if n9_out > 0.5:
            n9_out = 1
        else:
            n9_out = 0
        if n10_out > 0.5:
            n10_out = 1
        else:
            n10_out = 0
        if n11_out > 0.5:
            n11_out = 1
        else:
            n11_out = 0
        final_output = [n9_out, n10_out, n11_out]

        print(final_output)
        print(sample[4])
        print('--------SPACER--------')
        # print(t)

    t_rmse_avg = t_rmse / len(test_data)
    print('t_rmse average:' + str(t_rmse_avg))


data = normalize(data)
train, test = train_test_sets(data)
new_weights = train_model(train)
test_model(test, new_weights)
