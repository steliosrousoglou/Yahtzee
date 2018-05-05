import sys
import csv

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# if trouble with CUDA on shared system then
# export CUDA_VISIBLE_DEVICES=''
# to turn CUDA off

def main():
    x_all = []
    y_all = []

    # read the data
    reader = csv.reader(sys.stdin)
    c=0
    for row in reader:
        # split into inputs...
        x_all.append([float(x) for x in row[:4]])

        # ...and output, which is converted to three booleans
        if row[-1] == "Iris-setosa":
            y_all.append([1, 0, 0])
        elif row[-1] == "Iris-versicolor":
            y_all.append([0, 1, 0])
        else:
            y_all.append([0, 0, 1])
        c+=1
        if c>1000:
            break

    # normalize the data to range [0.0, 1.0]
    norm_low = 0.0
    norm_high = 1.0
    mins = [0.0] * 4
    maxes = [0.0] * 4
    for i in range(0, 4):
        mins[i] = min([x[i] for x in x_all])
        maxes[i] = max([x[i] for x in x_all])

    x_norm = [[(x[i] - mins[i]) / (maxes[i] - mins[i]) * (norm_high - norm_low) + norm_low for i in range(0, 4)] for x in x_all]
    print(x_all)
    # split into training data and test data
    test_size = int(len(x_norm) / 5)
    train_size = len(x_norm) - test_size

    x_train = np.matrix(x_norm[:train_size])
    y_train = np.matrix(y_all[:train_size])

    x_test = np.matrix(x_norm[train_size:])
    y_test = y_all[train_size:]

    # set the topology of the neural network
    model = Sequential()
    model.add(Dense(10, activation="relu", input_dim = x_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(y_train.shape[1], activation = "softmax"))

    # set up optimizer
    # lr = learning rate (amount to change weights by during each backprop)
    # decay = decay in learning rate (starts high, decreases)
    # momentum = resistance in change in direction of descent
    # categorical_crossentropy =~= measure of information in difference between
    # predicted and expected output (want low information)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.8, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    # train!
    model.fit(x_train, y_train, epochs=100, batch_size=50)

    # get predictions (one-hot encoded row for each test input)
    # convert to class 0, 1, 2 by finding index of maximum value;
    y_predict = [max(enumerate(y), key=lambda x:x[1])[0] for y in model.predict(x_test)]

    # do the same for test data
    y_correct = [max(enumerate(y), key=lambda x:x[1])[0] for y in y_test]

    # get list of (prediction, expectation) pairs, convert to list of
    # 1's where equal, 0's where unequal, and sum result to get number
    # of correct predictions
    print(sum((1 if y[0] == y[1] else 0) for y in zip(y_predict, y_correct)) / len(y_predict))

if __name__ == "__main__":
    main()
