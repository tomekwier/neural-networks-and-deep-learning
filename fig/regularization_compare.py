"""regularization_types
~~~~~~~~~~~~

Plot graphs to illustrate the performance of MNIST when different
regularization types (specifically L1 and L2 regularization)

"""

# Standard library
import json
import random
import sys

# My library
sys.path.append('../src/')
import mnist_loader
import network2

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

TRAINING_SIZE = 10000
NUM_EPOCHS = 1500000 // TRAINING_SIZE


def main():
    # run_networks(TRAINING_SIZE, NUM_EPOCHS)  
    make_plots(TRAINING_SIZE, NUM_EPOCHS)


def run_networks(training_size, num_epochs):
    run_network(training_size, num_epochs, network2.L1Regularization,
                "regularization_l1.json")
    run_network(training_size, num_epochs, network2.L2Regularization,
                "regularization_l2.json")


def run_network(training_size, num_epochs, regularization_type, file_name):
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost(),
                           regularization=regularization_type)

    print("\n\nTraining network with data set size %s" % (training_size))
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data[:training_size], num_epochs,
                  10, 0.5, lmbda=training_size*0.0001,
                  evaluation_data=test_data,
                  monitor_evaluation_cost=True,
                  monitor_evaluation_accuracy=True,
                  monitor_training_cost=True,
                  monitor_training_accuracy=True)

    accuracy = net.accuracy(validation_data) / 100.0
    print("Accuracy on validation data was %s percent" % accuracy)

    f = open(file_name, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()


def make_plots(training_size, num_epochs):
    make_a_plot("regularization_l1.json", num_epochs,
              training_set_size=training_size)
    make_a_plot("regularization_l2.json", num_epochs,
                training_set_size=training_size)


def make_a_plot(filename, num_epochs,
                training_cost_xmin=0,
                test_accuracy_xmin=0,
                test_cost_xmin=0,
                training_accuracy_xmin=0,
                training_set_size=1000):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
    plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs,
                           training_accuracy_xmin, training_set_size)
    plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size)


def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs),
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()


def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs),
            [accuracy/100.0
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()


def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs),
            test_cost[test_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()


def plot_training_accuracy(training_accuracy, num_epochs,
                           training_accuracy_xmin, training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs),
            [accuracy*100.0/training_set_size
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()


def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy/100.0 for accuracy in test_accuracy],
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs),
            [accuracy*100.0/training_set_size
             for accuracy in training_accuracy],
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
