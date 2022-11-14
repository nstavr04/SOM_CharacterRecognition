# Implementation of a Kohonens Self Organizing Map
# Author: Nikolas Stavrou

# Clustering of the handwritten character vectors into the 26 letters of the alphabet

import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd

from numpy import log

# Self Organizing Map
class SOM:

    def __init__(self, v_size, grid_size, lr, epochs):

        # Initializing values
        self.initial_lr = lr
        self.lr = self.initial_lr
        self.dimentions = v_size

        self.epochs = epochs
        self.grid_size = grid_size

        self.initial_neigh_size = grid_size / 2
        self.neigh_size = self.initial_neigh_size

        self.J = (self.epochs / log(self.initial_neigh_size))

        # Create the map
        # Each cell is a vector of size 16 with random values between 0.0 and 1.0

        grid_weight_dict = {}

        for x in range(grid_size * grid_size):
            grid = np.random.rand(v_size)
            grid_weight_dict[x] = grid

        self.grid_weight_dict = grid_weight_dict

        # Create a mapping of neuron number to a cord on grid
        cords_map = {}
        cnt = 0
        for y in range(grid_size):
            for x in range(grid_size):
                cords_map[cnt] = (y,x)
                cnt += 1

        self.cords_map = cords_map          
    
    def train_testing_calls(self, train_i, test_i, filename):

        error_file_output = ""

        for i in range(self.epochs):

            print("Epoch: %2d" % (i+1))
            train_error = self.train(train_i)
            test_error = self.test(test_i)

            self.update_lr(i)
            self.update_neigh_size(i)

            # Concatenate results
            error_file_output += str(i) + " " + str(train_error) + " " + str(test_error) + "\n"

         # Write into errors.txt file
        error_file = filename.joinpath("results.txt")
        with open(error_file, "w") as file:
            file.write(error_file_output)   

    def train(self, train_inputs):

        sum_error_train = 0

        for input_v in train_inputs:

            # Go through all neurons and find winner neuron and its distance from input vector
            winner_neuron_num, min_dist = self.winner_neuron(input_v)
            # Update weights of each neuron
            for i in range(self.grid_size * self.grid_size): 
                # Calculate neigh function
                h = self.neigh_function(winner_neuron_num,i)
                for z in range(self.dimentions):
                    self.grid_weight_dict[i][z] = self.grid_weight_dict[i][z] + (self.lr * (input_v[z] - self.grid_weight_dict[i][z]) * h)

            # Calculate error
            sum_error_train += min_dist

        return ((sum_error_train ** 2) / len(train_inputs))

    def test(self, test_inputs):

        sum_error_test = 0

        for input_v in test_inputs:

            # Go through all neurons and find winner neuron and its distance from input vector
            winner_neuron_num, min_dist = self.winner_neuron(input_v)

            # Calculate error
            sum_error_test += min_dist

        return (sum_error_test ** 2 / len(test_inputs))

    # Return winner neuron and its distance from input vector
    def winner_neuron(self, input_v):
        
        # Get the distance of input vector to all neurons
        # Find the smallest one
        min_distance = sys.maxsize
        winner_neuron = -1
        for i in range(self.grid_size * self.grid_size):
            dist = self.euc_dist(input_v, self.grid_weight_dict[i])
            if (dist < min_distance):
                min_distance = dist
                winner_neuron = i

        return winner_neuron, min_distance
    
    # Return euclidean distance between 2 vectors
    def euc_dist(self, v1, v2):
        dist = 0;    
        for i in range(self.dimentions):
            dist = dist + ((v1[i] - v2[i]) * (v1[i] - v2[i]))

        return dist

    # Return distance between 2 neurons
    def euc_dist_cord(self, x1, y1, x2, y2):
        return (((x1 - x2) * (x1 - x2)) + ((y1 - y2) *(y1 - y2)))  

    # Return the value of the neighborhood function
    def neigh_function(self, winner_n, other_n):
        return math.exp(- self.euc_dist_cord(self.cords_map[winner_n][1], self.cords_map[winner_n][0], self.cords_map[other_n][1], self.cords_map[other_n][0]) / 2*self.neigh_size**2)

    # Update the neighborhood size
    def update_neigh_size(self, curr_epoch):
        self.neigh_size = self.initial_neigh_size * math.exp(- curr_epoch / self.J)

    # Update learning rate (gain)
    def update_lr(self, curr_epoch): 
        self.lr = self.initial_lr * math.exp( - curr_epoch / self.epochs)

    # Function used in the end to create the clustering.txt file, labeling each neuron
    def labeling(self, test_inputs,test_outputs, filename):
        
        cluster_file_output = ""
        cnt = 1

        grid_print_end = ""

        # For each neuron
        for j in range(self.grid_size * self.grid_size):

            min_distance = sys.maxsize
            label = ""

            # Go through all input vectors and find the one that has min distance from our neuron
            for i, input in enumerate(test_inputs):
                
                dist = self.euc_dist(input, self.grid_weight_dict[j])
                # Apply that input vectors label to our neuron
                if (dist < min_distance):
                    min_distance = dist
                    label = test_outputs.iloc[[i]]

            cluster_file_output += "Neuron [" + str(self.cords_map[j][0]) + "][" + str(self.cords_map[j][1]) + "] is labeled as letter: " + label.to_string(header=False,index=False,index_names=False) + "\n"
            
            # Used for printing in the end a better representation of the labels as a 2D grid
            if cnt == self.grid_size:
                grid_print_end += "\n"
                cnt = 1
            else:
                grid_print_end += label.to_string(header=False,index=False,index_names=False) + " "
                cnt += 1

        cluster_file_output += "\n" + grid_print_end

    # Write into errors.txt file
        cluster_file = filename.joinpath("clustering.txt")
        with open(cluster_file, "w") as file:
            file.write(cluster_file_output)

# Class for plotting our gathered error and success rate data
class Plots:

    def error_plot(filename):

        errors_filename = filename.joinpath("results.txt")

        # Plot for error file
        epochs = np.loadtxt(errors_filename, usecols = (0))
        training_error = np.loadtxt(errors_filename, usecols = (1))
        test_error = np.loadtxt(errors_filename, usecols = (2))

        plt.title("Error File")
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.plot(epochs, training_error, color = 'r', label= 'Training Error')
        plt.plot(epochs, test_error, color = 'b', label= 'Test Error')
        plt.legend()

        plt.show()

def main():

    # Set Parameters

    # 16 values per vector
    dimentions = 16
    grid_size = 20
    learning_rate = 0.5
    epochs = 200

    # The path used for finding the files. All files need to be in the same directory
    filename = pathlib.Path(__file__).parent

    # Obtaining training inputs and outputs
    training_filename = filename.joinpath("training_mini.txt")
    train_inputs = np.loadtxt(training_filename, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), dtype=np.float64)

    # Obtaining testing inputs and outputs
    test_filename = filename.joinpath("test_mini.txt")
    test_inputs = np.loadtxt(test_filename, usecols= (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), dtype=np.float64)
    test_outputs_target = pd.read_csv(test_filename, usecols = [0], sep=" ", header=None)

    # Initialize SOM
    self_org_map = SOM(dimentions, grid_size, learning_rate, epochs)

    self_org_map.train_testing_calls(train_inputs, test_inputs, filename)
    self_org_map.labeling(test_inputs,test_outputs_target, filename)

    # Plots
    Plots.error_plot(filename)

if __name__ == '__main__':
    main()