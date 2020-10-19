# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:57:02 2020 

@author: lewis 
"""
import numpy as np
from perceptron_template1 import *
import sys
from time import sleep

#Target digit
target = 7

#Node creation
p = Perceptron(28*28+1)   
p.print_details()

print("Loading data...")
image_pixels = 28*28
data_path = "./"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter = ",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter = ",")
print("done,")

print ("Training...")
training_data = [np.append([1], d[1:]) for d in train_data] #changed d[1:1] to d[1:] as 1:1 is only taking in one variable
labels = [d[0] == target for d in train_data]
p.train(training_data, labels) #Original/PART 1
#p.batch_train(training_data, labels) #PART 2
#p.train_sigmoid(training_data, labels) #PART 4
print("done...")

print ("Testing...")
testing_data = [ np.append([1], d[1:]) for d in test_data]
labels = [d[0] == target for d in test_data]
p.test(testing_data, labels) #Original/PART 1
#p.test_sigmoid(testing_data, labels) #Sigmoid Test/PART 4
print("done...")

# =============================================================================
# PART 3
# Commentted out part 3 to test parts 1/2/4/5 properly
# Part 3 does work (I think) so just uncomment to test it!
# =============================================================================

#actual_train_labels = train_data[:, :1]
#actual_test_labels = test_data[:, :1]
#nodes = []
#for i in range(10): #PART 3
#    labels = [d[0] == i for d in train_data]
#    p = Perceptron(28*28+1)
#    p.train_sigmoid(training_data, labels)
#    nodes.append(p)
#  
#for ind, d in enumerate(testing_data):
#    pred_list = []
#    for i in range(len(nodes)):
#        pred_list.append(nodes[i].predict_sigmoid(d))
#  
#    print(pred_list)
#    num = pred_list.index(max(pred_list))
#    actual_label = actual_train_labels[ind]
#  
#    print('\n-----------\n')
#    print(f"Prediction: {num}")
#    print(f"Actual Label: {actual_label}")
#    print('\n------------\n')
#     
# END PART 3 
# =============================================================================



#PART 5
p.print_pixel() 

printcharacters = True 
if printcharacters:
    for d in test_data:
        prediction = p.predict(np.append([1], d[1:])) #Original/PART 1
        #prediction = p.predict_sigmoid(np.append([1], d[1:])) #PART 4
        dat = d[1:].reshape((28, 28))
        for k in range(28):
            for j in range(28):
                if dat[k][j] > 0:
                    sys.stdout.write("#")
                else:
                    sys.stdout.write(",")
                sys.stdout.flush()
            sys.stdout.write("\n")
        
        if prediction:
            print(" ======= FOUND IT ======= ")
    sleep(1, 0)
    sys('clear')