import numpy as np
import matplotlib.pyplot as plt
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score


class Perceptron(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, threshold=20, learning_rate=0.001):
        self.no_inputs = no_inputs
        #self.weights = np.zeros(no_inputs) / no_inputs #Changed .ones to .zeros
        #self.weights = np.zeros(no_inputs + 1)
        self.ranom_state = np.random.RandomState(42)
        self.weights = self.ranom_state.normal(loc=0.0, scale=0.01,
                                               size=no_inputs)
        self.threshold = threshold
        self.learning_rate = learning_rate

    

    #=======================================#
    # Prints the details of the perceptron. #
    #=======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Threshold:\t" + str(self.threshold)) #Changed max_iterations to threshold
        print("Learning rate:\t" + str(self.learning_rate))
        
    def print_pixel(self):    
        pixels = self.weights[1:]
        pixels = np.array(pixels, dtype = 'uint8')
        pixels = pixels.reshape((28, 28))
        plt.imshow(pixels, cmap = 'gray')
        plt.show()
        

    #=========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    #=========================================#
    def predict(self, inputs):
        activation = np.dot(inputs, self.weights) #Activation function
        if activation > 0:
            return 1
        else:
            return 0
        
    def predict_sigmoid(self, inputs): #PART 4
        activation = np.dot(inputs, self.weights)
        sigmoid = 1/(1 + np.exp(-activation))
        return sigmoid
    
    #======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    #======================================#
    def train(self, training_data, labels): #Orignial train
        assert len(training_data) == len(labels)
        
        for i in range(self.threshold):
            for inputs, label in zip(training_data, labels): #changed data to inputs
                predictions = self.predict(inputs)
                self.weights = self.weights + self.learning_rate*(label - predictions)*inputs #? this may be incorrect
        return
    
    def batch_train(self, training_data, labels): #PART 2-Implementing bacth training
        assert len(training_data) == len(labels)
        
        for i in range(self.threshold):
            weight_update = np.zeros(785)
            for inputs, label in zip(training_data, labels): #changed data to inputs
                predictions = self.predict(inputs)
                weight_update += self.learning_rate*(label - predictions)*inputs
                self.weights += weight_update/len(training_data)
        return
    
    def train_sigmoid(self, training_data, labels): #PART 3&4
        assert len(training_data) == len(labels)

        for i in range(self.threshold):
            for inputs, label in zip(training_data, labels): #changed data to inputs
                predictions = self.predict_sigmoid(inputs)
                update = self.learning_rate * (label - predictions) * inputs
                update = np.asarray(update)
                self.weights += update * inputs
                #self.weights[1:] += update * inputs #? this may be incorrect
                #self.weights[0] += update[0]
    
    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy of the perceptron. #
    #=========================================#
    def test(self, testing_data, labels):
        assert len(testing_data) == len(labels)
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0
#Manually calculated score measures as built in methods and others were showing several errors
        for inputs, labels in zip(testing_data, labels):
            prediction = self.predict(inputs)
            if labels == 1 and prediction == 1:
                TP = TP + 1
            if labels == 0 and prediction == 0:
                TN = TN + 1
            if labels == 0 and prediction == 1:
                FP = FP + 1
            if labels == 1 and prediction == 0:
                FN = FN + 1
        
        # accuracy: (tp + tn) / (p + n)
        accuracy = ((TP + TN) / (TP + FP + TN + FN))#*100
        #print("Accuracy:\t"+str(accuracy + "%"))
        print("Accuracy: %.2f%%" % (accuracy*100))
        
        # precision: tp / (tp + fp)
        precision = TP / (TP + FP)
        #print("Precision:\t"+str(precision))
        print("Precision: %.2f%%" % (precision*100))
        
        # recall: tp / (tp + fn)
        recall = TP / (TP + FN)
        #print("Recall:\t"+str(recall))
        print("Recall: %.2f%%" % (recall*100))

    def test_sigmoid(self, testing_data, labels): #PART 4
        assert len(testing_data) == len(labels)
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0
#Manually calculated score measures as built in methods and others were showing several errors
        for inputs, labels in zip(testing_data, labels):
            prediction = self.predict(inputs)
            if labels >= 0.5 and prediction >= 0.5:
                TP = TP + 1
            if labels < 0.5 and prediction < 0.5:
                TN = TN + 1
            if labels < 0.5 and prediction >= 0.5:
                FP = FP + 1
            if labels >= 0.5 and prediction < 0.5:
                FN = FN + 1
        # accuracy: (tp + tn) / (p + n)
        accuracy = ((TP + TN) / (TP + FP + TN + FN))#*100
        #print("Accuracy:\t"+str(accuracy + "%"))
        print("Accuracy: %.2f%%" % (accuracy*100))
        
        # precision: tp / (tp + fp)
        precision = TP / (TP + FP)
        #print("Precision:\t"+str(precision))
        print("Precision: %.2f%%" % (precision*100))
        
        # recall: tp / (tp + fn)
        recall = TP / (TP + FN)
        #print("Recall:\t"+str(recall))
        print("Recall: %.2f%%" % (recall*100))