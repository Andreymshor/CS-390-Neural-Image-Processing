import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_CLASSES = 10
IMAGE_SIZE = 784
ALGORITHM = "custom_net"


class NeuralNetwork():
    def __init__(self, data, inputSize, outputSize, neuronsPerLayer, numLayers, learningRate = 0.1):
        self.inputs = np.array(data[0]) # will be a array of arrays
        self.labels = np.array(data[1]) # will be a array of arrays
        self.inputSize = inputSize # used for input layer size
        self.outputSize = outputSize # used for out put layer size
        self.neuronsPerLayer = neuronsPerLayer # used for number of neurons in each hidden layer 
        self.lr = learningRate # used to control the learning rate
        self.numLayers = numLayers # used to determine number of hidden layers
        self.weights = [] # used to initialize each layer with weights
        self.model = []


        inputLayerWeights = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.weights.append(inputLayerWeights)

        # creates weights for n - 1 layers
        for _ in range(1, numLayers - 1):
            hiddenLayer = np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer)
            self.weights.append(hiddenLayer)
        
        outputlayerWeights = np.random.randn(self.neuronsPerLayer, self.outputSize)
        self.weights.append(outputlayerWeights)
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __sigmoidDerivative(self, x):
        return x * (1 - x) # because already activated
    

    def __forward(self, X): # at each layer, we dot product input and weights, and pass it into the activation function.
        # X is a batch of examples
        batchOutputList = []
        for example in X:

            layerOutputList = []
            currInput = example # (1 x n) matrix
            for weight in self.weights: # weight is an n x m matrix
                currLayer = np.dot(currInput, weight) 
                activationList = [self.__sigmoid(elem) for elem in currLayer]
                layerOutputList.append(activationList)
                currInput = activationList
            batchOutputList.append(layerOutputList)
        return batchOutputList 

    def predict(self):
        prediction = self.__forward()[len(self.__forward()) - 1] # array of 1 by 10

        # predict the actual classifed value
        return prediction

    

    def __avgloss(self, predictedValues, actualValues):
        # predictedValues: Will be an array of arrays
        # actualValues: Will be an array of arrays
        # cost function: (predicted - actual)^2
        # cost function derivative: 2 * (predicted - actual)
        # returns the avg loss and the avg loss derivative
        cost = 0
        costDer = 0
        for i in range(len(predictedValues)):
            currPrediction = predictedValues[i][len(predictedValues[i]) - 1] # ith array in predicted
            currActual = actualValues[i] # ith array in actual
            for j in range(len(currPrediction)):
                cost += (actualValues[i] - predictedValues[i])**2
                costDer += 2 * (actualValues[i] - predictedValues[i]) # add to weights
        return (cost / len(predictedValues) , costDer / len(predictedValues))


        

    def __backPropogation(self, batchOutputList, Y):
        weights = self.weights
        # predictedValues = layerOutputList[len(layerOutputList) - 1]
        # LCost, LCostDer = self.__avgloss(predictedValues, Y)
        # for layer in range(len(layerOutputList) - 1, -1, -1):
        #     currLayer = layerOutputList[layer]
        #     sigDelt = [self.__sigmoidDerivative(elem) for elem in currLayer]

             
        
        pass

    # brains behind training neural network. Will use a loop to go through the examples
    # provided in the input data and labels, with or without batches, along w/ a size 
    # for the minibatches.
    
    def train(self, epochs = 100000, minibatches = True, mbs = 100):
        for i in epochs:
           batchX, batchY = self.__batchGenerator(self.inputs, self.labels, mbs)
           batchOutputList = self.__forward(batchX)
           self.__backPropogation(batchOutputList, batchY)
        
        return self.weights
    
    
    # idea from https://datascience.stackexchange.com/questions/47623/how-feed-a-numpy-array-in-batches-in-keras
    def __batchGenerator(self, X, Y, batch_size = 1):
        indices = np.arange(len(X))
        batch = []
        while len(batch) != batch_size:
            np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
            
        return X[batch], Y[batch]

    



    #------------------------- TESTER METHODS ---------------------------#

    # tester method for making sure I printed the right number of weights
    def printWeights(self): 
        # test back home
        print(f"Length of weights array {len(self.weights)}")
        # print(type(self.weights))
        for array in self.weights:
            print(f"array: {array.shape}")
    
    # tester method to make sure my forward list looks right
    def printForwardList(self):
        print("Output of each Layer is:")
        layerOutputList = np.array(self.__forward(self.inputs))
        for k, array in enumerate(layerOutputList):
            print(f"Layer {k + 1}: {array.shape}")
        
        print(f"length of layer list: {len(layerOutputList)}")

    # tester method to make sure my multiplication isnt goofy for Forward
    def checkForwardMultiplication(self):
        currInput = self.inputs # (1 x n) matrix
        for i,weight in enumerate(self.weights): # weight is an n x m matrix
            print(f"Current Weight used for layer {i + 1}: {weight}")
            print(f"Current Input used for layer {i + 1} is: {currInput}")
            currLayer = np.dot(currInput, weight) 
            activationList = []
            for elem in currLayer:
                activationList.append(self.__sigmoid(elem))
            currInput = activationList



# Classifer that just guesses the class Label
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


#-----------------------------PIPELINE FUNCTIONS!------------------------------#

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = np.true_divide(xTrain, 255.0)
    xTest = np.true_divide(xTest, 255.0)

    finalXTrain = []
    for matrix in xTrain:
        finalXTrain.append(np.reshape(matrix, (matrix.shape[0] * matrix.shape[1])))
    finalXTrain = np.array(finalXTrain)
    
    finalXTest = []
    for matrix in xTest:
        finalXTest.append(np.reshape(matrix, (matrix.shape[0] * matrix.shape[1])))
    finalXTest = np.array(finalXTest)
    yTrainP = to_categorical(yTrain, NUM_CLASSES) # converts to an array between 0 and 9. The ith index represents the number. A 1 represents the actual number.

    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(finalXTrain.shape))
    print("New shape of xTest dataset: %s." % str(finalXTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((finalXTrain, yTrainP), (finalXTest, yTestP))




def main():

    ((xTrain, yTrainP), (xTest, yTestP)) = preprocessData(getRawData())
    #matrix1 = xTrain[0]
    #print(type(matrix1))
    #matrix2 = yTrainP[0]
    #print(matrix2)
    print("TESTER 1")
    tester1 = NeuralNetwork([xTrain, yTrainP], 784, 10, 16, 2)
    # tester1.train()
    tester1.printWeights()
    tester1.printForwardList()
    


if __name__ == '__main__':
    main()