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
    def __init__(self, inputs, outputSize, neuronsPerLayer, numLayers, learningRate = 0.1):
        self.inputs = np.array(inputs)
        self.inputSize = len(inputs)
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer 
        self.lr = learningRate
        self.numLayers = numLayers
        self.weights = []

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
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    

    def __forward(self): # at each layer, we dot product input and weights, and pass it into the activation function.
        layerOutputList = []
        currInput = self.inputs # (1 x n) matrix
        for weight in self.weights: # weight is an n x m matrix
            currLayer = np.dot(currInput, weight) 
            activationList = []
            for elem in currLayer:
                activationList.append(self.__sigmoid(elem))
            layerOutputList.append(activationList)
            currInput = activationList
            
        return layerOutputList

    def predict(self):
        prediction = self.__forward()[len(self.__forward()) - 1]
        print(prediction)
        return prediction


    def __loss(self):
        pass

    def __backPropogation(self):
        pass

    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]
        pass
    



    #------------------------- TESTER METHODS ---------------------------#

    # tester method for making sure I printed the right number of weights
    def printWeights(self): 
        # test back home
        print(f"Length of weights array {len(self.weights)}")
        # print(type(self.weights))
        for array in self.weights:
            print(f"array: {array}")
    
    # tester method to make sure my forward list looks right
    def printForwardList(self):
        print("Output of each Layer is:")

        for k, array in enumerate(self.__forward()):
            print(f"Layer {k + 1}: {array}")

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
    yTrainP = to_categorical(yTrain, NUM_CLASSES) # converts to an array between 0 and 9. The ith index represents the number. A 1 represents the actual number.
    for elem in yTrainP:
        print(elem)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))




def main():

    preprocessData(getRawData())
    #print("TESTER 1")
    #tester = NeuralNetwork(2, 1, 3, 2)
    #tester.printWeights()
    #tester2 = NeuralNetwork([1.0,2.0],10,5,3)
    # tester2.printWeights()
    # tester2.checkForwardMultiplication()
    #tester2.predict()


if __name__ == '__main__':
    main()