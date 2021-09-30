import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from tqdm import tqdm

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
                activationList = list(self.__sigmoid(np.array(currLayer, dtype=np.longdouble)))
                layerOutputList.append(activationList)
                currInput = activationList
            batchOutputList.append(layerOutputList)
        return batchOutputList 

    def predict(self, testX, testY):
        predictions = self.__forward(testX)
        print(len(predictions))
        accuracy = 0
        for i, layerOutputList in enumerate(predictions):
            currentPredictionArr = layerOutputList[len(layerOutputList) - 1]
            #print(np.asarray(currentPredictionArr).shape)
            pred = currentPredictionArr.index(max(currentPredictionArr))
            actual = list(testY[i]).index(max(list(testY[i])))
            if pred == actual:
                accuracy += 1
            

        # predict the actual classifed value
        return accuracy / len(testY)

    

    def __avgloss(self, predictedValues, actualValues):
        # predictedValues: Will be an array of arrays
        # actualValues: Will be an array of arrays
        # cost function: (predicted - actual)^2
        # cost function derivative: 2 * (predicted - actual)
        # returns the avg loss and the avg loss derivative
        # print(f"Shape of predicted Values: {np.array(predictedValues).shape}")
        cost = 0
        costDer = 0
        for i in range(len(predictedValues)):
            currPrediction = predictedValues[i][len(predictedValues[i]) - 1] # ith array in predicted
            # print(f"currPrediction: {currPrediction}")
            currActual = actualValues[i] # ith array in actual
            # print(currActual)
            for j in range(len(currPrediction)):
                cost += (currActual[j] - currPrediction[j])**2
                costDer += 2 * (currActual[j] - currPrediction[j]) # add to weights
        return (cost / len(predictedValues) , costDer / len(predictedValues))


        

    def __backPropogation(self, X,batchOutputList,Y):
        #print("In backProp")
        weightRecs = []
        avgCost, avgCostDer = self.__avgloss(batchOutputList, Y)
        # for each example in batch outputlist, find the values of the last layer neurons,
        # those are predictions
        for i, example in enumerate(X):
            exCostDer = avgCostDer
            weights = self.weights
            adjustedWeight = []
            layerDelta = [] # layerDelta is in reverse order (starts from end and goes to forward), so make sure to reverse the list
            layerOutputList = batchOutputList[i]
            #for layer in range(self.numLayers, )
            for layer in range(len(layerOutputList) - 1, -1, -1):
                currLayer = layerOutputList[layer]
                # prevLayer = []
                if layer == 0:
                    dP = np.dot(example, self.weights[0])
                    delta = np.array([self.__sigmoidDerivative(elem) for elem in dP]) * exCostDer
                    layerDelta.append(delta)
                else: 
                    prevLayer = layerOutputList[layer - 1]
                    dotProductArr = np.dot(prevLayer, self.weights[layer])
                    currLayerDelta = np.array([self.__sigmoidDerivative(elem) for elem in dotProductArr]) * exCostDer
                    layerDelta.append(currLayerDelta)
                    exCostDer = np.dot(currLayerDelta, np.transpose(self.weights[layer]))
            

            # COMPUTE ADJUSTMENTS 
            # Reverse layerDelta list 
            layerDelta = layerDelta[::-1]
            adjustments = []
            
            layerZeroAdjustment = np.dot(example.reshape(len(example), 1), np.transpose(layerDelta[0].reshape(len(layerDelta[0]),1))) * self.lr
            adjustments.append(layerZeroAdjustment)
            for i in range(1, len(layerDelta)):
                arrayToAppend = np.dot(np.transpose(np.array(layerOutputList[i])), layerDelta[i]) * self.lr
                adjustments.append(arrayToAppend)

            for i in range(len(weights)):
                adjustedWeight.append(np.add(np.array(weights[i]), adjustments[i]))
            
            
            weightRecs.append(adjustedWeight)


        # Compute average adjustment for each layer and set that to be the new weights  
        averagedWeights = []
        
        for i in range(len(self.weights)):
            sum = np.zeros(self.weights[i].shape)
            for adjustedWeight in weightRecs:
                sum = np.add(sum, adjustedWeight[i])
            averagedWeights.append(np.true_divide(sum, len(weightRecs)))
        
        self.weights = averagedWeights



                

    # brains behind training neural network. Will use a loop to go through the examples
    # provided in the input data and labels, with or without batches, along w/ a size 
    # for the minibatches.
    
    def train(self, epochs = 100000, minibatches = True, mbs = 100):
        pbar = tqdm(total=epochs)
        for i in range(epochs):
           batchesOfX = [x for x in self.__batchGenerator(self.inputs, mbs)]
        #    print(np.array(batchesOfX).shape)
           batchesOfY = [y for y in self.__batchGenerator(self.labels, mbs)]
        #    print(np.array(batchesOfY).shape)
           #batchX = []
           #batchY = []
           if i < len(batchesOfX):
               batchX = batchesOfX[i]
               batchY = batchesOfY[i] 
           else:
               i = 0
           
           # print(batchX.shape)
        #    print(batchY.shape)
           
           batchOutputList = self.__forward(batchX)
           self.__backPropogation(batchX, batchOutputList, batchY)
           pbar.update(1)
        
        return self.weights
    
    
    # idea from https://datascience.stackexchange.com/questions/47623/how-feed-a-numpy-array-in-batches-in-keras
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    



    #------------------------- TESTER METHODS ---------------------------#
    def printOnlyWeights(self):
        for weight in self.weights:
            print(weight)

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
        for k, array in enumerate(layerOutputList[0]):
            print(f"Layer {k + 1}: {np.asarray(array).shape}")
            print(f"{np.asarray(array)}")
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
    print("ORIGINAL WEIGHTS:")
    tester1.printOnlyWeights()
    model = tester1.train(epochs=1000)
    print("MODEL AFTER TRAINING:")
    for weight in model: 
        print(weight)
    print(tester1.predict(xTest,yTestP))
    # tester1.printWeights()
    # tester1.printForwardList()
    


if __name__ == '__main__':
    main()