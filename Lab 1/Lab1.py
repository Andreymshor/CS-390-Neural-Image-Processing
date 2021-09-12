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
    def __init__(self, inputSize, outputSize, neuronsPerLayer, numLayers, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.numLayers = numLayers
        self.weights = []
    

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __sigmoidDerivative(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    

    def __batchGenerator(self, l, n):
        # for i in range(0, len(l), n):
        #     yield l[i : i + n]
        pass
    






# Classifer that just guesses the class Label
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



def main():
    print("In here")


if __name__ == '__main__':
    main()