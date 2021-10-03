
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
#from sklearn.metrics import confusion_matrix
import random
from tqdm import tqdm

# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return x * (1 - x) # because already activated

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        pbar = tqdm(total=epochs)
        batchesOfX = self.__batchGenerator(xVals, mbs)
        batchesOfY = self.__batchGenerator(yVals, mbs)

        for i in range(epochs):
            
            for batchX, batchY in zip(batchesOfX, batchesOfY):
                L1Out, L2Out = self.__forward(batchX)
                L2e = L2Out - batchY
                L2d = L2e * self.__sigmoidDerivative(L2Out)
                L1e = np.dot(L2d, np.transpose(self.W2))
                L1d = L1e * self.__sigmoidDerivative(L1Out)
                L1a = (np.dot(np.transpose(batchX), L1d)) * self.lr
                L2a = (np.dot(np.transpose(L1Out), L2d)) * self.lr
                
                self.W1 -= L1a
                self.W2 -= L2a
            
            
            pbar.update(1)
        
        
        
    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2

    def evalModel(self, xVals, yVals):
        accuracy = 0
        preds = []
        actuals = []
        for i, example in enumerate(xVals):
            predArr = list(self.predict(example))
            pred = predArr.index(max(predArr))
            actualArr = list(yVals[i])
            actual = actualArr.index(max(actualArr))
            if pred == actual: 
                accuracy += 1
            preds.append(pred)
            actuals.append(actual)
            
        print(f"ACCURACY: {accuracy / len(xVals) * 100}%")
        confusionMatrix = tf.math.confusion_matrix(actuals, preds, num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None)
        print(f"Confusion Matrix:\n{confusionMatrix}")
        return accuracy / len(xVals)
    
    def getWeights(self):
        print(f"Weights for layer 1: {self.W1}")
        print(f"Weights for layer 2: {self.W2}")



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

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
    xTrain = xTrain / 255
    xTest = xTest / 255
    xTrain = xTrain.reshape(60000, IMAGE_SIZE)
    xTest = xTest.reshape(10000, IMAGE_SIZE)
    yTrainP = to_categorical(yTrain, NUM_CLASSES) # converts to an array between 0 and 9. The ith index represents the number. A 1 represents the actual number.

    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))

# For tensorflow: https://www.tensorflow.org/tutorials/keras/classification
def trainModel(xTrain, yTrain):
    xTrain, yTrain = xTrain, yTrain
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and Training custom_net model:")
        NeuralNetwork = NeuralNetwork_2Layer(784, 10, 16)
        NeuralNetwork.train(xTrain, yTrain, mbs=25)
        return NeuralNetwork
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(784, activation='relu'), tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)])
        optimizer = tf.keras.optimizers.Adam(0.001)
        loss = tf.keras.losses.categorical_crossentropy
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=10)
        return model
        
    else:
        raise ValueError("Algorithm not recognized.")


# for confusion matrix: https://www.tensorflow.org/api_docs/python/tf/math/confusion_matrix
def runModel(xTest, yTest, model):
    if ALGORITHM == "guesser":
        print("predicting and evaluating guesser results")
        ans = guesserClassifier(xTest)
        evalResults((xTest, yTest), ans)
    elif ALGORITHM == "custom_net":
        print("predicting and evaluating custom_net results:")
        return model.evalModel(xTest, yTest)
    elif ALGORITHM == "tf_net":
        print("predicting and evaluating TF_NN.")
        pred = model.predict(xTest)
        print(pred.shape)
        print(yTest.shape)
        test_loss, test_acc = model.evaluate(xTest, yTest, verbose=2)
        confusionMatrix = tf.math.confusion_matrix(tf.argmax(yTest,1), tf.argmax(pred,1), num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None)
        #confusionMatrix = confusion_matrix(np.argmax(yTest, axis=1), np.argmax(pred, axis=1))
        print(f'\nTest accuracy:{test_acc}')
        print(f'\nTest loss: {test_loss}')
        print(f"\nConfusion matrix:\n {confusionMatrix}")

        
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]

    confusionMatrix = tf.math.confusion_matrix(tf.argmax(yTest,1), tf.argmax(preds,1), num_classes=None, weights=None, dtype=tf.dtypes.int32, name=None)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print(f"\nConfusion matrix:\n {confusionMatrix}")



#=========================<Main>================================================

def main():
    raw = getRawData()
    ((xTrain, yTrainP), (xTest, yTestP)) = preprocessData(raw)
    model = trainModel(xTrain, yTrainP)
    preds = runModel(xTest, yTestP, model)



if __name__ == '__main__':
    main()