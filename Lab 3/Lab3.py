

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.externals._pilutil import imsave
import warnings
tf.compat.v1.disable_eager_execution()

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "cute_cats.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "starry_night.jpg"             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    convertedImg = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    return convertedImg


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    imgH, imgW, numFilters = style.shape
    return K.sum(K.square(gramMatrix(gen) - gramMatrix(style))) / (4. * (numFilters**2) * ((imgH * imgW)**2))   #TODO: implement.


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    return x





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = np.reshape(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData, dtype=tf.float64)
    styleTensor = K.variable(sData, dtype=tf.float64)
    gFlat = K.placeholder(CONTENT_IMG_H * CONTENT_IMG_W *3, dtype=tf.float64)
    genTensor = K.reshape(gFlat, (1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)   #TODO: implement.
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])

    print("   VGG19 model loaded.")
    totalLoss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    totalLoss += contentLoss(contentOutput, genOutput) * CONTENT_WEIGHT #TODO: implement.
    print("   Calculating style loss.")
    styleL = 0.0
    for layerName in styleLayerNames:
        currLayer = outputDict[layerName]

        styleL += styleLoss(currLayer[1,:,:,:], currLayer[2,:,:,:])   #TODO: implement.
    totalLoss += STYLE_WEIGHT * styleL  #TODO: implement.
    # TODO: Setup gradients or use K.gradients().
    
    gradients = K.gradients(totalLoss, gFlat)
    outputs = [totalLoss, gradients[0]]
    kFunction = K.function([gFlat], outputs)

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        dpImg, tLoss, _ = fmin_l_bfgs_b(func=kFunction, x0=tData.flatten(), maxiter=200, maxfun=25)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(dpImg)
        saveFile = "./styleTransfer.png"   #TODO: Implement.
        #Image.fromarray(img).save(saveFile)   #Uncomment when everything is working right.
        imsave(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()