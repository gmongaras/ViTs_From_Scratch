from ViT import ViT
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
torch.autograd.set_detect_anomaly(True)



def main():
    # Hyperparameters
    patchWidth = 16             # The Width of each image patch
    patchHeight = 16            # The height of each image patch
    numSteps = 1000             # Number of steps to train the model
    batchSize = 15              # Size of each minibatch
    numBlocks = 12              # Number of transformer blocks
    numHeads = 12               # Number of attention heads to use
    keySize = 16                # Size of each key matrix
    querySize = keySize         # Size of each query matrix
    valueSize = 16              # Size of each value matrix
    hiddenSize = 768            # Size of the hidden Linear layer
    MLPSize = 3072              # Size of the final MLP layer
    trainPercent = 0.85         # Percent of data that should be train data
    warmupSteps = 40            # Nuber of warmup steps when chainging the larning rate of the model
    
    
    # Other parameters
    pathName = "data"           # Path to load data from
    numImages = 250             # Number of images to load from each class
                                # (use -1 to load all images)
    imgWidth = 256              # Width of each image
    imgHeight = 256             # Height of each image
    resize = False              # True to resize the images, False otherwise
    
    
    # Saving parameters
    fileSaveName = "models/modelCkPt.pt" # Name of file to save model to
    fileLoadName = "models/modelCkPt.pt" # Name of file to load model from
    stepsToSave = 5                      # Number of steps before saving the model
    saveAtBest = True           # Save the model only if it's the best so far, if
                                # if set to False, the model will overwrite the
                                # old saved model even if it's worse
    
    
    # Model run modes
    trainModel = True           # True to train the model
    loadModel = False           # True to load the model before training
    shuffleTrain = True         # True to shuffle data on training
    shuffleFor = False          # True to shuffle data on forward pass
    
    
    
    
    ### Load the data ###
    
    # Holds all the images
    images = []
    
    # Dictionary to convert the classes to a number
    classToNum = {"cat": 0, "butterfly": 1, "chicken": 2, "cow": 3, "dog": 4, "elephant": 5, "horse": 6, "sheep": 7, "spider": 8, "squirrel": 9}
    
    # Holds all image labels
    labels = []
    
    # Holds the total number of classes
    numClasses = 0
    
    # Iterate over all folders in the image directory
    for path in os.listdir(pathName):
        # Increase the number of classes
        numClasses += 1
        
        # Construct the path
        dirPath = os.path.join(pathName, path)
        
        # The number of files in the directory
        imgCt = 0
        
        # Iterate over all files in the data directory
        for file in os.listdir(dirPath):
            # Get the full file name
            fullFile = os.path.join(dirPath, file)
            
            # Load the image into memory
            img = Image.open(fullFile)
            
            # Resize the image if specified
            if resize:
                img = img.resize((imgWidth, imgHeight), Image.ANTIALIAS)
            
            # Convert the image to RGB
            img = img.convert('RGB')
            
            # Conver the image to a numpy array
            img = np.array(img, dtype=np.float)

            # Normalize the image between 0 and 1
            img /= 255
            
            # Convert the image to a tensor
            img = torch.tensor(img, device=device, dtype=torch.float16)
            
            # Store the image in a list of images
            images.append(img)
            
            # Store the image label in the list of image labels
            labels.append(classToNum[path])
            
            # Increase the image count
            imgCt += 1
            
            # If the number of images to load in is reached,
            # break the loop
            if imgCt >= numImages and numImages != -1:
                break
    
    
    
    
    ### Train the Model ###
    
    # Create a ViT Model
    model = ViT(patchWidth, patchHeight, numBlocks, keySize, querySize, valueSize, numHeads, numClasses, hiddenSize, MLPSize)
    
    # Split the data into test and train data
    trainX = images[:int(len(images)*trainPercent)]
    trainY = labels[:int(len(images)*trainPercent)]
    testX = images[int(len(images)*trainPercent):]
    testY = labels[int(len(images)*trainPercent):]
    
    # Load the model if requested to do so
    if loadModel:
        model.loadModel(fileLoadName)
    
    # Train the model if requested to do so
    if trainModel:
        model.trainModel(trainX, trainY, numSteps, batchSize, fileSaveName, stepsToSave, saveAtBest, shuffleTrain, warmupSteps)
    
    
    # Get a prediction on the test data
    preds, loss = model.forward(testX, testY, shuffleFor)
    print(f"Predictions: {preds}")
    print(f"Labels: {testY}")
    print(f"Loss: {loss}")





if __name__=='__main__':
    main()