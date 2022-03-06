from ViT import ViT
import re
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
    batchSize = 10              # Size of each minibatch
    numBlocks = 12              # Number of transformer blocks
    numHeads = 12               # Number of attention heads to use
    keySize = 16                # Size of each key matrix
    querySize = keySize         # Size of each query matrix
    valueSize = 16              # Size of each value matrix
    hiddenSize = 768            # Size of the hidden Linear layer
    MLPSize = 3072              # Size of the final MLP layer
    
    
    # Other parameters
    pathName = "data"           # Path to load data from
    numImages = 5               # Number of images to load from each class
                                # (use -1 to load all images)
    imgWidth = 256              # Width of each image
    imgHeight = 256             # Height of each image
    resize = False              # True to resize the images, False otherwise
    
    
    
    
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
            img = np.array(img)
            
            # Convert the image to a tensor
            img = torch.tensor(img, device=device)
            
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
    
    # Train the model
    model.train(images, labels, numSteps, batchSize)





if __name__=='__main__':
    main()