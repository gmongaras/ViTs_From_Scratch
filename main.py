from ViT import ViT
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch



def main():
    # Hyperparameters
    #
    
    
    # Other parameters
    pathName = "data"           # Path to load data from
    numImages = 5               # Number of images to load from each class
                                # (use -1 to load all images)
    imgWidth = 300              # Width of each image
    imgHeight = 300             # Height of each image
    resize = False              # True to resize the images, False otherwise
    
    
    
    
    ### Load the data ###
    
    # Holds all the images
    images = []
    
    # Dictionary to convert the classes to a number
    classToNum = {"cat": 1, "butterfly": 2, "chicken": 3, "cow": 4, "dog": 5, "elephant": 6, "horse": 7, "sheep": 8, "spider": 9, "squirrel": 10}
    
    # Holds all image labels
    labels = []
    
    # Iterate over all folders in the image directory
    for path in os.listdir(pathName):
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
            img = torch.tensor(img)
            
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
    model = ViT()
    
    # Train the model





if __name__=='__main__':
    main()