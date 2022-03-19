from ViT import ViT
import numpy as np
import os
from PIL import Image
import torch
import random


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
#torch.autograd.set_detect_anomaly(True)



def main():
    # Hyperparameters
    patchWidth = 16             # The Width of each image patch
    patchHeight = 16            # The height of each image patch
    numSteps = 1000             # Number of steps to train the model
    batchSize = 75              # Size of each minibatch
    numBlocks = 8               # Number of transformer blocks
    numHeads = 8                # Number of attention heads to use
    keySize = 16                # Size of each key matrix
    querySize = keySize         # Size of each query matrix
    valueSize = 16              # Size of each value matrix
    hiddenSize = 768            # Size of the hidden Linear layer
    trainPercent = 0.90         # Percent of data that should be train data
    warmupSteps = 10000         # Nuber of warmup steps when changing the learning rate of the model
    numClasses = 11             # Number of classes for the newtork to predict
    
    
    # Other parameters
    pathName = "data"           # Path to load data from
    numImages = 1100            # Number of images to load from each class
                                # (use -1 to load all images)
    imgWidth = 256              # Width of each image
    imgHeight = 256             # Height of each image
    resize = False              # True to resize the images, False otherwise
    
    
    # Saving parameters
    fileSaveName = "models/modelCkPt" # Name of file to save model to
    fileLoadName = "models/savedModel" # Name of file to load model from
    stepsToSave = 2                      # Number of steps before saving the model
    saveAtBest = True           # Save the model only if it's the best so far, if
                                # if set to False, the model will overwrite the
                                # old saved model even if it's worse
    newName = True              # Use a new filename to save the model at each step
    
    
    # Model run modes
    trainModel = True           # True to train the model
    loadModel = False           # True to load the model before training
    shuffleData = True          # True to shuffle data before training and testing
    shuffleDuringTrain = True   # True to shuffle data after each training epoch
    
    
    
    
    ### Load the data ###
    
    # Holds all the images
    images = []
    
    # Dictionary to convert the classes to a number and vice versa
    classToNum = {"cat": 0, "butterfly": 1, "chicken": 2, "cow": 3, "dog": 4,
                  "elephant": 5, "horse": 6, "sheep": 7, "spider": 8,
                  "squirrel": 9, "other": 10}
    NumToClass = {v: k for k, v in classToNum.items()}
    
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
            img = np.array(img, dtype=np.float)

            # Normalize the image between 0 and 1
            img /= 255
            
            # Convert the image to a tensor
            img = torch.tensor(img, device=device, dtype=torch.float16)
            
            # Store the image in a list of images
            images.append(img)
            
            # Store the image label in the list of image labels
            try:
                labels.append(classToNum[path])
            except:
                labels.append(classToNum["other"])
            
            # Increase the image count
            imgCt += 1
            
            # If the number of images to load in is reached,
            # break the loop
            if imgCt >= numImages and numImages != -1:
                break
    
    
    
    
    ### Train the Model ###
    
    # Create a ViT Model
    model = ViT(patchWidth, patchHeight, numBlocks, keySize, querySize, valueSize, numHeads, numClasses, hiddenSize)
    
    # Convert the data to tensors
    images = torch.stack(images).float().to(device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    # Shuffle the data
    if shuffleData == True:
        shuffleArr = [i for i in range(0, images.shape[0])]
        random.shuffle(shuffleArr)
        images = images[shuffleArr]
        labels = labels[shuffleArr]

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
        model.trainModel(trainX, trainY, numSteps, batchSize, fileSaveName, stepsToSave, saveAtBest, warmupSteps, shuffleDuringTrain, newName)
    
    
    # Get a prediction on the test data
    preds = model.forward(testX)
    print(f"Predictions: {preds}")
    print(f"Labels:      {testY.detach().cpu().numpy()}")
    print(f"Diff: {np.sum(np.abs(preds-testY.detach().cpu().numpy()))}")
    print("Correct:")
    for i in range(0, len(preds)):
        if testY[i].item() == preds[i].item():
            print(f"Label: {NumToClass[testY[i].item()]}, Pred: {NumToClass[preds[i].item()]})")
    print("\nIncorrect:")
    for i in range(0, len(preds)):
        if testY[i].item() != preds[i].item():
            print(f"Label: {NumToClass[testY[i].item()]}, Pred: {NumToClass[preds[i].item()]})")
    input()






if __name__=='__main__':
    main()
