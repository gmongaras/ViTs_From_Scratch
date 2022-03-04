import re
import torch
from torch import nn
from torch import optim
import numpy as np
import os















class ViT(nn.Module):
    # Parameters:
    #   patchWidth - width of each patch for each image
    #   patchHeight - height of each patch for each image
    def __init__(self, patchWidth, patchHeight):
        super(ViT, self).__init__()
        
        # Save the hyperparameters of the model
        self.patchWidth = patchWidth
        self.patchHeight = patchHeight
        
        # The optimizer for this model
        #self.optimizer = optim.Adam(self.parameters())
    
    
    
    # Convert a set of images to a set of arrays of patches
    # Input:
    #   Array of size (B, L, W, C)
    #   - B = batch size
    #   - L = length of an image
    #   - W = width of an image
    #   - C = channels in an image (3 if RGB)
    # Output:
    #   Array of size (B, N, I)
    #   - B = batch size
    #   - N = number of patches where N = ((HW)/P^2)
    #   - C = channels in an image (3 is RGB)
    #   - I = flattened patch size where I = CP^2
    def getPatches(self, x):
        # Calculate the number of patches and flattened patch size
        N = (x[0].shape[0]*x[0].shape[1])/(self.patchWidth*self.patchHeight)
        I = x[0].shape[2]*self.patchWidth*self.patchHeight
        
        # Holds the reshaped images
        x_reshaped = []
        
        # Iterate over all images in the batch
        for image in range(0, len(x)):
            # Holds an array of patches for the image
            patchArr = []
            
            # Iterate over all patches
            for i in range(0, x[0].shape[0], self.patchWidth):
                for j in range(0, x[0].shape[1], self.patchHeight):
                    # Get the patch
                    patch = x[image][i:i+self.patchWidth, j:j+self.patchHeight]
                    
                    # Flatten the patch
                    patch_flat = patch.reshape(I)
                    
                    # Add the patch to the array of patches
                    patchArr.append(patch_flat)
            
            # Convert the patch array to a tensor and store it
            x_reshaped.append(torch.stack(patchArr))

        # Convert the array to a tensor and return it
        return torch.stack(x_reshaped)
    
    
    # Train the model
    # Input:
    #   x - The batch of images to classify
    #   Y - The batch of labels for each image
    #   numSteps - Number of steps to train the model
    #   batchSize - The size of each batch
    def train(self, x, Y, numSteps, batchSize):
        # Convert the images to arrays of flattened patches
        x_reshaped = self.getPatches(x)
        
        # Train the model for numSteps number of steps
        for step in range(0, numSteps):
            print()