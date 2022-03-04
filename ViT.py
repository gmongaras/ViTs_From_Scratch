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
        
    
    
    # Positionally encode the images after flatenning
    # Input:
    #   Array of size (N+1, I)
    #   - N = number of patches
    #   - I = flattened size of each patch
    # Outputs:
    #   Array of same shape as inptu with positional encodings
    def positonallyEncode(self, x_flat):
        # Make sure the dimensions are correct
        assert len(x_flat.shape) == 2
        
        # Get the positional encodings angle
        dModel = x_flat.shape[1]
        if hasattr(self, 'posEncAngle') == False:
            self.posEncAngle = torch.FloatTensor([[pos/(10000**((2*i)/dModel)) for pos in range(0, dModel)] for i in range(0, x_flat.shape[0])])
        
        # Add the positional encodings to the array
        x_flat[:, 0::2] = torch.sin(self.posEncAngle[:, 0::2])
        x_flat[:, 1::2] = torch.cos(self.posEncAngle[:, 1::2])
        
        return x_flat
    
    
    
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
    #     - The extra 1 comes from the class token
    #   - C = channels in an image (3 is RGB)
    #   - I = flattened patch size where I = CP^2
    def getPatches(self, x):
        # Make sure the dimensions are correct
        assert len(x[0].shape) == 3
        
        # Calculate the number of patches and flattened patch size
        N = (x[0].shape[0]*x[0].shape[1])/(self.patchWidth*self.patchHeight)
        I = x[0].shape[2]*self.patchWidth*self.patchHeight
        
        # Holds the reshaped images
        x_reshaped = []
        
        # Iterate over all images in the batch
        for image in range(0, len(x)):
            # Holds an array of patches for the image
            patchArr = []
            
            # Add the class token to the beginning of the patch array
            patchArr.append(torch.FloatTensor(np.random.uniform(low=0, high=x[0].shape[0], size=(I))))
            
            # Iterate over all patches
            for i in range(0, x[0].shape[0], self.patchWidth):
                for j in range(0, x[0].shape[1], self.patchHeight):
                    # Get the patch
                    patch = x[image][i:i+self.patchWidth, j:j+self.patchHeight]
                    
                    # Flatten the patch
                    patch_flat = patch.reshape(I)
                    
                    # Add the patch to the array of patches
                    patchArr.append(patch_flat)
            
            # Convert the patch array to a tensor
            patchArr_T = torch.stack(patchArr)
            
            # Positionally encode the patch array and store it
            x_reshaped.append(self.positonallyEncode(patchArr_T))

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