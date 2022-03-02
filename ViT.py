import re
import torch
from torch import nn
from torch import optim
import numpy as np
import os















class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        
        #
        
        # The optimizer for this model
        self.optimizer = optim.Adam(self.parameters())
    
    
    # Train the model
    # Input:
    #   x - The batch of images to classify
    #   Y - The batch of labels for each image
    #   numSteps - Number of steps to train the model
    def train(x, Y, numSteps):
        # Train the model for numSteps number of steps
        for step in range(0, numSteps):
            print()