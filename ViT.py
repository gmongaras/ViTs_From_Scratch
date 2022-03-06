import re
from tkinter.tix import Y_REGION
import torch
from torch import nn
from torch import optim
import numpy as np
import os
import random


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    
    
    
    
class multiHeadAttention(nn.Module):
    # Inputs:
    #   keySize - Size of each key matrix for self attention
    #   querySize - Size of each query matrix for self attention
    #   valueSize - Size of each value matrix for self attention
    #   numHeads - Number of self attention heads to use
    #   embeddingSize (I) - The embedding size of each patch (I = CP^2)
    def __init__(self, keySize, querySize, valueSize, numHeads, embeddingSize):
        super(multiHeadAttention, self).__init__()
        
        # Store the hyperparameters
        self.keySize = keySize
        self.querySize = querySize
        self.valueSize = valueSize
        self.numHeads = numHeads
        self.embeddingSize = embeddingSize
        
        # Create the matrices
        self.keyWeights = [nn.Parameter(torch.tensor(np.random.uniform(low=-1, high=1, size=(embeddingSize, keySize)), requires_grad=True, device=device, dtype=torch.float32)) for i in range(0, numHeads)]
        self.queryWeights = [nn.Parameter(torch.tensor(np.random.uniform(low=-1, high=1, size=(embeddingSize, querySize)), requires_grad=True, device=device, dtype=torch.float32)) for i in range(0, numHeads)]
        self.valueWeights = [nn.Parameter(torch.tensor(np.random.uniform(low=-1, high=1, size=(embeddingSize, valueSize)), requires_grad=True, device=device, dtype=torch.float32)) for i in range(0, numHeads)]
        
        # Convert the matrices to parameters
        self.keyWeights = nn.ParameterList(self.keyWeights)
        self.queryWeights = nn.ParameterList(self.queryWeights)
        self.valueWeights = nn.ParameterList(self.valueWeights)
        
        # The matrix to convert the attention back to the
        # input shape
        self.conversionMatrix = nn.Parameter(torch.tensor(np.random.uniform(0, 1, size=(numHeads*valueSize, embeddingSize)), requires_grad=True, dtype=torch.float32, device=device), requires_grad=True)
    
    
    
    # Compute the self attention for each image in the batch
    def selfAttention(self, x, idx):
        # Compute the key, queries, and values
        keys = torch.matmul(x, self.keyWeights[idx])
        queries = torch.matmul(x, self.queryWeights[idx])
        values = torch.matmul(x, self.valueWeights[idx])
        
        # Compute the attention
        return torch.matmul(nn.functional.softmax(torch.matmul(queries, keys.reshape(keys.shape[0], keys.shape[2], keys.shape[1]))/int(x.shape[1])**0.5, dim=-1), values)

    
    
    # Compute the multihead attention of the inputs
    def forward(self, x):
        # Compute `numHeads` number of attention
        attention = torch.tensor([], requires_grad=True, dtype=x.dtype)
        for att in range(0, self.numHeads):
            attention = torch.cat((attention, self.selfAttention(x, att)), dim=-1)
        
        # Convert the attention to the original dimensions
        return torch.matmul(attention, self.conversionMatrix)






class TransformerBlock(nn.Module):
    # Inputs:
    #   keySize - Size of each key matrix for self attention
    #   querySize - Size of each query matrix for self attention
    #   valueSize - Size of each value matrix for self attention
    #   numHeads - Number of self attention heads to use
    #   embeddingSize (I) - The embedding size of each patch (I = CP^2)
    #   hiddenSize - The size of the hideen linear layer
    def __init__(self, keySize, querySize, valueSize, numHeads, embeddingSize, hiddenSize):
        super(TransformerBlock, self).__init__()
        
        # The norm blocks in the model
        self.norm1 = nn.LayerNorm(embeddingSize)
        self.norm2 = nn.LayerNorm(embeddingSize)
        
        # The multihead attention for this block
        self.multiHeadAttention = multiHeadAttention(keySize, querySize, valueSize, numHeads, embeddingSize)
        
        # The linear layers of the model
        self.linear1 = nn.Linear(embeddingSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, embeddingSize)
        
        # The GELU layer
        self.GELU = nn.GELU()
    
    
    
    # The forward feed method of this transformer block
    # Input:
    #   Array of size (B, N+1, I)
    #   - B = batch size
    #   - N = number of patches where N = ((HW)/P^2)
    #   - I = flattened patch size where I = CP^2
    # Output:
    #   Encoded array of size (B, N+1, I)
    #   - B = batch size
    #   - N = number of patches where N = ((HW)/P^2)
    #   - I = flattened patch size where I = CP^2
    def forward(self, x):
        # First sub-block
        norm1 = self.norm1(x)
        MHA = self.multiHeadAttention(norm1)
        
        # First residual connection
        res1 = MHA + x
        
        # Second sub-block
        norm2 = self.norm2(res1)
        linear1 = self.linear1(norm2)
        GELU = self.GELU(linear1)
        linear2 = self.linear2(GELU)
        
        # Final residual connection
        res2 = linear2 + res1
        
        # Return the final output
        return res2
        






class ViT(nn.Module):
    # Parameters:
    #   patchWidth - width of each patch for each image
    #   patchHeight - height of each patch for each image
    #   numBlocks - Number of transformer blocks to use
    #   keySize - Size of each key matrix for self attention
    #   querySize - Size of each query matrix for self attention
    #   valueSize - Size of each value matrix for self attention
    #   numHeads - Number of self attention heads to use
    #   numClasses - Number of classes to predict
    #   hiddenSize - Size of the hidden Linear layer
    #   MLPSize - Size of the final MLP layer
    def __init__(self, patchWidth, patchHeight, numBlocks, keySize, querySize, valueSize, numHeads, numClasses, hiddenSize, MLPSize):
        super(ViT, self).__init__()
        
        # Save the hyperparameters of the model
        self.patchWidth = patchWidth
        self.patchHeight = patchHeight
        self.numChannels = 3
        
        # Create the transformer blocks while registering the parameters
        self.transformerBlocks = []
        self.inputParameters = []
        for i in range(0, numBlocks):
            self.transformerBlocks.append(TransformerBlock(keySize, querySize, valueSize, numHeads, patchWidth*patchWidth*self.numChannels, hiddenSize))
            self.inputParameters += [b for b in self.transformerBlocks[i].parameters()]
        self.inputParameters = nn.ParameterList(self.inputParameters)
        
        # MLP and softmax layers for the final output
        self.linear1 = nn.Linear(patchWidth*patchHeight*self.numChannels, MLPSize)
        self.linear2 = nn.Linear(MLPSize, numClasses)
        self.softmax = nn.Softmax(dim=-1)
        
        # The optimizer for this model
        self.optimizer = optim.Adam(self.parameters())
        
    
    
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
            self.posEncAngle = torch.FloatTensor([[pos/(10000**((2*i)/dModel)) for pos in range(0, dModel)] for i in range(0, x_flat.shape[0])], device=device)
        
        # Add the positional encodings to the array
        x_flat[:, 0::2] += torch.sin(self.posEncAngle[:, 0::2])
        x_flat[:, 1::2] += torch.cos(self.posEncAngle[:, 1::2])
        
        return x_flat
    
    
    
    # Convert a set of images to a set of arrays of patches
    # Input:
    #   Array of size (B, L, W, C)
    #   - B = batch size
    #   - L = length of an image
    #   - W = width of an image
    #   - C = channels in an image (3 if RGB)
    # Output:
    #   Array of size (B, N+1, I)
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
            patchArr.append(torch.FloatTensor(np.random.uniform(low=0, high=x[0].shape[0], size=(I)), device=device))
            
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
        return torch.stack(x_reshaped).to(device=device)

    
    
    # Cross Entropy The loss function for the model
    # Inputs:
    #   p - The probabilities we want (Probably a one-hot vector)
    #   q - The probabilities the model predicted
    def CrossEntropyLoss(self, p, q):
        # Ensure the values cannot be 0 or 1 to prevent
        # inf and NaN in the output and the gradients
        q = torch.where(q < 0.000001, q+0.000001, q)
        q = torch.where(q > 0.999999, q-0.000001, q)
        
        return -(1/p.shape[0])*torch.sum(p*torch.log(q) + (1-p)*torch.log(1-q), dim=-1)

    
    # MSE Loss function for the model
    # Inputs:
    #   Y - The labels we want the model to predict
    #   Y_hat - The values the model predicted
    def MSE(self, Y, Y_hat):
        return (1/Y.shape[0])*torch.sum((Y-Y_hat)**2)
    
    
    
    # Train the model
    # Input:
    #   x - The batch of images to classify
    #   Y - The batch of labels for each image
    #   numSteps - Number of steps to train the model
    #   batchSize - The size of each batch
    #   fileSaveName - name of file to save model to
    #   stepsToSave - Number of steps before saving the model
    #   saveAtBest - Whether the file should only be saved if
    #                it's the new best model
    def train(self, x, Y, numSteps, batchSize, fileSaveName, stepsToSave, saveAtBest):
        # Convert the images to arrays of flattened patches
        x_reshaped = self.getPatches(x)
        
        # Shuffle the inputs and labels
        shuffleArr = [i for i in range(0, x_reshaped.shape[0])]
        random.shuffle(shuffleArr)
        x_reshaped = x_reshaped[shuffleArr]
        Y = torch.tensor(Y, dtype=torch.long, device=device)[shuffleArr]
        
        # Split the data into batches
        x_batches = torch.split(x_reshaped, batchSize)
        Y_batches = torch.split(Y, batchSize)
        
        # The best lost out of all steps
        bestLoss = np.inf
        
        # Train the model for numSteps number of steps
        for step in range(1, numSteps+1):
            # The total loss over batches
            totalLoss = 0
            
            # Iterate over all batches
            for batch in range(0, len(x_batches)):
                
                # Get the batch
                x_batch = x_batches[batch]
                Y_batch = Y_batches[batch]
                
                # Send the images through the transformer blocks
                trans = x_batch
                for block in range(0, len(self.transformerBlocks)):
                    trans = self.transformerBlocks[block](trans)
                
                # Get the softmax predictions from the network
                linear1 = self.linear1(trans[:, 0])
                linear2 = self.linear2(linear1)
                soft = self.softmax(linear2)
                
                # Get the class prediction
                classPreds = torch.argmax(soft, dim=-1)
                
                # One hot encode the labels
                Y_oneHot = nn.functional.one_hot(Y_batch, num_classes=soft.shape[-1])
                
                # Get the loss for the batch
                loss = self.CrossEntropyLoss(Y_oneHot, soft).mean()
                totalLoss += loss.detach().item()
                
                # Backpropogate the loss
                loss.backward()
                
                # Step the optimizer
                self.optimizer.step()
                
                # Zero the gradients
                self.optimizer.zero_grad()
            
            # Print the loss and example output
            print(f"Step number: {step}")
            print(f"Total loss: {totalLoss}")
            print(f"Actual Labels: {Y_batch.detach().numpy()}")
            print(f"Predictions: {classPreds.detach().numpy()}")
            print()
            
            # Check if the model should be saved every `stepsToSave` steps
            if step%stepsToSave == 0:
                if saveAtBest == True:
                    if totalLoss < bestLoss:
                        bestLoss = totalLoss
                        print("Saving Model\n")
                        self.saveModel(fileSaveName)
                else:
                    print("Saving Model\n")
                    self.saveModel(fileSaveName)
    
    
    
    # Get a prediction from the network on a batch of data
    # Inputs:
    #   x - The batch of images to classify
    #   Y - The classes of the images we want to classify
    #       (this variable defaults to None which won't produce a loss)
    def forward(self, x, Y=None):
        # Convert the images to arrays of flattened patches
        x_reshaped = self.getPatches(x)
        
        # Shuffle the inputs and labels
        shuffleArr = [i for i in range(0, x_reshaped.shape[0])]
        random.shuffle(shuffleArr)
        x_reshaped = x_reshaped[shuffleArr]
        Y = torch.tensor(Y, dtype=torch.long, device=device)[shuffleArr]
        
        
        
        # Send the images through the transformer blocks
        trans = x_reshaped
        for block in range(0, len(self.transformerBlocks)):
            trans = self.transformerBlocks[block](trans)
        
        # Get the softmax predictions from the network
        linear1 = self.linear1(trans[:, 0])
        linear2 = self.linear2(linear1)
        soft = self.softmax(linear2)
        
        # Get the class prediction
        classPreds = torch.argmax(soft, dim=-1)
        
        # If labels were given, get the loss
        if Y != None:
            # One hot encode the labels
            Y_oneHot = nn.functional.one_hot(Y, num_classes=soft.shape[-1])
            
            # Get the loss for the batch
            loss = self.CrossEntropyLoss(Y_oneHot, soft).mean()
        
            # Return the predictions and loss
            return classPreds.detach().numpy(), loss.detach().item()

        # Return the predictions
        return classPreds.detach().numpy()
    
    
    # Save a model to the specified path name
    # Input:
    #   fileName - The name of the file to save the model to
    def saveModel(self, fileName):
        # Get the last separator in the filename
        dirName = "/".join(fileName.split("/")[0:-1])
    
        # If the directory doesn't exist, create it
        try:
            if (not os.path.isdir(dirName) and dirName != ''):
                os.makedirs(dirName, exist_ok=True)
        
            torch.save(self.state_dict(), fileName)
        
        except:
            torch.save(self.state_dict(), fileName)
            
    
    
    # Load a model from the specified path name
    # Input:
    #   fileName - The name of the file to load the model from
    def loadModel(self, fileName):
        # If the file doesn't exist, raise an error
        if (not os.path.isfile(fileName)):
            raise Exception("Specified model file does no exist")
        
        # Load the model
        self.load_state_dict(torch.load(fileName))
        self.eval()