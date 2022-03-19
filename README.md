# General Proect Information
<b>Title:</b> ViTs From Scratch

<b>Data From:</b> https://www.kaggle.com/alessiocorrado99/animals10

# Project Description:
In this project, I aim to classify an image of an animal into one of 10 different animal classes using a vision transformer (ViT). This project is based on the paper "An Image is Worth 16x16 Words":

https://arxiv.org/abs/2010.11929

Using Pytorch, I built the ViT model proposed in the paper from scratch to learn about the inner wokrings of the model.

## Model Architecture

The model uses a tranformer-based architecture to analyze an image using the following steps:
1. Split each image into 16x16 patches
2. Add positional encodings to each patch
3. Flatten each patch from a 3-dimensional matrix to a 1-dimensional vector
4. Concatentate each patch along with a class token (which is just a vector the same length as each image patch) to produce a vector of flattened image patches
6. Send the flattened patch vector through L transformer encoder block
7. Send the outputs from the final transformer encoder block through a MLP and softmax layer
8. Get the final prediction from the softmax layer

The model architecture can be seen below

<img width="658" alt="image" src="https://user-images.githubusercontent.com/43501738/159134339-857a563b-e341-418f-ab00-e9a117d683f5.png">

# Project Requirements
Below are the following python library requirements to be able to run the project

- PyTorch - 1.8.1
- NumPy - 1.19.5
- Pillow (PIL) - 8.2.0

Note: The library version I used are listed as well to ensure the model can be run successfully on other computers.

Additionally, I used `Python 3.8.10` when testing the model.

# Project Execution
To execute this project, download the repository and use the following command to run the project:

`python3 main.py`

## Model Parameters
There are many parameters that can be tuned in the model, the parameters can be found at the beginning of the main function in `main.py`. Please note that the larger the keys, values, pathWidth, etc., the more memory the model will require to train.

Below is a description of each parameter:

#### Hyperparameters
Hyperparameters used to tune the model
- patchWidth (16 pixels) - The width of each image patch
- pathHeight (16 pixels) - The height of each image patch
- numSteps (1000 steps) - The number of steps to train the model for
- batchSize (75 images) - The number of images in each minibatch
- numBlocks (8 blocks) - The number of blocks used in the transformer encoder
- numHeads (8 heads) - The number of heads to use for each multi-head attention block
- keySize (16 parameters) - The size of each key matrix to use in each self-attention block
- querySize (16 parameters) - The size of each query matrix to use in each self-attention block
- valueSize (16 parameters) - The size of each value matrix to use in each self-attention block
- hiddenSize (768 parametes) - The size of each matrix used in the multi-head attention blocks to convert the multi-head attention to a shape of the same size as the input encodings.
- trainPercent (90 percent) - The percent of data that will be train data (1-trainPercent will be test data)
- warmupSteps (10000 steps) - The number of warmup steps which is a value used in the optimizer
- numClasses (10 classes) - The total number of classes the classifier can choose from

#### Other parameters
Other parameters used before training
- pathName ("data") - The location of the directory to load images from
- numImages (1100) - The number of images to load from each class (instead of loading all data, this parameter will load 'numImages' number of images from each class)
  - Note: Use -1 to load all images
- imgWidth (256) - The width of each input image (or the desired with of each input image)
- imgHeight (256) - The height of each input image (or the desired height of each input image)
- resize (False) - If this flag is True, the script will reize all images using the imgWidth and imgHeight parameters before training. So, if the input images are not of the same size, this flag will be useful to resize the images before training automatically

#### Saving parameters
Parameters used to save the model
- fileSaveName ("models/modelCkPt") - The file to save model checkpoints to
- fileLoadName ("models/savedModel") - The file to load a model from, if the loadModel flag is set to True
- stepsToSave (2) - Number of steps before saving a checkpoint of the model
- saveAtBest (True) - If the flag is set to True, the model will be saved only if it has a lower loss at the next save step. Otherwise, it will always save at each save step
- newName (True) - Use a different name for each model checkpoint. The step which the model was saved will be appended to the end of the name of the file specified in the fileSaveName parameter

#### Model Run Modes
Different modes to run the model
- trainModel (True) - If True, the model is trained, otherwise the model is not trained
- loadModel (False) - If True, a model is loaded from the path specified by the fileLoadName parameter, otherwise the model is initialized to a random model
- shuffleData (True) - If True, the data is shuffled before created the test-train split (helps the model perform better)
- shuffleDuringTrain (True) - If True, the data is shuffled before sending it through the model (help the model learn better)


## Training the Model
To train the model, set the
