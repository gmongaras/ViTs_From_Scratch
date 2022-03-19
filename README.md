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
