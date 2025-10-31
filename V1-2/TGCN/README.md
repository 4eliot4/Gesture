Code for the TGCN model from Spring 2025


## What is this?

This folder details the implementation of the Temporal Graph Convolution Network model for GESTURE. It contains preprocessing, training, and inference code.

## How to run

Run from folder root (same directory as this file) using the following structure
python -m foldername.filename (eg python -m training.train_tgcn)

- Run keypoint extraction from dataset-preprocessing
- Run training by running training/train_tgcn
- Model used is described in models folder 
- Inference is described in the inference folder. Contains code for the chatbot, and inference app itself.

## Data

- Keypoint extraction: Please extract and insert "videos" folder from Sharepoint into training folder
- Training: Please extract and insert "keypoints" folder from Sharepoint into dataset-preprocessing folder in V1-2
- Please insert "TGCN.pt" from Sharepoint under the model_weights folder for inference
