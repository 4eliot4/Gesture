# GESTURE

A ML system for real-time hand gesture recognition using computer vision. The system captures hand gestures through a webcam, processes them using MediaPipe, and classifies them using a Random Forest model.

## Overview

Three main components:
1. Data Collection and Preprocessing
2. Model Training
3. Real-time Recognition

### Features
- Real-time hand gesture detection and recognition
- Support for multiple hand detection
- Visual feedback with landmark visualization
- High accuracy classification using Random Forest
- Comprehensive performance metrics

## Requirements

### Dependencies
```bash
opencv-python
mediapipe
numpy
scikit-learn
matplotlib
seaborn
pickle
```

### Hardware
- Webcam
- Computer with sufficient processing power for real-time video analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EPFL-AI-Team/GESTURE.git
cd GESTURE
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data_collection.py     # Script for collecting and preprocessing training data
├── model_training.py      # Script for training the Random Forest classifier
├── recognition.py         # Real-time recognition script
├── utils.py               # Utility functions and constants
├── .env                   # Environment variables to be adapted (see structure on .env_example)
├── model.p                # Trained model (generated after training)
├── data.pickle            # Processed dataset (generated after data collection)
├── requirements.txt       # Required library to be installed
└── README.md
```

## Usage

### 1. Data Collection

The system requires a structured dataset of hand gesture images. Images should be organized in directories, where each directory name represents the gesture label.

```bash
python data_formatter.py
```

This script will:
- Process images from the specified data directory
- Extract hand landmarks using MediaPipe
- Normalize the coordinates
- Save processed data to `data.pickle`

### 2. Model Training

```bash
python model_training.py
```

This script will:
- Load the processed data
- Filter and pad sequences to ensure uniform length
- Split data into training and testing sets
- Train a Random Forest classifier
- Generate performance metrics and confusion matrix
- Save the trained model to `model.p`

### 3. Real-time Recognition

```bash
python inference.py
```

This script will:
- Access your webcam
- Detect and track hands in real-time
- Display landmark visualization
- Show gesture predictions
- Press 'q' to quit the application

## Model Performance

The Random Forest classifier achieves:
- Accuracy: ~X%
- Precision: ~X%
- Recall: ~X%
- F1 Score: ~X%

A confusion matrix visualization is saved as `confusion_matrix.png` after training.

## Data Processing

The system processes hand gestures through several steps:
1. Hand detection using MediaPipe
2. Landmark extraction (21 points per hand)
3. Coordinate normalization
4. Sequence padding/truncation to ensure uniform length
5. Classification using the trained model

## License

To be discussed

## Contact

EPFL AI Team 