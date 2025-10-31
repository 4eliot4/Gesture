
# Import standard and third-party libraries
import os  # For file and directory operations
import pickle  # For saving processed data
import mediapipe as mp  # For hand landmark detection
import cv2  # For image processing
import numpy as np  # For numerical operations

# Import environment and project-specific utilities
from dotenv import load_dotenv  # For loading environment variables
from utils import *  # Custom utility functions (e.g., min_max_scale, mirror_hand_landmarks, signs)



# Load environment variables from .env file
load_dotenv(override=True)

# Directory containing the dataset (set in .env)
DATA_DIR = os.getenv('DATA_DIR')

# TODO: normalize the hand: only have right oriented or left oriented,
# Implies that we have to stick with e.g right and invert the left hands to keep consistancy

# Initialize MediaPipe components for hand detection and drawing
mp_hands = mp.solutions.hands  # Hand landmark model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing styles

# Set up MediaPipe Hands: static_image_mode=True treats each image independently
# min_detection_confidence=0.3 sets the minimum confidence for detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a dictionary mapping numbers to lowercase letters (0='a', 1='b', ...)
signs = {i: c for i, c in enumerate(signs, 0)}


def process_image_directory(directory, valid_extensions):
    """
    Process all subdirectories in a given directory, extracting hand landmark data from images.
    Each subdirectory is treated as a class label.

    Args:
        directory (str): Path to the parent directory containing class subfolders.
        valid_extensions (tuple): Valid image file extensions (e.g., ('.png', '.jpg', '.jpeg')).

    Returns:
        data (list): List of normalized landmark coordinate data for all images.
        labels (list): List of class labels corresponding to each image (directory names).
    """


    data = []  # List to store landmark data for all images
    labels = []  # List to store corresponding class labels

    # Iterate over each subdirectory (class) in the parent directory
    for dir_ in os.listdir(directory):
        print(f'Loading data from directory "{dir_}"...')
        dir_path = os.path.join(directory, dir_)

        # Skip if not a directory
        if not os.path.isdir(dir_path):
            print(f'Skipping "{dir_}", is not a directory.')
            continue

        # Process each image file in the class directory
        for img_path in os.listdir(dir_path):
            if not img_path.lower().endswith(valid_extensions):
                print(f'Skipping file "{img_path}", does not have a valid file type.')
                continue
            
            data_aux = []
            x_ = []
            y_ = []
            z_ = []
            
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            image_height, image_width, _ = img.shape

            # Run MediaPipe Hands to detect hand landmarks
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Get hand label (Left or Right) and confidence
                    hand_label = handedness.classification[0].label
                    confidence = handedness.classification[0].score

                    # Mirror left hand landmarks for consistency
                    if hand_label == 'Left':
                        print(f"Detected {hand_label} hand with confidence {confidence:.2f} ---> Mirroring the hand")
                        landmarks = np.array([[lm.x * image_width, lm.y * image_height, lm.z] for lm in hand_landmarks.landmark])
                        mirrored_landmarks = mirror_hand_landmarks(landmarks, image_width)
                    else:
                        mirrored_landmarks = np.array([[lm.x * image_width, lm.y * image_height, lm.z] for lm in hand_landmarks.landmark])

                    # Normalize coordinates to [0, 1] range
                    x_scaled = min_max_scale(mirrored_landmarks[:, 0])
                    y_scaled = min_max_scale(mirrored_landmarks[:, 1])
                    z_scaled = min_max_scale(mirrored_landmarks[:, 2])

                    # Flatten and collect scaled coordinates
                    data_aux = []
                    for x, y, z in zip(x_scaled, y_scaled, z_scaled):
                        data_aux.append(x)
                        data_aux.append(y)
                        data_aux.append(z)

                    data.append(data_aux)
                    labels.append(dir_)

    return data, labels


f = open('data.pickle', 'wb')

# Main script: process all class directories in DATA_DIR and save results
data = []     # All landmark data
labels = []  # All labels
source = []  # Source index for each label

for i, directory in enumerate(os.listdir(DATA_DIR)):

    if not os.path.isdir(os.path.join(DATA_DIR, directory)):
            print(f'Skipping directory "{directory}", is not a directory.')
            continue
    
    print(f"Source: {directory}, {i}.")
    d, l = process_image_directory(os.path.join(DATA_DIR, directory), valid_extensions)
    data.extend(d)
    labels.extend(l)
    source.extend([i]*len(l))

# Save the processed data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'source': source}, f)