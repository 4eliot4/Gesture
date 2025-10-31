from string import ascii_lowercase
import numpy as np


def mirror_hand_landmarks(landmarks, image_width):
    mirrored_landmarks = landmarks.copy()
    mirrored_landmarks[:, 0] = image_width - landmarks[:, 0]  # Mirror x-coordinates
    return mirrored_landmarks


def min_max_scale(values, feature_range=(0, 1)):
    min_val, max_val = feature_range
    v_min = np.min(values)
    v_max = np.max(values)
    
    # Avoid division by zero
    if v_max == v_min:
        return np.full_like(values, min_val, dtype=np.float64)  # Return array with min_val if range is zero
    
    scaled_values = (values - v_min) / (v_max - v_min)  # Scale to [0, 1]
    return scaled_values * (max_val - min_val) + min_val 


valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',\
                11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',\
                      21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4',\
                          31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

# SIGNS = {i: c for i, c in enumerate(ascii_lowercase + "123456789")}
SIGNS = {i: c for i, c in enumerate(ascii_lowercase)}

# signs = ascii_lowercase + "0123456789"
signs = ascii_lowercase

valid_labels = list(ascii_lowercase)