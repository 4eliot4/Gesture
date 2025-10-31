import cv2
import os
from string import ascii_lowercase
from utils import *

# Directory configurations
DATA_DIR = './dataIMAGES'  # Directory where collected images will be saved
REFERENCE_DIR = './reference_signs'  # Directory containing reference images for each sign
ESC_key = 27  # ASCII code for the ESC key

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Number of sign classes (e.g., 26 letters + 9 numbers)
NUMBER_OF_CLASSES = 35
# Number of images to collect per class
DATASET_SIZE = 100



def load_reference_image(character):
    """
    Load and resize the reference image for a given character.
    Returns the image if found, otherwise None.
    """
    ref_path = os.path.join(REFERENCE_DIR, f"{character}.png")
    if os.path.exists(ref_path):
        ref_img = cv2.imread(ref_path)
        return cv2.resize(ref_img, (200, 250))
    return None



def create_info_display(frame, reference_img, current_letter, counter, dataset_size):
    """
    Create a display window that shows:
    - The camera feed
    - A black bar at the top with info text
    - The reference image for the current sign (if available)
    - Progress and instructions
    """
    h, w = frame.shape[:2]
    # Add space at the top for info text
    display = cv2.copyMakeBorder(frame, 100, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Draw black rectangle for text background
    display[0:100, 0:w] = (0, 0, 0)

    # Prepare info text lines
    text_lines = [
        f"Current Sign: {current_letter.upper()}",
        f"Progress: {counter}/{dataset_size}",
        "Press 'q' to start collecting",
        "Press 'ESC' to exit"
    ]
    # Draw each line of text
    for i, line in enumerate(text_lines):
        y_offset = 30 + i * 30
        cv2.putText(display, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Overlay reference image in bottom left if available
    if reference_img is not None:
        ref_h, ref_w = reference_img.shape[:2]
        display[h - ref_h:h, 0:ref_w] = reference_img

    return display



def start_data_collection(cap):
    """
    Main loop for collecting images for each sign class.
    For each class:
      - Wait for user to press 'q' to start collecting
      - Show reference image and info
      - Save DATASET_SIZE images to the class directory
      - Allow exit at any time with ESC
    """
    for j in range(NUMBER_OF_CLASSES):
        current_letter = SIGNS[j]  # Get the current sign/letter
        letter_dir = os.path.join(DATA_DIR, current_letter)
        os.makedirs(letter_dir, exist_ok=True)  # Ensure class directory exists

        reference_img = load_reference_image(current_letter)
        print(f"Collecting data for class {current_letter.upper()}")

        # Wait for user to start collecting for this class
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            display = create_info_display(frame, reference_img, current_letter, 0, DATASET_SIZE)
            cv2.imshow('Data Collection', display)

            key = cv2.waitKey(1)
            if key == ord('q'):  # Start collecting images
                break
            elif key == ESC_key:  # ESC key to exit
                cap.release()
                cv2.destroyAllWindows()
                return

        # Collect and save images for this class
        for counter in range(DATASET_SIZE):
            ret, frame = cap.read()
            if not ret:
                continue

            display = create_info_display(frame, reference_img, current_letter, counter, DATASET_SIZE)
            cv2.imshow('Data Collection', display)

            # Save the current frame as an image in the class directory
            cv2.imwrite(os.path.join(letter_dir, f'{counter}.jpg'), frame)

            key = cv2.waitKey(25)
            if key == 27:  # ESC key to exit
                cap.release()
                cv2.destroyAllWindows()
                return


# Change this if needed
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit(0)
try:
    start_data_collection(cap)
finally:
    # Release camera and close all OpenCV windows on exit
    cap.release()
    cv2.destroyAllWindows()
