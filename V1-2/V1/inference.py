import pickle
import cv2
import mediapipe as mp
import numpy as np
from utils import *


# Load the pre-trained machine learning model from pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # The trained classifier


# Open the default camera (change index if you have multiple cameras)
cap = cv2.VideoCapture(0)


# Initialize MediaPipe components for hand detection and drawing
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles 

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)


# Main loop: capture frames and run inference
while True:
    ret, frame = cap.read()
    if not ret:
        continue  # Skip this iteration if frame capture failed

    H, W, _ = frame.shape  # Get frame dimensions

    # Convert BGR frame to RGB (MediaPipe requires RGB format for processing)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands and landmarks in the frame
    results = hands.process(frame_rgb)

    # If any hands are detected in the frame
    if results.multi_hand_landmarks:
        for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):

            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),  
                mp_drawing_styles.get_default_hand_connections_style())  

            # Extract hand landmarks as numpy array (pixel coordinates)
            original_landmarks = np.array([[lm.x * W, lm.y * H, lm.z] for lm in hand_landmarks.landmark])
            landmarks = original_landmarks.copy()

            # Mirror landmarks if hand is left for consistency
            hand_label = handedness.classification[0].label
            if hand_label == 'Left':
                landmarks = mirror_hand_landmarks(landmarks, W)

            # Perform Min-Max scaling on each axis
            x_scaled = min_max_scale(landmarks[:, 0])
            y_scaled = min_max_scale(landmarks[:, 1])
            z_scaled = min_max_scale(landmarks[:, 2])

            # Prepare the input for prediction (flattened list)
            data_aux = []
            for x, y, z in zip(x_scaled, y_scaled, z_scaled):
                data_aux.extend([x, y, z])

            # Predict the label using the pre-trained model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0].upper()

            # Calculate bounding box based on original landmarks (before mirroring)
            x1 = int(np.min(original_landmarks[:, 0])) - 10
            y1 = int(np.min(original_landmarks[:, 1])) - 10
            x2 = int(np.max(original_landmarks[:, 0])) + 10
            y2 = int(np.max(original_landmarks[:, 1])) + 10

            # Offset text for multiple hands
            text_offset = hand_idx * 30
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, 
                       f"Hand {hand_idx + 1}: {predicted_character}", 
                       (x1, y1 - 10 - text_offset),  
                       cv2.FONT_HERSHEY_SIMPLEX,  
                       1.3,  
                       (0, 0, 0),  
                       3,  
                       cv2.LINE_AA)  

    # Show the annotated frame
    cv2.imshow('frame', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()