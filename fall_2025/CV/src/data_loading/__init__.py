import mediapipe as mp

# The landmarks we chose
POSE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
FACE_LANDMARKS = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 361] 
HAND_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Get MediaPipe connections
mp_holistic = mp.solutions.holistic

# Pose connections (33 landmarks)
POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS  # Set of tuples (start, end)

# Face connections (468 landmarks)
FACEMESH_CONTOURS = mp_holistic.FACEMESH_CONTOURS

# Hand connections (21 landmarks each)
HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS