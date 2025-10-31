import numpy as np
from utils import *
import pickle
import mediapipe as mp

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

def get_prediction(frame):
  # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  H, W, _ = frame.shape
  results = hands.process(frame)
  if results.multi_hand_landmarks:
    for hand_idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):

      # Extract hand landmarks as numpy array
      original_landmarks = np.array([[lm.x * W, lm.y * H, lm.z] for lm in hand_landmarks.landmark])
      landmarks = original_landmarks.copy()

      # Mirror landmarks if hand is left
      hand_label = handedness.classification[0].label
      if hand_label == 'Left':
          landmarks = mirror_hand_landmarks(landmarks, W)

      # Perform Min-Max scaling on each axis
      x_scaled = min_max_scale(landmarks[:, 0])
      y_scaled = min_max_scale(landmarks[:, 1])
      z_scaled = min_max_scale(landmarks[:, 2])

      # Prepare the input for prediction
      data_aux = []
      for x, y, z in zip(x_scaled, y_scaled, z_scaled):
          data_aux.extend([x, y, z])

      # Predict the label using the pre-trained model
      prediction = model.predict([np.asarray(data_aux)])
      predicted_character = prediction[0].upper()
      return predicted_character