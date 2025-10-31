

# PLEASE PUT THE 'sorted_videos' in dataset-preprocesseing before running this script.
# Its job is to generate both the landmarks and features from those videos
import os
import cv2
import json
import time
import logging
import torch
import numpy as np
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ------------- Helper Functions -------------

def compute_difference(x_list):
    """Compute pairwise differences among elements of x_list (a Python list of numbers)."""
    diff = []
    for i, xi in enumerate(x_list):
        row = []
        for j, xj in enumerate(x_list):
            if i != j:
                row.append(xi - xj)
        diff.append(row)
    return diff

def flatten_landmarks(landmarks):
    """
    Given a list of landmark dictionaries (each with keys "x", "y", "z"),
    return a flattened list [x1, y1, z1, x2, y2, z2, ...].
    """
    return [v for pt in landmarks for v in (pt["x"], pt["y"], pt["z"])]

def augment_landmarks(landmarks, rotation_range=5, translation_range=0.05, 
                      scaling_range=0.1, shearing_range=5, noise_std=0.01, 
                      horizontal_flip_prob=0.5):
    """
    Apply geometric augmentations to landmarks:
      - Horizontal flip with probability horizontal_flip_prob.
      - Random Translation: Shifts each landmark by a small offset.
      - Random Rotation: Rotates landmarks by a small random angle (in degrees) about the centroid.
      - Random Scaling: Scales landmarks relative to the centroid.
      - Random Shearing: Applies a shear transformation.
      - Noise Injection: Adds Gaussian noise to each landmark.
    """
    # Copy landmarks to avoid modifying the original list
    augmented = [{"x": lm["x"], "y": lm["y"], "z": lm["z"]} for lm in landmarks]

    # Horizontal flip: flip x coordinate relative to 1.0 (assuming normalized coordinates)
    if np.random.rand() < horizontal_flip_prob:
        for lm in augmented:
            lm["x"] = 1.0 - lm["x"]

    # Random translation
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    for lm in augmented:
        lm["x"] += tx
        lm["y"] += ty

    # Compute centroid of the translated landmarks.
    cx = np.mean([lm["x"] for lm in augmented])
    cy = np.mean([lm["y"] for lm in augmented])

    # Random rotation
    angle_deg = np.random.uniform(-rotation_range, rotation_range)
    angle_rad = np.deg2rad(angle_deg)

    # Random scaling
    scale = np.random.uniform(1 - scaling_range, 1 + scaling_range)

    # Random shearing (horizontal shear)
    shear_deg = np.random.uniform(-shearing_range, shearing_range)
    shear = np.tan(np.deg2rad(shear_deg))

    # Construct transformation matrix M = scale * R with shearing adjustment.
    a = scale * np.cos(angle_rad)
    b = -scale * np.sin(angle_rad)
    c = scale * np.sin(angle_rad)
    d = scale * np.cos(angle_rad)
    M11 = a + shear * c
    M12 = b + shear * d
    M21 = c
    M22 = d

    # Apply affine transformation about the centroid and inject noise.
    for lm in augmented:
        x_shifted = lm["x"] - cx
        y_shifted = lm["y"] - cy
        x_trans = M11 * x_shifted + M12 * y_shifted
        y_trans = M21 * x_shifted + M22 * y_shifted
        lm["x"] = x_trans + cx + np.random.normal(0, noise_std)
        lm["y"] = y_trans + cy + np.random.normal(0, noise_std)
        lm["z"] = lm["z"] + np.random.normal(0, noise_std)
    
    return augmented

def process_frame(frame, holistic_detector):
    """
    Process a single frame using MediaPipe Holistic.
    Returns a dictionary with keys "left", "right", and "pose" mapping to the corresponding list of landmarks.
    If no landmarks are detected, returns None.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_detector.process(image_rgb)
    holistic_data = {}
    
    # Extract left-hand landmarks.
    if results.left_hand_landmarks:
        landmarks = []
        for lm in results.left_hand_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        holistic_data["left"] = landmarks

    # Extract right-hand landmarks.
    if results.right_hand_landmarks:
        landmarks = []
        for lm in results.right_hand_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        holistic_data["right"] = landmarks

    # Extract pose (body) landmarks.
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
        holistic_data["pose"] = landmarks

    return holistic_data if holistic_data else None

def generate_features(holistic_data, image_width, image_height, hand_weight=2.0):
    """
    Given a dictionary with keys "left", "right", and "pose" (each a list of landmarks),
    generate a combined feature vector.
    
    Process:
      - For each hand (left and right): flatten, normalize x/y to [-1,1],
        compute pairwise differences for x and y coordinates, and form a (21,4) tensor.
      - For pose: flatten 33 landmarks similarly to create a (33,4) tensor.
      - Multiply hand features by hand_weight to emphasize them.
      - Concatenate hand and pose features vertically and flatten.
    
    Returns a torch.Tensor of shape (300,) = ( (42+33)*4, ).
    """
    # Process hands
    expected_hand = 21
    left = holistic_data.get("left", None)
    right = holistic_data.get("right", None)
    if left is None:
        left = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_hand)]
    else:
        left = augment_landmarks(left)
    if right is None:
        right = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_hand)]
    else:
        right = augment_landmarks(right)
    
    left_flat = flatten_landmarks(left)   # 63 numbers
    right_flat = flatten_landmarks(right)   # 63 numbers

    # Extract and normalize x and y for each hand
    left_x = [left_flat[i] for i in range(0, len(left_flat), 3)]
    left_y = [left_flat[i] for i in range(1, len(left_flat), 3)]
    right_x = [right_flat[i] for i in range(0, len(right_flat), 3)]
    right_y = [right_flat[i] for i in range(1, len(right_flat), 3)]
    left_x = [2 * (x - 0.5) for x in left_x]
    left_y = [2 * (y - 0.5) for y in left_y]
    right_x = [2 * (x - 0.5) for x in right_x]
    right_y = [2 * (y - 0.5) for y in right_y]
    
    left_x_diff = np.mean(compute_difference(left_x), axis=1)
    left_y_diff = np.mean(compute_difference(left_y), axis=1)
    right_x_diff = np.mean(compute_difference(right_x), axis=1)
    right_y_diff = np.mean(compute_difference(right_y), axis=1)
    
    left_features = torch.stack([
        torch.tensor(left_x, dtype=torch.float32),
        torch.tensor(left_y, dtype=torch.float32),
        torch.tensor(left_x_diff, dtype=torch.float32),
        torch.tensor(left_y_diff, dtype=torch.float32)
    ], dim=1)  # (21, 4)
    
    right_features = torch.stack([
        torch.tensor(right_x, dtype=torch.float32),
        torch.tensor(right_y, dtype=torch.float32),
        torch.tensor(right_x_diff, dtype=torch.float32),
        torch.tensor(right_y_diff, dtype=torch.float32)
    ], dim=1)  # (21, 4)
    
    hands_features = torch.cat([left_features, right_features], dim=0)  # (42, 4)
    
    # Process pose landmarks (33 landmarks expected)
    expected_pose = 33
    pose = holistic_data.get("pose", None)
    if pose is None:
        pose = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_pose)]
    # For pose, we use no augmentation (or you could apply a lighter version if needed)
    pose_flat = flatten_landmarks(pose)  # 99 numbers
    
    pose_x = [pose_flat[i] for i in range(0, len(pose_flat), 3)]
    pose_y = [pose_flat[i] for i in range(1, len(pose_flat), 3)]
    pose_x = [2 * (x - 0.5) for x in pose_x]
    pose_y = [2 * (y - 0.5) for y in pose_y]
    pose_x_diff = np.mean(compute_difference(pose_x), axis=1)
    pose_y_diff = np.mean(compute_difference(pose_y), axis=1)
    
    pose_features = torch.stack([
        torch.tensor(pose_x, dtype=torch.float32),
        torch.tensor(pose_y, dtype=torch.float32),
        torch.tensor(pose_x_diff, dtype=torch.float32),
        torch.tensor(pose_y_diff, dtype=torch.float32)
    ], dim=1)  # (33, 4)
    
    # Emphasize hand features using hand_weight.
    hands_features *= hand_weight

    # Concatenate hand and pose features: final shape = (42+33, 4) = (75,4)
    final_features = torch.cat([hands_features, pose_features], dim=0)
    return final_features.flatten()  # 75*4 = 300 features

def process_video(video_path, output_dir, frame_drop_prob=0.1, hand_weight=2.0):
    """
    Process a single video file:
      - Extracts holistic landmarks (hands and pose) from each frame using MediaPipe Holistic.
      - Generates a combined feature vector per frame.
      - Applies temporal augmentation by randomly dropping frames.
      - Saves the features for all frames as a single .pt file.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_features.pt")
    
    # Initialize MediaPipe Holistic.
    mp_holistic = mp.solutions.holistic
    holistic_detector = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return
    features_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Temporal augmentation: randomly drop frames.
        if np.random.rand() < frame_drop_prob:
            continue

        H, W, _ = frame.shape
        holistic_data = process_frame(frame, holistic_detector)
        if holistic_data is not None:
            ft = generate_features(holistic_data, W, H, hand_weight=hand_weight)
            features_list.append(ft.unsqueeze(0))  # Add batch dimension.
        else:
            features_list.append(torch.zeros(300).unsqueeze(0))
    cap.release()
    holistic_detector.close()
    
    if features_list:
        features_tensor = torch.cat(features_list, dim=0)  # (num_frames, 300)
    else:
        features_tensor = torch.empty((0, 300))
    
    torch.save(features_tensor, save_path)
    logging.info(f"Saved features for video '{video_path}' to '{save_path}'")

def process_sorted_videos(input_root, output_root, frame_drop_prob=0.1, hand_weight=2.0):
    """
    Iterates through the sorted_videos folder, where each subfolder is a numeric label.
    For each video in each subfolder, extract features and save them.
    The output folder will mirror the input structure.
    If the output directory for a label already contains files, the label is skipped.
    """
    for label_folder in os.listdir(input_root):
        label_folder_path = os.path.join(input_root, label_folder)
        if not os.path.isdir(label_folder_path):
            continue
        
        output_label_dir = os.path.join(output_root, label_folder)
        # Skip processing if output folder exists and is non-empty.
        if os.path.exists(output_label_dir) and os.listdir(output_label_dir):
            logging.info(f"Skipping label folder '{label_folder}' because features already exist.")
            continue
        
        logging.info(f"Processing label folder: {label_folder}")
        os.makedirs(output_label_dir, exist_ok=True)
        for video_file in os.listdir(label_folder_path):
            if not video_file.lower().endswith('.mp4'):
                continue
            video_path = os.path.join(label_folder_path, video_file)
            logging.info(f"Processing video: {video_file} under label {label_folder}")
            process_video(video_path, output_label_dir, frame_drop_prob=frame_drop_prob, hand_weight=hand_weight)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract holistic landmarks (hands and pose) and features from videos with augmentations.")
    parser.add_argument('--input_root', type=str,
                        help='Root folder containing input videos.')
    parser.add_argument('--output_root', type=str,
                        help='Folder where generated features will be saved.')
    parser.add_argument('--frame_drop_prob', type=float, default=0.1,
                        help='Probability of dropping a frame for temporal augmentation.')
    parser.add_argument('--hand_weight', type=float, default=2.0,
                        help='Weight factor to emphasize hand features.')
    args = parser.parse_args()
    
    start = time.time()
    process_sorted_videos(args.input_root, args.output_root, frame_drop_prob=args.frame_drop_prob, hand_weight=args.hand_weight)
    elapsed = time.time() - start
    logging.info(f"Processing completed in {elapsed:.2f} seconds")