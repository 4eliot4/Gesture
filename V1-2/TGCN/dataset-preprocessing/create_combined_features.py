#!/usr/bin/env python3
import os
import cv2
import time
import logging
import torch
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ------------- Helper Functions -------------
def compute_difference(x_list):
    """Compute pairwise differences among elements of x_list."""
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

def robust_augment_landmarks(landmarks, rotation_range=5, translation_range=0.05, 
                             scaling_range=0.1, shearing_range=5, noise_std=0.01, 
                             horizontal_flip_prob=0.3, perspective_prob=0.3, occlusion_prob=0.1):
    """
    Apply a robust set of geometric augmentations to landmarks:
      - Conditional horizontal flip with reduced probability.
      - Random translation, rotation, scaling, and shearing.
      - Optional perspective transformation.
      - Occlusion simulation by randomly zeroing out a few landmarks.
      - Noise injection.
    """
    augmented = [{"x": lm["x"], "y": lm["y"], "z": lm["z"]} for lm in landmarks]
    if np.random.rand() < horizontal_flip_prob:
        for lm in augmented:
            lm["x"] = 1.0 - lm["x"]
    tx = np.random.uniform(-translation_range, translation_range)
    ty = np.random.uniform(-translation_range, translation_range)
    for lm in augmented:
        lm["x"] += tx
        lm["y"] += ty
    cx = np.mean([lm["x"] for lm in augmented])
    cy = np.mean([lm["y"] for lm in augmented])
    angle_deg = np.random.uniform(-rotation_range, rotation_range)
    angle_rad = np.deg2rad(angle_deg)
    scale = np.random.uniform(1 - scaling_range, 1 + scaling_range)
    shear_deg = np.random.uniform(-shearing_range, shearing_range)
    shear = np.tan(np.deg2rad(shear_deg))
    a = scale * np.cos(angle_rad)
    b = -scale * np.sin(angle_rad)
    c = scale * np.sin(angle_rad)
    d = scale * np.cos(angle_rad)
    M11 = a + shear * c
    M12 = b + shear * d
    M21 = c
    M22 = d
    persp_shift = np.random.uniform(-0.02, 0.02) if np.random.rand() < perspective_prob else 0.0
    for lm in augmented:
        x_shifted = lm["x"] - cx
        y_shifted = lm["y"] - cy
        x_trans = M11 * x_shifted + M12 * y_shifted + persp_shift * x_shifted
        y_trans = M21 * x_shifted + M22 * y_shifted + persp_shift * y_shifted
        lm["x"] = x_trans + cx + np.random.normal(0, noise_std)
        lm["y"] = y_trans + cy + np.random.normal(0, noise_std)
        lm["z"] = lm["z"] + np.random.normal(0, noise_std)
    if np.random.rand() < occlusion_prob:
        num_occlusions = np.random.randint(1, max(2, len(augmented)//10))
        indices = np.random.choice(len(augmented), num_occlusions, replace=False)
        for idx in indices:
            augmented[idx] = {"x": 0.0, "y": 0.0, "z": 0.0}
    return augmented

def process_frame(frame, holistic_detector):
    """
    Process a single frame using MediaPipe Holistic.
    Returns a dictionary with keys "left", "right", and "pose" mapping to their landmarks.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_detector.process(image_rgb)
    holistic_data = {}
    if results.left_hand_landmarks:
        landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} 
                     for lm in results.left_hand_landmarks.landmark]
        holistic_data["left"] = landmarks
    if results.right_hand_landmarks:
        landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} 
                     for lm in results.right_hand_landmarks.landmark]
        holistic_data["right"] = landmarks
    if results.pose_landmarks:
        landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} 
                     for lm in results.pose_landmarks.landmark]
        holistic_data["pose"] = landmarks
    return holistic_data if holistic_data else None

def generate_features(holistic_data, image_width, image_height):
    """
    Generate augmented features (300-dimensional vector) from the holistic data.
    """
    expected_hand = 21
    left = holistic_data.get("left", None)
    right = holistic_data.get("right", None)
    if left is None:
        left = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_hand)]
    else:
        left = robust_augment_landmarks(left)
    if right is None:
        right = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_hand)]
    else:
        right = robust_augment_landmarks(right)
    
    left_flat = flatten_landmarks(left)
    right_flat = flatten_landmarks(right)
    left_x = [2 * (left_flat[i] - 0.5) for i in range(0, len(left_flat), 3)]
    left_y = [2 * (left_flat[i+1] - 0.5) for i in range(0, len(left_flat), 3)]
    right_x = [2 * (right_flat[i] - 0.5) for i in range(0, len(right_flat), 3)]
    right_y = [2 * (right_flat[i+1] - 0.5) for i in range(0, len(right_flat), 3)]
    left_x_diff = np.mean(compute_difference(left_x), axis=1)
    left_y_diff = np.mean(compute_difference(left_y), axis=1)
    right_x_diff = np.mean(compute_difference(right_x), axis=1)
    right_y_diff = np.mean(compute_difference(right_y), axis=1)
    left_features = torch.stack([
        torch.tensor(left_x, dtype=torch.float32),
        torch.tensor(left_y, dtype=torch.float32),
        torch.tensor(left_x_diff, dtype=torch.float32),
        torch.tensor(left_y_diff, dtype=torch.float32)
    ], dim=1)
    right_features = torch.stack([
        torch.tensor(right_x, dtype=torch.float32),
        torch.tensor(right_y, dtype=torch.float32),
        torch.tensor(right_x_diff, dtype=torch.float32),
        torch.tensor(right_y_diff, dtype=torch.float32)
    ], dim=1)
    hands_features = torch.cat([left_features, right_features], dim=0)
    
    expected_pose = 33
    pose = holistic_data.get("pose", None)
    if pose is None:
        pose = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_pose)]
    pose_flat = flatten_landmarks(pose)
    pose_x = [2 * (pose_flat[i] - 0.5) for i in range(0, len(pose_flat), 3)]
    pose_y = [2 * (pose_flat[i+1] - 0.5) for i in range(0, len(pose_flat), 3)]
    pose_x_diff = np.mean(compute_difference(pose_x), axis=1)
    pose_y_diff = np.mean(compute_difference(pose_y), axis=1)
    pose_features = torch.stack([
        torch.tensor(pose_x, dtype=torch.float32),
        torch.tensor(pose_y, dtype=torch.float32),
        torch.tensor(pose_x_diff, dtype=torch.float32),
        torch.tensor(pose_y_diff, dtype=torch.float32)
    ], dim=1)
    final_features = torch.cat([hands_features, pose_features], dim=0)
    return final_features.flatten()

def generate_raw_features(holistic_data, image_width, image_height):
    """
    Generate raw (non-augmented) features as a 300-dimensional vector.
    This follows the same procedure as generate_features but skips augmentation.
    """
    expected_hand = 21
    left = holistic_data.get("left", None)
    right = holistic_data.get("right", None)
    if left is None:
        left = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_hand)]
    if right is None:
        right = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_hand)]
    
    left_flat = flatten_landmarks(left)
    right_flat = flatten_landmarks(right)
    left_x = [2 * (left_flat[i] - 0.5) for i in range(0, len(left_flat), 3)]
    left_y = [2 * (left_flat[i+1] - 0.5) for i in range(0, len(left_flat), 3)]
    right_x = [2 * (right_flat[i] - 0.5) for i in range(0, len(right_flat), 3)]
    right_y = [2 * (right_flat[i+1] - 0.5) for i in range(0, len(right_flat), 3)]
    left_x_diff = np.mean(compute_difference(left_x), axis=1)
    left_y_diff = np.mean(compute_difference(left_y), axis=1)
    right_x_diff = np.mean(compute_difference(right_x), axis=1)
    right_y_diff = np.mean(compute_difference(right_y), axis=1)
    left_features = torch.stack([
        torch.tensor(left_x, dtype=torch.float32),
        torch.tensor(left_y, dtype=torch.float32),
        torch.tensor(left_x_diff, dtype=torch.float32),
        torch.tensor(left_y_diff, dtype=torch.float32)
    ], dim=1)
    right_features = torch.stack([
        torch.tensor(right_x, dtype=torch.float32),
        torch.tensor(right_y, dtype=torch.float32),
        torch.tensor(right_x_diff, dtype=torch.float32),
        torch.tensor(right_y_diff, dtype=torch.float32)
    ], dim=1)
    hands_features = torch.cat([left_features, right_features], dim=0)
    
    expected_pose = 33
    pose = holistic_data.get("pose", None)
    if pose is None:
        pose = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(expected_pose)]
    pose_flat = flatten_landmarks(pose)
    pose_x = [2 * (pose_flat[i] - 0.5) for i in range(0, len(pose_flat), 3)]
    pose_y = [2 * (pose_flat[i+1] - 0.5) for i in range(0, len(pose_flat), 3)]
    pose_x_diff = np.mean(compute_difference(pose_x), axis=1)
    pose_y_diff = np.mean(compute_difference(pose_y), axis=1)
    pose_features = torch.stack([
        torch.tensor(pose_x, dtype=torch.float32),
        torch.tensor(pose_y, dtype=torch.float32),
        torch.tensor(pose_x_diff, dtype=torch.float32),
        torch.tensor(pose_y_diff, dtype=torch.float32)
    ], dim=1)
    final_features = torch.cat([hands_features, pose_features], dim=0)
    return final_features.flatten()

def process_video(video_path, frame_drop_prob=0.1):
    """
    Process a single video file, computing features per frame.
    Returns a tensor of shape (T, 600) where T is the number of frames processed.
    The 600 dimensions come from concatenating a 300-d vector (features)
    and its temporal difference (another 300-d vector) per frame.
    """
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
        return torch.empty((0,300))
    feats_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if np.random.rand() < frame_drop_prob:
            continue
        H, W, _ = frame.shape
        holistic_data = process_frame(frame, holistic_detector)
        if holistic_data is not None:
            feat = generate_features(holistic_data, W, H)
            feats_list.append(feat.unsqueeze(0))
        else:
            feats_list.append(torch.zeros(300).unsqueeze(0))
    cap.release()
    holistic_detector.close()
    if feats_list:
        feats = torch.cat(feats_list, dim=0)  # (T, 300)
    else:
        feats = torch.empty((0,300))
    if feats.shape[0] > 1:
        delta = feats[1:] - feats[:-1]
        delta = torch.cat([torch.zeros(1,300), delta], dim=0)
    else:
        delta = torch.zeros_like(feats)
    combined = torch.cat([feats, delta], dim=1)  # (T, 600)
    return combined

def process_video_raw(video_path, frame_drop_prob=0.1):
    """
    Process a video file using the raw feature extraction pipeline (no augmentation).
    Returns a tensor of shape (T, 600) with raw features concatenated with temporal differences.
    """
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
        return torch.empty((0,300))
    feats_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if np.random.rand() < frame_drop_prob:
            continue
        H, W, _ = frame.shape
        holistic_data = process_frame(frame, holistic_detector)
        if holistic_data is not None:
            feat = generate_raw_features(holistic_data, W, H)
            feats_list.append(feat.unsqueeze(0))
        else:
            feats_list.append(torch.zeros(300).unsqueeze(0))
    cap.release()
    holistic_detector.close()
    if feats_list:
        feats = torch.cat(feats_list, dim=0)
    else:
        feats = torch.empty((0,300))
    if feats.shape[0] > 1:
        delta = feats[1:] - feats[:-1]
        delta = torch.cat([torch.zeros(1,300), delta], dim=0)
    else:
        delta = torch.zeros_like(feats)
    combined = torch.cat([feats, delta], dim=1)  # (T, 600)
    return combined

def combine_augmented_and_raw(aug_features_root, videos_root, output_root, frame_drop_prob=0.1):
    """
    For each augmented feature file in aug_features_root, this function computes the raw features
    from the corresponding video (using the raw pipeline), and then saves a dictionary in the output_root
    with two keys: "augmented" and "raw".
    
    It assumes that the video file names are the augmented base names with the trailing '_features' removed,
    plus a '.mp4' extension.
    """
    os.makedirs(output_root, exist_ok=True)
    for label in os.listdir(aug_features_root):
        aug_label_dir = os.path.join(aug_features_root, label)
        video_label_dir = os.path.join(videos_root, label)
        out_label_dir = os.path.join(output_root, label)
        if not os.path.isdir(aug_label_dir):
            continue
        os.makedirs(out_label_dir, exist_ok=True)
        for file in os.listdir(aug_label_dir):
            if not file.endswith("_features.pt"):
                continue
            aug_file_path = os.path.join(aug_label_dir, file)
            # Remove trailing '_features' from file base name to get video name.
            base_name = os.path.splitext(file)[0]
            if base_name.endswith("_features"):
                base_name = base_name[:-len("_features")]
            video_file = base_name + ".mp4"
            video_path = os.path.join(video_label_dir, video_file)
            
            logging.info(f"Processing {aug_file_path}")
            try:
                aug_features = torch.load(aug_file_path)
            except Exception as e:
                logging.error(f"Error loading augmented features from {aug_file_path}: {e}")
                continue
            # Compute raw features if the video file exists.
            if os.path.exists(video_path):
                raw_features = process_video_raw(video_path, frame_drop_prob=frame_drop_prob)
            else:
                logging.error(f"Video file {video_path} not found for {aug_file_path}.")
                raw_features = None
            combined = {"augmented": aug_features, "raw": raw_features}
            out_file_path = os.path.join(out_label_dir, file)
            torch.save(combined, out_file_path)
            logging.info(f"Saved combined features to {out_file_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Combine the augmented features with raw features (recomputed from videos) into one .pt file per sample."
    )
    parser.add_argument('--aug_root', type=str, default='./features_out',
                        help='Directory containing augmented feature .pt files.')
    parser.add_argument('--videos_root', type=str, default='./sorted_videos',
                        help='Directory containing original video files.')
    parser.add_argument('--output_root', type=str, default='./combined_features',
                        help='Directory to save combined feature files.')
    parser.add_argument('--frame_drop_prob', type=float, default=0.1,
                        help='Frame drop probability used during raw feature extraction.')
    args = parser.parse_args()
    
    start = time.time()
    combine_augmented_and_raw(args.aug_root, args.videos_root, args.output_root, frame_drop_prob=args.frame_drop_prob)
    elapsed = time.time() - start
    logging.info(f"Processing completed in {elapsed:.2f} seconds")