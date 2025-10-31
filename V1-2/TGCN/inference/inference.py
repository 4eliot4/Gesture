import os
import torch
import torch.nn.functional as F 
import torch.nn as nn
import logging
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import math
import argparse
import collections # For deque
import time
import asyncio, threading
from chatbot import chat_complete
from models.TGCN.tgcn_model import GCN_muti_att_Model
_bg_loop = asyncio.new_event_loop()
threading.Thread(target=_bg_loop.run_forever, daemon=True).start()


# --- Constants for Preprocessing (MUST MATCH TRAINING) ---
V_POSE, V_HAND = 33, 21
NUM_NODES_MODEL = V_POSE + V_HAND + V_HAND  # 75
RAW_FEATURES_PER_NODE_XYZ = 3

# Slicing constants for MediaPipe Holistic output (assuming full output)
MP_POSE_LANDMARKS = 33
MP_FACE_LANDMARKS = 468
MP_HAND_LANDMARKS = 21

# Normalization constants
POSE_CENTER_JOINTS = (11, 12) # mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, RIGHT_SHOULDER
HAND_SCALE_JOINT = 9 # Middle finger MCP, relative to hand landmarks
HAND_WRIST_IDX = 0   # Wrist, relative to hand landmarks

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')

# --- Preprocessing Functions (Adapted from NPYBasedDataset) ---
def _normalize_single_part_np_inference(pts_xyz_part, center_idx, s1_idx, s2_idx, scale_joints=True):
    """ Normalizes a single part for inference. pts_xyz_part: (T, Num_Joints_Part, 3) """
    if pts_xyz_part.ndim == 2: pts_xyz_part = pts_xyz_part[np.newaxis,...]
    T, Num_J, Dims = pts_xyz_part.shape
    if Dims != RAW_FEATURES_PER_NODE_XYZ:
        # If last dim is visibility for pose, make it Z=0
        if Dims == 3 and Num_J == V_POSE : # Check if it's pose and has 3rd dim
             temp_pts = np.zeros_like(pts_xyz_part)
             temp_pts[:,:,:2] = pts_xyz_part[:,:,:2]
             pts_xyz_part = temp_pts
        else:
            raise ValueError(f"_normalize_single_part_np expects {RAW_FEATURES_PER_NODE_XYZ} dims (x,y,z), got {Dims}")


    if isinstance(center_idx, tuple):
        c = (pts_xyz_part[:,center_idx[0],:] + pts_xyz_part[:,center_idx[1],:]) / 2.0
    else:
        c = pts_xyz_part[:,center_idx,:]
    
    pts_c = pts_xyz_part - c[:,None,:]
    
    if scale_joints:
        d = np.linalg.norm(pts_xyz_part[:,s1_idx,:] - pts_xyz_part[:,s2_idx,:], axis=-1)
        d_safe = np.where(d < 1e-6, 1.0, d)[:,None,None]
        return pts_c / d_safe
    else:
        return pts_c

def preprocess_live_frames(keypoint_frames_list, max_frames, use_normalizations=True, scale_joints=True):
    """
    Processes a list of keypoint dictionaries (one dict per frame) into a model-ready tensor.
    keypoint_frames_list: List of dicts, each {'pose': (33,3), 'lh': (21,3), 'rh': (21,3)}
                         Pose 3rd dim is visibility, Hands 3rd dim is z.
    Returns: Tensor of shape (max_frames, NUM_NODES_MODEL, features_per_node_final)
    """
    processed_frames_for_stacking = [] # Store NumPy arrays of shape (NUM_NODES_MODEL, features_per_node_final)
    
    features_per_node_final = (RAW_FEATURES_PER_NODE_XYZ * 2) if use_normalizations else RAW_FEATURES_PER_NODE_XYZ

    for kps_dict_single_frame in keypoint_frames_list: # kps_dict_single_frame is for ONE frame
        
        pose_mp_frame = kps_dict_single_frame.get('pose')
        if pose_mp_frame is None or not isinstance(pose_mp_frame, np.ndarray) or \
           pose_mp_frame.shape != (V_POSE, 3): # Expect (33, 3) for x,y,visibility
            pose_mp_frame = np.zeros((V_POSE, 3), dtype=np.float32)
        
        pose_xyz_raw_frame = np.zeros((V_POSE, RAW_FEATURES_PER_NODE_XYZ), dtype=np.float32)
        pose_xyz_raw_frame[:, :2] = pose_mp_frame[:, :2] # Copy x,y; z for pose remains 0

        lh_mp_frame = kps_dict_single_frame.get('lh')
        if lh_mp_frame is None or not isinstance(lh_mp_frame, np.ndarray) or \
           lh_mp_frame.shape != (V_HAND, RAW_FEATURES_PER_NODE_XYZ): # Expect (21, 3) for x,y,z
            # logging.debug("Missing/malformed LH for a frame, using zeros.")
            lh_mp_frame = np.zeros((V_HAND, RAW_FEATURES_PER_NODE_XYZ), dtype=np.float32)
        lh_xyz_raw_frame = lh_mp_frame # This is already (21,3) with x,y,z

        # --- RIGHT HAND Data for this frame ---
        rh_mp_frame = kps_dict_single_frame.get('rh')
        if rh_mp_frame is None or not isinstance(rh_mp_frame, np.ndarray) or \
           rh_mp_frame.shape != (V_HAND, RAW_FEATURES_PER_NODE_XYZ): # Expect (21, 3) for x,y,z
            # logging.debug("Missing/malformed RH for a frame, using zeros.")
            rh_mp_frame = np.zeros((V_HAND, RAW_FEATURES_PER_NODE_XYZ), dtype=np.float32)
        rh_xyz_raw_frame = rh_mp_frame # This is already (21,3) with x,y,z

        # --- Normalization (if enabled) for THIS FRAME ---
        if use_normalizations:
            # _normalize_single_part_np_inference expects (T, J, 3) input,
            # so we add a temporary time dimension (T=1) and then squeeze it out.
            pose_xyz_norm_frame = _normalize_single_part_np_inference(pose_xyz_raw_frame.copy()[np.newaxis, ...], POSE_CENTER_JOINTS, POSE_CENTER_JOINTS[0], POSE_CENTER_JOINTS[1], scale_joints).squeeze(0)
            lh_xyz_norm_frame   = _normalize_single_part_np_inference(lh_xyz_raw_frame.copy()[np.newaxis, ...], HAND_WRIST_IDX, HAND_WRIST_IDX, HAND_SCALE_JOINT, scale_joints).squeeze(0)
            rh_xyz_norm_frame   = _normalize_single_part_np_inference(rh_xyz_raw_frame.copy()[np.newaxis, ...], HAND_WRIST_IDX, HAND_WRIST_IDX, HAND_SCALE_JOINT, scale_joints).squeeze(0)

            pose_processed_frame = np.concatenate([pose_xyz_raw_frame, pose_xyz_norm_frame], axis=-1) # (33, 6)
            lh_processed_frame   = np.concatenate([lh_xyz_raw_frame,   lh_xyz_norm_frame],   axis=-1) # (21, 6)
            rh_processed_frame   = np.concatenate([rh_xyz_raw_frame,   rh_xyz_norm_frame],   axis=-1) # (21, 6)
        else:
            pose_processed_frame = pose_xyz_raw_frame # (33, 3)
            lh_processed_frame   = lh_xyz_raw_frame   # (21, 3)
            rh_processed_frame   = rh_xyz_raw_frame   # (21, 3)
            
        # Concatenate parts for THIS SINGLE FRAME: (NUM_NODES_MODEL, features_per_node_final)
        concatenated_single_frame_features = np.concatenate([pose_processed_frame, lh_processed_frame, rh_processed_frame], axis=0)
        processed_frames_for_stacking.append(concatenated_single_frame_features)

    if not processed_frames_for_stacking: # If keypoint_frames_list was empty
        return torch.zeros((max_frames, NUM_NODES_MODEL, features_per_node_final), dtype=torch.float32)

    # Stack all processed frames: (T_current, NUM_NODES_MODEL, features_per_node_final)
    # T_current will be len(keypoint_frames_list)
    video_data_np = np.stack(processed_frames_for_stacking, axis=0)
    video_tensor_processed = torch.from_numpy(video_data_np).float()

    # Temporal Padding/Sampling
    T_current = video_tensor_processed.shape[0]
    final_tensor_for_model = torch.zeros((max_frames, NUM_NODES_MODEL, features_per_node_final), dtype=video_tensor_processed.dtype)
    
    actual_frames_to_copy = min(T_current, max_frames)

    if T_current == 0: 
        pass # final_tensor_for_model is already zeros
    elif T_current < max_frames:
        final_tensor_for_model[:T_current, :, :] = video_tensor_processed[:T_current, :, :]
        # Pad with repetition of the last valid frame if T_current > 0
        if T_current > 0:
             final_tensor_for_model[T_current:, :, :] = video_tensor_processed[T_current-1:T_current, :, :].repeat(max_frames - T_current, 1, 1)
    elif T_current > max_frames: 
        # This case should ideally be prevented if keypoint_frames_list has maxlen=max_frames
        # If it occurs, linspace sampling is a good way to downsample.
        indices = np.linspace(0, T_current - 1, num=max_frames, dtype=int)
        final_tensor_for_model = video_tensor_processed[indices, :, :]
    else: # T_current == max_frames
        final_tensor_for_model = video_tensor_processed
        
    # Final shape check
    if final_tensor_for_model.shape != (max_frames, NUM_NODES_MODEL, features_per_node_final):
        logging.error(f"Shape mismatch for video after temporal processing. Got {final_tensor_for_model.shape}. Expected {(max_frames, NUM_NODES_MODEL, features_per_node_final)}. Returning zeros.")
        return torch.zeros((max_frames, NUM_NODES_MODEL, features_per_node_final), dtype=torch.float32)

    return final_tensor_for_model


def extract_holistic_from_frame(image_rgb, holistic_mp_instance):
    image_rgb.flags.writeable = False
    results = holistic_mp_instance.process(image_rgb)
    image_rgb.flags.writeable = True
    
    kps_dict = {'pose': None, 'lh': None, 'rh': None}
    if results.pose_landmarks:
        kps_dict['pose'] = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32)
    if results.left_hand_landmarks:
        kps_dict['lh'] = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
    if results.right_hand_landmarks:
        kps_dict['rh'] = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
    return kps_dict, results



def main(args):

    print("Main started")
    _chat_future = None
    inference_started = False  # NEW: State to control inference start
    # Use sentence_font_scale from args if added, otherwise default
    font_scale_sentence = getattr(args, 'sentence_font_scale', 1.5)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load Gloss Map ---
    if not os.path.exists(args.gloss_map):
        raise FileNotFoundError(f"Gloss map file not found: {args.gloss_map}")
    gloss_df = pd.read_csv(args.gloss_map)
    if 'label' not in gloss_df.columns or 'gloss' not in gloss_df.columns:
        raise ValueError(f"Gloss map {args.gloss_map} must contain 'label' and 'gloss' columns.")
    idx2g = {row['label']: row['gloss'] for _, row in gloss_df.iterrows()}
    num_model_classes = args.num_classes_trained
    # ... (rest of gloss map loading logic) ...
    logging.info(f"Loaded {gloss_df['label'].nunique()} glosses from map. Model configured for {num_model_classes} classes.")

    # --- Determine Model Input Features ---
    features_per_node_final = (RAW_FEATURES_PER_NODE_XYZ * 2) if args.use_normalizations else RAW_FEATURES_PER_NODE_XYZ
    model_input_feature_dim = args.max_frames * features_per_node_final
    logging.info(f"Model expects input_feature_dim (Time*Features per node): {model_input_feature_dim}")
    logging.info(f"Normalization for inference: {'Yes' if args.use_normalizations else 'No'}. Scaling: {'Yes' if args.scale_joints and args.use_normalizations else 'No'}")

    # --- Load Model ---
    model = GCN_muti_att_Model(
        input_feature_dim=model_input_feature_dim,
        hidden_feature_dim=args.hidden_feature,
        num_classes=num_model_classes,
        dropout_rate=args.dropout,
        num_gcn_stages=args.num_gcn_stages,
        num_graph_nodes=NUM_NODES_MODEL,
        pooling_type=args.pooling_type
    ).to(device)
    logging.info(f"Loading model weights from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except RuntimeError as e:
        logging.error(f"Error loading state_dict: {e}")
        logging.error("Ensure ALL model __init__ args match the saved checkpoint: "
                      "num_classes_trained, hidden_feature, dropout, num_gcn_stages, pooling_type.")
        return
    model.eval()
    logging.info("Model loaded successfully and set to evaluation mode.")

    logging.info("Model loaded successfully and set to evaluation mode.")

    # --- MediaPipe Holistic ---
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    holistic_mp = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.mp_model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise IOError(f"Cannot open camera {args.camera_id}")

    keypoint_frames_collector = []
    display_predictions = []
    pred_history = [] 
    
    accumulated_gloss_string_for_chatbot = "" 
    last_significant_gloss_sent = "" 
    numIters_chatbot_trigger = 0 
    completed_sentence_from_chatbot = "" 

    logging.info(f"Press 's' to start inference. 'r' to reset. 'q' to quit.")


    def reset_system_state():
        nonlocal numIters_chatbot_trigger, completed_sentence_from_chatbot, inference_started, \
               accumulated_gloss_string_for_chatbot, last_significant_gloss_sent, \
               display_predictions, keypoint_frames_collector, _chat_future
        
        numIters_chatbot_trigger = 0
        completed_sentence_from_chatbot = ""
        inference_started = False # Require 's' to start again
        accumulated_gloss_string_for_chatbot = ""
        last_significant_gloss_sent = ""
        display_predictions = []
        keypoint_frames_collector.clear()
        
        if _chat_future is not None and not _chat_future.done():
            _chat_future.cancel()
            logging.info("Cancelled pending chatbot request during reset.")
        _chat_future = None
        logging.info("System reset. Press 's' to start inference.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            logging.warning("Failed to grab frame from camera."); time.sleep(0.1); continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kps_dict_current_frame, holistic_results_mp = extract_holistic_from_frame(frame_rgb, holistic_mp)

        if inference_started:
            keypoint_frames_collector.append(kps_dict_current_frame)

            if len(keypoint_frames_collector) == args.max_frames:
                processed_keypoints_tensor = preprocess_live_frames(
                    keypoint_frames_collector,
                    args.max_frames,
                    args.use_normalizations,
                    args.scale_joints
                )
                input_to_permute = processed_keypoints_tensor.unsqueeze(0).to(device)
                bsz, T, N, F_node = input_to_permute.shape
                model_input_tensor = input_to_permute.permute(0, 2, 1, 3).contiguous().view(bsz, N, T * F_node)

                current_top_gloss = "" # The single most confident gloss for this chunk
                with torch.no_grad():
                    logits = model(model_input_tensor)
                    probabilities = F.softmax(logits.squeeze(0), dim=0).cpu()
                
                top_k_probs, top_k_indices = torch.topk(probabilities, args.top_k_display)
                
                display_predictions = [] # Update predictions for this new chunk
                if top_k_indices.numel() > 0:
                    # Get the top gloss for potential chatbot string
                    current_top_gloss = idx2g.get(top_k_indices[0].item(), f"L:{top_k_indices[0].item()}")
                    top_prob = top_k_probs[0].item()

                    for i in range(min(args.top_k_display, top_k_indices.numel())):
                        pred_idx = top_k_indices[i].item()
                        prob = top_k_probs[i].item()
                        gloss_name = idx2g.get(pred_idx, f"L:{pred_idx}")
                        display_predictions.append(f"{gloss_name}: {prob:.2f}")

                    confidence_threshold_for_chatbot = 0.3 # Tune this
                    if current_top_gloss and top_prob > confidence_threshold_for_chatbot and current_top_gloss != last_significant_gloss_sent:
                        if accumulated_gloss_string_for_chatbot: # Add space if not the first gloss
                            accumulated_gloss_string_for_chatbot += " "
                        accumulated_gloss_string_for_chatbot += current_top_gloss
                        pred_history.append([current_top_gloss]) # Log the recognized gloss
                        last_significant_gloss_sent = current_top_gloss
                        logging.debug(f"Added to chatbot string: '{current_top_gloss}'. Current: '{accumulated_gloss_string_for_chatbot}'")
                
                # --- Chatbot Trigger Logic ---
                numIters_chatbot_trigger += 1
                if numIters_chatbot_trigger % 8 == 0 and _chat_future is None and accumulated_gloss_string_for_chatbot:
                    logging.info(f"Sending to chatbot: '{accumulated_gloss_string_for_chatbot}'")
                    _chat_future = asyncio.run_coroutine_threadsafe(
                        chat_complete(accumulated_gloss_string_for_chatbot),
                        _bg_loop
                    )
                    accumulated_gloss_string_for_chatbot = "" # Clear after sending
                    last_significant_glogiss_sent = ""      # Allow next sign to be added immediately
                    completed_sentence_from_chatbot = ""

                # --- Check Chatbot Future ---
                if _chat_future is not None and _chat_future.done():
                    try:
                        response = _chat_future.result()
                        if completed_sentence_from_chatbot: completed_sentence_from_chatbot += " " # Append
                        completed_sentence_from_chatbot = response
                        logging.info(f"Chatbot response: '{response}'. Full sentence: '{completed_sentence_from_chatbot}'")
                    except Exception as e:
                        logging.error(f"ChatGPT error: {e}")
                        if completed_sentence_from_chatbot: completed_sentence_from_chatbot += " "
                        completed_sentence_from_chatbot += "[Chatbot Error]"
                    _chat_future = None
                
                # Discard some frames to create a refractory period after processing a chunk
                hold_counter = args.max_frames // 3 # e.g., if max_frames=60, discard 20
                keypoint_frames_collector = keypoint_frames_collector[hold_counter:]

        else: # inference_started is False
            keypoint_frames_collector.clear() # Don't accumulate frames before 's'
            display_predictions = []          # No predictions to show

        # --- Display ---
        annotated_image = frame_bgr.copy()
        if holistic_results_mp: # Always draw landmarks
            if holistic_results_mp.pose_landmarks: mp_drawing.draw_landmarks(annotated_image, holistic_results_mp.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if holistic_results_mp.left_hand_landmarks: mp_drawing.draw_landmarks(annotated_image, holistic_results_mp.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
            if holistic_results_mp.right_hand_landmarks: mp_drawing.draw_landmarks(annotated_image, holistic_results_mp.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

        # Display Top-K Predictions (if available and inference started)
        if inference_started and display_predictions:
            y_offset_preds = 30
            for pred_text_item in display_predictions:
                cv2.putText(annotated_image, pred_text_item, (10, 2*y_offset_preds), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                y_offset_preds += 22
        
        # Buffer Status Display
        buffer_text = f"Buffer: {len(keypoint_frames_collector)}/{args.max_frames}" if inference_started else "Buffer: Idle"
        (buf_w, buf_h), _ = cv2.getTextSize(buffer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (annotated_image.shape[1] - buf_w - 20, 5), (annotated_image.shape[1] - 5, buf_h + 15), (0,0,0), -1)
        cv2.putText(annotated_image, buffer_text, (annotated_image.shape[1] - buf_w - 10, buf_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # --- Main Sentence Display (with dynamic background) ---
        text_to_show_on_screen = ""
        if inference_started:
            if _chat_future is not None and not _chat_future.done():
                text_to_show_on_screen = "Prediction: Thinking..."
            elif completed_sentence_from_chatbot:
                text_to_show_on_screen = "Translation: " + completed_sentence_from_chatbot
            else:
                text_to_show_on_screen = "Status: Listening..."
        else:
            text_to_show_on_screen = "Press 's' to start. 'r' to reset. 'q' to quit."
        
        font_thickness_sentence = 2
        (text_width_main, text_height_main_above_baseline), baseline_offset_main = cv2.getTextSize(
            text_to_show_on_screen, cv2.FONT_HERSHEY_SIMPLEX, font_scale_sentence, font_thickness_sentence
        )

        frame_w_main = annotated_image.shape[1]
        frame_h_main = annotated_image.shape[0]

        # Position text near the bottom, centered
        text_origin_x_main = int((frame_w_main - text_width_main) / 2)
        text_baseline_y_main = int(frame_h_main - 25) # Baseline y-coordinate

        padding_h_main = 20  # Horizontal padding for background
        padding_v_main = 15  # Vertical padding for background

        # Calculate rectangle coordinates, ensuring they are within frame boundaries
        rect_tl_x_main = max(0, text_origin_x_main - padding_h_main)
        rect_tl_y_main = max(0, (text_baseline_y_main - text_height_main_above_baseline) - padding_v_main)
        rect_br_x_main = min(frame_w_main, (text_origin_x_main + text_width_main) + padding_h_main)
        rect_br_y_main = min(frame_h_main, (text_baseline_y_main + baseline_offset_main) + padding_v_main)

        if text_to_show_on_screen.strip(): # Draw background only if there's text
            cv2.rectangle(annotated_image, (rect_tl_x_main, rect_tl_y_main), (rect_br_x_main, rect_br_y_main), (0, 0, 0), thickness=-1)
        
        cv2.putText(
            annotated_image, text_to_show_on_screen, (text_origin_x_main, text_baseline_y_main),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale_sentence, (255, 255, 255), font_thickness_sentence, cv2.LINE_AA
        )
        # --- End Main Sentence Display ---

        cv2.imshow('Live Sign Language Prediction', annotated_image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            reset_system_state()
        if key == ord('s'):
            if not inference_started:
                inference_started = True
                # Reset relevant states when starting explicitly
                completed_sentence_from_chatbot = ""
                accumulated_gloss_string_for_chatbot = ""
                last_significant_gloss_sent = ""
                keypoint_frames_collector.clear() # Fresh buffer
                numIters_chatbot_trigger = 0
                logging.info("Inference started by 's' key press.")
            else:
                logging.info("Inference is already running.")

    cap.release()
    cv2.destroyAllWindows()
    if _bg_loop.is_running():
        _bg_loop.call_soon_threadsafe(_bg_loop.stop)
    time.sleep(0.5) # Give asyncio loop time to close

    if pred_history:
        try:
            with open("pred_history.txt", "w", encoding='utf-8') as f:
                for gloss_list_item in pred_history: # Each item in pred_history is a list like ['GLOSS']
                    f.write(f"{gloss_list_item[0]}\n") # Write the gloss itself
            logging.info("Prediction history saved to pred_history.txt")
        except Exception as e:
            logging.error(f"Failed to save prediction history: {e}")

    holistic_mp.close()
    logging.info("Inference stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live TGCN Inference Script (Chunk-based).")
    # Paths
    parser.add_argument('--checkpoint_path', default= "inference/TGCNModel.pt", help="Path to trained model checkpoint (.pth).")
    parser.add_argument('--gloss_map',  default= "inference/gloss_map.csv", help="Path to gloss map CSV (must have dense 0 to N-1 labels).")
    
    # Model Hyperparameters (MUST MATCH THE TRAINED MODEL'S CONFIGURATION)
    parser.add_argument('--num_classes_trained', default= 208, type=int, help="Number of classes the loaded model was trained on.")
    parser.add_argument('--hidden_feature', type=int, default=128, help="Hidden feature dimension in GCN blocks used during training.")
    parser.add_argument('--dropout', type=float, default=0.4, help="Dropout rate used during training for the model.")
    parser.add_argument('--num_gcn_stages', type=int, default=2, help="Number of GC_Block stages used during training.")
    parser.add_argument('--pooling_type', type=str, default='flatten', choices=['mean', 'flatten'], help="Pooling type used during training.")
    
    # Data Parameters (MUST MATCH TRAINING PREPROCESSING)
    # max_frames is the CHUNK_SIZE for this script
    parser.add_argument('--max_frames', type=int, default=64, help="Number of frames per chunk for inference (must match model's training max_frames).")
    parser.add_argument('--use_normalizations', action=argparse.BooleanOptionalAction, default=True, help="If model was trained with normalized features (6ch).")
    parser.add_argument('--scale_joints', action=argparse.BooleanOptionalAction, default=True, help="If model was trained with scaling in normalization.")
    
    # Inference Specific
    parser.add_argument('--camera_id', type=int, default=0, help="Camera ID.")
    parser.add_argument('--device', default='cuda', help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--top_k_display', type=int, default=5, help="Number of top predictions to display.")
    parser.add_argument('--mp_model_complexity', type=int, default=1, choices=[0,1,2], help="MediaPipe Holistic model complexity.")
        
    args = parser.parse_args()
    main(args)