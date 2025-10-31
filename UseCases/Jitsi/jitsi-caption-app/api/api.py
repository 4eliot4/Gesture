# Flask imports
import time
from flask import Flask, request, jsonify
from flask import make_response
from flask_cors import CORS
import logging
import os
import asyncio
import collections # For deque if we choose sliding window later
import threading # To prevent race conditions on shared state (optional but good practice)

# Processing imports
import numpy as np
import io
from PIL import Image
import cv2 # Needed for color conversion
from chatbot import chat_complete

# Inference imports (New Model)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pandas as pd
import mediapipe as mp

# --- Configuration Constants (Replace with your actual values) ---
CHECKPOINT_PATH = './bestNewVers.pt' # IMPORTANT: Path to your trained .pth file
GLOSS_MAP_PATH = './gloss_map_211.csv' # IMPORTANT: Path to your gloss map CSV
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Hyperparameters (MUST MATCH TRAINING)
NUM_CLASSES_TRAINED = 211 # IMPORTANT: Set correctly for your WLASL-2000 model
HIDDEN_FEATURE_DIM = 128   # Example, match training
DROPOUT_RATE = 0.4       # Example, match training
NUM_GCN_STAGES = 2         # Example, match training
POOLING_TYPE = 'flatten'   # Example, match training

# Preprocessing Parameters (MUST MATCH TRAINING)
MAX_FRAMES = 64            # IMPORTANT: Sequence length model expects
USE_NORMALIZATIONS = True  # Example, match training
SCALE_JOINTS = True        # Example, match training

# Inference Specific
TOP_K_PREDICTIONS = 1     # How many top predictions to return
MP_MODEL_COMPLEXITY = 1   # MediaPipe Holistic complexity

# --- Model Definitions (Copied from your script) ---
# (Ensure these classes are identical to your training script)
class GraphConvolution_att(nn.Module):
    def __init__(self, in_features, out_features, num_nodes=75, bias=True):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features; self.out_features = out_features
        self.num_nodes = num_nodes
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if bias: self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else: self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.att.data, gain=nn.init.calculate_gain('tanh'))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, input_tensor):
        support = torch.matmul(input_tensor, self.weight)
        output = torch.matmul(self.att, support)
        return output + self.bias if self.bias is not None else output
    def __repr__(self): return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features}, num_nodes={self.num_nodes})"

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, num_nodes=75, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features=in_features; self.out_features=in_features; self.is_resi=is_resi; self.num_nodes=num_nodes
        self.gc1 = GraphConvolution_att(in_features,in_features,num_nodes=num_nodes,bias=bias)
        self.bn1 = nn.BatchNorm1d(num_nodes * in_features)
        self.gc2 = GraphConvolution_att(in_features,in_features,num_nodes=num_nodes,bias=bias)
        self.bn2 = nn.BatchNorm1d(num_nodes * in_features)
        self.do = nn.Dropout(p_dropout); self.act_f = nn.Tanh()
    def forward(self, x):
        y = self.gc1(x); b,n,f = y.shape; y = self.bn1(y.view(b,-1)).view(b,n,f); y = self.act_f(y); y = self.do(y)
        y = self.gc2(y); b,n,f = y.shape; y = self.bn2(y.view(b,-1)).view(b,n,f); y = self.act_f(y); y = self.do(y)
        return y + x if self.is_resi else y
    def __repr__(self): return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features}, num_nodes={self.num_nodes})"

class GCN_muti_att_Model(nn.Module):
    def __init__(self, input_feature_dim, hidden_feature_dim, num_classes, dropout_rate,
                 num_gcn_stages=1, use_residual_gcn=True, num_graph_nodes=75, pooling_type='flatten'):
        super(GCN_muti_att_Model, self).__init__()
        self.num_graph_nodes = num_graph_nodes
        self.pooling_type = pooling_type.lower()
        self.gc1 = GraphConvolution_att(input_feature_dim, hidden_feature_dim, num_nodes=num_graph_nodes)
        self.bn1 = nn.BatchNorm1d(num_graph_nodes * hidden_feature_dim)
        self.gc_blocks = nn.ModuleList()
        for _ in range(num_gcn_stages):
            self.gc_blocks.append(GC_Block(hidden_feature_dim, p_dropout=dropout_rate,
                                           num_nodes=num_graph_nodes, is_resi=use_residual_gcn))
        self.dropout_after_gcn = nn.Dropout(dropout_rate)
        self.activation_after_gcn = nn.Tanh()
        if self.pooling_type == 'mean': self.fc_out = nn.Linear(hidden_feature_dim, num_classes)
        elif self.pooling_type == 'flatten': self.fc_out = nn.Linear(hidden_feature_dim * num_graph_nodes, num_classes)
        else: raise ValueError(f"Unsupported pooling_type: {pooling_type}")
    def forward(self, x):
        b, n, _ = x.shape
        y = self.gc1(x); y = self.bn1(y.view(b, -1)).view(b, n, -1)
        y = self.activation_after_gcn(y); y = self.dropout_after_gcn(y)
        for block in self.gc_blocks: y = block(y)
        out_pooled = torch.mean(y, dim=1) if self.pooling_type == 'mean' else y.contiguous().view(b, -1)
        return self.fc_out(out_pooled)
# --- End Model Definitions ---

# --- Constants for Preprocessing (Copied from your script) ---
V_POSE, V_HAND = 33, 21
NUM_NODES_MODEL = V_POSE + V_HAND + V_HAND  # 75
RAW_FEATURES_PER_NODE_XYZ = 3
POSE_CENTER_JOINTS = (11, 12)
HAND_SCALE_JOINT = 9
HAND_WRIST_IDX = 0
FEATURES_PER_NODE_FINAL = (RAW_FEATURES_PER_NODE_XYZ * 2) if USE_NORMALIZATIONS else RAW_FEATURES_PER_NODE_XYZ
MODEL_INPUT_FEATURE_DIM = MAX_FRAMES * FEATURES_PER_NODE_FINAL # Adjusted: This is per-node feature dim * time

# --- Preprocessing Functions (Copied & adapted slightly) ---
def _normalize_single_part_np_inference(pts_xyz_part, center_idx, s1_idx, s2_idx, scale_joints=True):
    """ Normalizes a single part for inference. pts_xyz_part: (T, Num_Joints_Part, 3) """
    if pts_xyz_part.ndim == 2: pts_xyz_part = pts_xyz_part[np.newaxis,...]
    T, Num_J, Dims = pts_xyz_part.shape
    if Dims != RAW_FEATURES_PER_NODE_XYZ:
        if Dims == 3 and Num_J == V_POSE :
             temp_pts = np.zeros_like(pts_xyz_part)
             temp_pts[:,:,:2] = pts_xyz_part[:,:,:2] # Use x, y from visibility
             pts_xyz_part = temp_pts
        else:
            # Log error instead of raising for API robustness
            logging.error(f"_normalize_single_part_np expects {RAW_FEATURES_PER_NODE_XYZ} dims (x,y,z), got {Dims}. Using zeros.")
            return np.zeros_like(pts_xyz_part)

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
    """ Processes a list of keypoint dictionaries into a model-ready tensor. """
    processed_frames_for_stacking = []
    features_per_node_final = FEATURES_PER_NODE_FINAL # Use global constant

    for kps_dict_single_frame in keypoint_frames_list: # Process frame by frame
        # POSE Data
        pose_mp_frame = kps_dict_single_frame.get('pose')
        if pose_mp_frame is None or not isinstance(pose_mp_frame, np.ndarray) or pose_mp_frame.shape != (V_POSE, 3):
            pose_mp_frame = np.zeros((V_POSE, 3), dtype=np.float32)
        pose_xyz_raw_frame = np.zeros((V_POSE, RAW_FEATURES_PER_NODE_XYZ), dtype=np.float32)
        pose_xyz_raw_frame[:, :2] = pose_mp_frame[:, :2] # x,y from MP, z is 0

        # LEFT HAND Data
        lh_mp_frame = kps_dict_single_frame.get('lh')
        if lh_mp_frame is None or not isinstance(lh_mp_frame, np.ndarray) or lh_mp_frame.shape != (V_HAND, RAW_FEATURES_PER_NODE_XYZ):
            lh_mp_frame = np.zeros((V_HAND, RAW_FEATURES_PER_NODE_XYZ), dtype=np.float32)
        lh_xyz_raw_frame = lh_mp_frame # Already (21, 3) with x,y,z

        # RIGHT HAND Data
        rh_mp_frame = kps_dict_single_frame.get('rh')
        if rh_mp_frame is None or not isinstance(rh_mp_frame, np.ndarray) or rh_mp_frame.shape != (V_HAND, RAW_FEATURES_PER_NODE_XYZ):
            rh_mp_frame = np.zeros((V_HAND, RAW_FEATURES_PER_NODE_XYZ), dtype=np.float32)
        rh_xyz_raw_frame = rh_mp_frame # Already (21, 3) with x,y,z

        # Normalization (if enabled)
        if use_normalizations:
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

        # Concatenate parts for THIS SINGLE FRAME
        concatenated_single_frame_features = np.concatenate([pose_processed_frame, lh_processed_frame, rh_processed_frame], axis=0)
        processed_frames_for_stacking.append(concatenated_single_frame_features)

    if not processed_frames_for_stacking: # Should not happen if list is not empty
        return torch.zeros((max_frames, NUM_NODES_MODEL, features_per_node_final), dtype=torch.float32)

    # Stack frames: (T_current, NUM_NODES_MODEL, features_per_node_final)
    video_data_np = np.stack(processed_frames_for_stacking, axis=0)
    video_tensor_processed = torch.from_numpy(video_data_np).float()

    # Temporal Padding/Sampling (exactly as in the live script)
    T_current = video_tensor_processed.shape[0]
    final_tensor_for_model = torch.zeros((max_frames, NUM_NODES_MODEL, features_per_node_final), dtype=video_tensor_processed.dtype)
    actual_frames_to_copy = min(T_current, max_frames)

    if T_current == 0: pass # Already zeros
    elif T_current < max_frames:
        final_tensor_for_model[:T_current, :, :] = video_tensor_processed[:T_current, :, :]
        if T_current > 0: # Pad with last frame
             final_tensor_for_model[T_current:, :, :] = video_tensor_processed[T_current-1:T_current, :, :].repeat(max_frames - T_current, 1, 1)
    elif T_current > max_frames: # Linspace sampling if buffer exceeds max (shouldn't if we clear)
        indices = np.linspace(0, T_current - 1, num=max_frames, dtype=int)
        final_tensor_for_model = video_tensor_processed[indices, :, :]
    else: # T_current == max_frames
        final_tensor_for_model = video_tensor_processed

    # Final shape check
    expected_shape = (max_frames, NUM_NODES_MODEL, features_per_node_final)
    if final_tensor_for_model.shape != expected_shape:
        logging.error(f"Shape mismatch after temporal processing. Got {final_tensor_for_model.shape}. Expected {expected_shape}. Returning zeros.")
        return torch.zeros(expected_shape, dtype=torch.float32)

    return final_tensor_for_model

def extract_holistic_from_frame(image_rgb):
    """Extract MP Holistic keypoints from a single RGB frame in one-shot (static_image_mode) mode."""
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=True,               # process each frame independently
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True

    kps = {'pose': None, 'lh': None, 'rh': None}
    if results.pose_landmarks:
        kps['pose'] = np.array(
            [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )
    if results.left_hand_landmarks:
        kps['lh'] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )
    if results.right_hand_landmarks:
        kps['rh'] = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )
    return kps

# --- Global Variables / State ---
app = Flask(__name__)

CORS(app)
# after CORS(app):
@app.after_request
def _add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin']  = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return response

# handle the OPTIONS preflight yourself so it never 405’s or 403’s
@app.route('/frames', methods=['OPTIONS'])
def _frames_preflight():
    # make_response() uses 200 OK by default
    return make_response()

logging.basicConfig(level=logging.INFO)

# Inference components (initialized at startup)
model = None
holistic_mp = None
idx2gloss = {}

# State for frame collection and prediction
keypoint_buffer = []
last_prediction = {"status": "Initializing..."}
buffer_lock = threading.Lock() # To manage access to buffer/prediction

# --- Initialization Function ---
def initialize_inference():
    global model, holistic_mp, idx2gloss, last_prediction
    logging.info(f"Initializing inference components...")
    logging.info(f"Using device: {DEVICE}")

    # Load Gloss Map
    if not os.path.exists(GLOSS_MAP_PATH):
        raise FileNotFoundError(f"Gloss map file not found: {GLOSS_MAP_PATH}")
    gloss_df = pd.read_csv(GLOSS_MAP_PATH)
    if 'label' not in gloss_df.columns or 'gloss' not in gloss_df.columns:
        raise ValueError(f"Gloss map {GLOSS_MAP_PATH} must contain 'label' and 'gloss' columns.")
    # Create mapping from index (label) to gloss string
    idx2gloss = {row['label']: row['gloss'] for _, row in gloss_df.iterrows()}
    num_classes_from_map = len(idx2gloss)
    if NUM_CLASSES_TRAINED != num_classes_from_map:
         logging.warning(f"Num classes from gloss map ({num_classes_from_map}) != "
                        f"num_classes_trained param ({NUM_CLASSES_TRAINED}). Ensure consistency.")
    logging.info(f"Loaded {num_classes_from_map} glosses.")

    # Load Model
    logging.info(f"Loading model from: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")

    model = GCN_muti_att_Model(
        input_feature_dim=MODEL_INPUT_FEATURE_DIM, # Corrected dim calculation
        hidden_feature_dim=HIDDEN_FEATURE_DIM,
        num_classes=NUM_CLASSES_TRAINED, # Use the parameter passed
        dropout_rate=DROPOUT_RATE,
        num_gcn_stages=NUM_GCN_STAGES,
        num_graph_nodes=NUM_NODES_MODEL,
        pooling_type=POOLING_TYPE
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        model.eval() # Set to evaluation mode
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}")
        logging.error("Ensure checkpoint exists and hyperparameters match training.")
        raise e # Stop server if model fails to load

    # Initialize MediaPipe Holistic with ImmediateInputStreamHandler
    logging.info("Initializing MediaPipe Holistic (with ImmediateInputStreamHandler)...")
    mp_holistic_solution = mp.solutions.holistic
    holistic_mp = mp_holistic_solution.Holistic(
        static_image_mode=True,
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    logging.info("MediaPipe Holistic initialized.")
    logging.info("MediaPipe Holistic initialized.")
    last_prediction = {"status": "Ready, collecting frames..."}
    logging.info("Initialization complete.")

# --- Flask Routes ---
@app.route('/time') # Test route
def get_current_time():
    return {'time': time.time()}

_chat_future = None
accumulated_gloss = ""
last_significant_gloss = ""
iterations = 0
completed_translation = ""
pred_history = []
_bg_loop = asyncio.new_event_loop()
threading.Thread(target=_bg_loop.run_forever, daemon=True).start()

@app.route('/frames', methods=['POST'])
def process_video_frame():
    global keypoint_buffer, last_prediction
    global accumulated_gloss, last_significant_gloss, iterations, completed_translation, pred_history

    # 1. Read & decode the incoming frame
    if 'frame' not in request.files:
        return jsonify({"error": "No frame file found"}), 400
    file = request.files['frame']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Error decoding image: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    # 2. Extract keypoints in one‐shot mode
    kps = extract_holistic_from_frame(img_np)  # your static_image_mode helper

    with buffer_lock:
        keypoint_buffer.append(kps)

        if len(keypoint_buffer) < MAX_FRAMES:
            last_prediction = {
                "status": f"Collecting frames: {len(keypoint_buffer)}/{MAX_FRAMES}"
            }
        else:
            # Run inference on a full buffer
            try:
                buffer_copy = list(keypoint_buffer)
                proc = preprocess_live_frames(buffer_copy, MAX_FRAMES,
                                             USE_NORMALIZATIONS, SCALE_JOINTS)
                T, N, F_node = proc.shape

                inp = proc.unsqueeze(0).permute(0,2,1,3).reshape(1, N, T*F_node).to(DEVICE)
                with torch.no_grad():
                    logit = model(inp).squeeze(0)
                    probs = F.softmax(logit, dim=0).cpu()
                topv, topi = torch.topk(probs, TOP_K_PREDICTIONS)

                # accumulate high-confidence glosses
                gloss = idx2gloss.get(topi[0].item(), f"Unknown:{topi[0].item()}")
                conf  = topv[0].item()
                if conf > 0.3 and gloss != last_significant_gloss:
                    accumulated_gloss += (" " if accumulated_gloss else "") + gloss
                    pred_history.append(gloss)
                    last_significant_gloss = gloss

                iterations += 1
                # every 5 chunks, run chat_complete synchronously
                if iterations % 5 == 0 and accumulated_gloss:
                    try:
                        completed_translation = chat_complete(accumulated_gloss)
                    except Exception:
                        completed_translation = "[Chatbot Error]"
                    accumulated_gloss = ""
                    last_significant_gloss = ""

                # build output
                preds = [{"gloss": idx2gloss.get(i.item(), str(i.item())),
                          "probability": f"{v.item():.4f}"}
                         for v, i in zip(topv, topi)]
                last_prediction = {
                    "status": "Prediction",
                    "predictions": preds
                }
                if completed_translation:
                    last_prediction["translation"] = completed_translation

                # slide window
                keypoint_buffer = buffer_copy[MAX_FRAMES//3:]

            except Exception as e:
                logging.exception("Error during inference:")
                last_prediction = {"status": "Error during processing", "error": str(e)}
                keypoint_buffer = []

        return jsonify(last_prediction)
    

initialize_inference()

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize components before starting the server
    try:
        initialize_inference()
        # Run Flask App
        # Use host='0.0.0.0' to make it accessible on your network
        app.run(host='0.0.0.0', port=5000, debug=False) # Turn debug=False for production/stable testing
    except Exception as e:
        logging.error(f"Failed to initialize or run the server: {e}")
    finally:
        # Cleanup MediaPipe
        if holistic_mp:
            logging.info("Closing MediaPipe Holistic...")
            holistic_mp.close()