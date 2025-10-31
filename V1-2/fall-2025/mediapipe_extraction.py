import cv2
import numpy as np
import torch
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import json
FACE_CENTER = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
POSE_LANDMARKS = 17

class MediaPipeLandmarkExtractor:
    def __init__(
        self,
        use_gpu: bool = False,
        extract_face=False,
        static_image_mode=True,
        pose_complexity: int = 1,
        hand_complexity: int = 0
    ):
        """
        Initialize MediaPipe Landmark Extractor

        Args:
            use_gpu: Whether to use GPU for torch operations
            extract_face: Whether to extract face landmarks
            static_image_mode: True for images, False for videos (better performance)
            pose_complexity: Pose model complexity (0=lite/fastest, 1=full, 2=heavy/slowest)
            hand_complexity: Hand model complexity (0=lite/fastest, 1=full/slowest)
        """
        # Set device for torch
        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Store configuration
        self.static_image_mode = static_image_mode
        self.extract_face = extract_face

        # Initialize models
        self.pose_landmark_indices = [i for i in range(POSE_LANDMARKS)]
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=pose_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            model_complexity=hand_complexity,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        if extract_face:
            self.mp_face_mesh = mp.solutions.face_mesh
            # Define the specific face landmarks we want (based on your document)
            self.face_landmark_indices = [
                234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                152, 377, 400, 378, 379, 365, 397, 367, 288, 435,
                361, 401, 323
            ]

            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


    def process_face(self, results, image_rgb: np.ndarray):
        """
        Process MediaPipe face landmarks from an RGB image.
        Optionally filters landmarks by specified indices and stores the result in 'results["face"]'.
        """
        # FIX: Use face_mesh instead of pose for face detection
        if not hasattr(self, 'face_mesh'):
            results['face'] = None
            return

        face_results_raw = self.face_mesh.process(image_rgb)
        if face_results_raw.multi_face_landmarks:
            if self.face_landmark_indices:
                # Keep only selected landmark indices - FIX: use .landmark attribute
                face_landmarks_full = face_results_raw.multi_face_landmarks[0].landmark
                filtered_landmarks = [face_landmarks_full[idx] for idx in self.face_landmark_indices]

                # Create a compatible result object
                class FilteredFaceLandmarks:
                    def __init__(self, landmarks):
                        self.landmark = landmarks

                results['face'] = type('obj', (object,), {
                    'multi_face_landmarks': [FilteredFaceLandmarks(filtered_landmarks)]
                })()
            else:
                # Store all face landmarks without selection
                results['face'] = face_results_raw
        else:
            results['face'] = None  # No face landmarks detected
        
    def process_pose(self, results, image_rgb: np.ndarray):
        """
        Process MediaPipe pose landmarks from an RGB image.
        Optionally filters landmarks by specified indices and stores the result in 'results["pose"]'.
        """
        pose_results_raw = self.pose.process(image_rgb)
        if pose_results_raw.pose_landmarks:
            if self.pose_landmark_indices:
                # Keep only selected landmark indices
                filtered_landmarks = [
                    pose_results_raw.pose_landmarks.landmark[idx]
                    for idx in self.pose_landmark_indices
                ]
                
                # Create filtered landmark object preserving MediaPipe structure
                class FilteredPoseLandmarks:
                    def __init__(self, landmarks, pose_landmark_indices):
                        self.landmark = landmarks
                        if hasattr(pose_results_raw.pose_landmarks, 'visibility'):
                            self.visibility = [
                                pose_results_raw.pose_landmarks.visibility[i] 
                                for i in pose_landmark_indices
                            ]

                # Store filtered results in compatible structure
                results['pose'] = type('obj', (object,), {
                    'pose_landmarks': FilteredPoseLandmarks(
                        filtered_landmarks,
                        self.pose_landmark_indices
                    )
                })()
            else:
                # Store all pose landmarks without selection
                results['pose'] = pose_results_raw
        else:
            results['pose'] = None # No pose landmarks detected
    
    def process_hands(self, results, image_rgb: np.ndarray):
        results['hands'] = self.hands.process(image_rgb)


    def process_image(self, image: np.ndarray, extract_face = False) -> Dict:
        """
        Process a single image and extract all landmarks
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing pose and hands (and face results if extract_face)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = {}
        
        self.process_pose(results, image_rgb)
        self.process_hands(results, image_rgb)
        if extract_face:
            self.process_face(results, image_rgb)
            
        return results
    

    def unify_face_to_shoulder_center(
            self, 
            face_results, 
            unified_data, 
            nose_centered
    ):
        if face_results and face_results.multi_face_landmarks:
            face_keypoints = face_results.multi_face_landmarks[0]
            

    def unify_one_part(self, unified_data, keypoints, anchor_centered, part = 'left_hand'):
        if not keypoints:
            return
        local_center = keypoints[0] #palm for hands and nose tip for face

        for landmark in keypoints:
            # Calculate displacement vector in hand's normalized space
            offset_x = landmark.x - local_center.x
            offset_y = landmark.y - local_center.y
            offset_z = landmark.z - local_center.z

            # Apply offset to anchor that is shoulder-centered
            unified_data[part].append({
                'x': anchor_centered['x'] + offset_x,
                'y': anchor_centered['y'] + offset_y,
                'z': anchor_centered['z'] + offset_z
            })

    def unify_hands_to_shoulder_center(
        self,
        hand_results,
        unified_data,
    ):
        if hand_results and hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[idx].classification[0].label
                hand_keypoints = hand_landmarks.landmark

                if handedness == "Left":
                    wrist_centered = unified_data['pose'][LEFT_WRIST]
                    self.unify_one_part(unified_data, hand_keypoints, wrist_centered, part = 'left_hand')
                
                elif handedness == "Right":
                    wrist_centered = unified_data['pose'][RIGHT_WRIST]
                    self.unify_one_part(unified_data, hand_keypoints, wrist_centered, part = 'right_hand')

    def unify_all_normalized_landmarks_to_shoulder_center(
            self,
            results,
            extract_face
    ):
        """
        Transform all MediaPipe landmarks to shoulder-centered coordinates using NORMALIZED landmarks.
        Shoulder center = midpoint between left shoulder (11) and right shoulder (12)
        
        Returns unified data in shoulder-centered coordinate system
        """
        if not results['pose'] or not results['pose'].pose_landmarks:
            return None
        
        pose_norm = results['pose'].pose_landmarks.landmark

        # Calculate shoulder center in NORMALIZED space (NEW ORIGIN)
        left_shoulder = pose_norm[LEFT_SHOULDER]  # Left shoulder
        right_shoulder = pose_norm[RIGHT_SHOULDER]  # Right shoulder

        shoulder_center = {
            'x': (left_shoulder.x + right_shoulder.x) / 2,
            'y': (left_shoulder.y + right_shoulder.y) / 2,
            'z': (left_shoulder.z + right_shoulder.z) / 2
        }

        unified_data = {
            'pose': [],
            'left_hand': [],
            'right_hand': []
        }

        if extract_face:
            unified_data['face']= []

        # Transform pose to shoulder-centered (normalized space)
        for landmark in pose_norm:
            unified_data['pose'].append({
                'x': landmark.x - shoulder_center['x'],
                'y': landmark.y - shoulder_center['y'],
                'z': landmark.z - shoulder_center['z']
            })
        
        # Transform left and right hand using wrists as anchors
        self.unify_hands_to_shoulder_center(results['hands'], unified_data)
        #Transform face using nose as anchor
        if extract_face:
            face_results = results['face']
            face_center = unified_data['pose'][FACE_CENTER]
            if face_results and face_results.multi_face_landmarks:
                # FIX: Access .landmark attribute for face keypoints
                face_keypoints = face_results.multi_face_landmarks[0].landmark
                self.unify_one_part(unified_data, face_keypoints, face_center, part = 'face')

        return unified_data

    def landmarks_to_tensor(
            self, 
            unified_data: Dict, 
            extract_face=False
    ) -> torch.Tensor:
        """
        Convert unified landmarks to PyTorch tensor
        
        Args:
            unified_data: Unified landmarks dictionary
            
        Returns:
            PyTorch tensor of shape [num_joints, 3]
        """
        all_landmarks = []
        
        # Add pose landmarks (using selected indices [0, 15] as mentioned in document)
        pose_indices = list(range(0, 16))  # 0-15 inclusive
        for idx in pose_indices:
            if idx < len(unified_data['pose']):
                landmark = unified_data['pose'][idx]
                all_landmarks.append([landmark['x'], landmark['y'], landmark['z']])
        
        # Add left hand landmarks
        for landmark in unified_data['left_hand']:
            all_landmarks.append([landmark['x'], landmark['y'], landmark['z']])
        
        # Add right hand landmarks
        for landmark in unified_data['right_hand']:
            all_landmarks.append([landmark['x'], landmark['y'], landmark['z']])

                # Add face landmarks
        if extract_face:
            for landmark in unified_data['face']:
                all_landmarks.append([landmark['x'], landmark['y'], landmark['z']])
        
        # Convert to tensor
        tensor = torch.tensor(all_landmarks, dtype=torch.float32, device=self.device)
        
        return tensor

    def draw_landmarks(self, image: np.ndarray, results: Dict, extract_face=False, inplace=False, lightweight=False):
        """
        Draw landmarks on the image for visualization

        Args:
            image: Original image
            results: MediaPipe results dictionary
            extract_face: Whether to draw face landmarks
            inplace: If True, draw directly on input image (faster). If False, create a copy.
            lightweight: If True, skip text labels for 30-40% speed improvement

        Returns:
            Image with drawn landmarks
        """
        annotated_image = image if inplace else image.copy()

        # Draw pose landmarks
        if results['pose'] and results['pose'].pose_landmarks:
            # Draw only the landmarks we have (no connections)
            for idx, landmark in enumerate(results['pose'].pose_landmarks.landmark):
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
                if not lightweight:
                    cv2.putText(annotated_image, str(self.pose_landmark_indices[idx]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw hand landmarks
        if results['hands'] and results['hands'].multi_hand_landmarks:
            for hand_landmarks in results['hands'].multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )


        # Draw face landmarks (simplified - just points)
        if extract_face and results['face'] and results['face'].multi_face_landmarks:
            # FIX: Access .landmark attribute for face landmarks
            face_landmarks = results['face'].multi_face_landmarks[0].landmark
            for landmark in face_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
        
        return annotated_image

    def process_single_image(
            self,
            image_path,
            visualize = True,
            extract_face = False
    ) -> Dict:
        """
        Process a single image and return unified landmarks

        Args:
            image_path: Path to the image file
            visualize: Whether to display the results
            extract_face: Whether to extract face landmarks

        Returns:
            Dictionary containing original results and unified landmarks
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Extract landmarks - FIX: pass extract_face parameter
        results = self.process_image(image, extract_face=extract_face)
        
        # Unify coordinates
        unified_landmarks = self.unify_all_normalized_landmarks_to_shoulder_center(results, extract_face=extract_face)
        
        # Convert to tensor
        if unified_landmarks:
            landmark_tensor = self.landmarks_to_tensor(
                unified_landmarks, extract_face=extract_face
            )
            print(f"Landmark tensor shape: {landmark_tensor.shape}")
            print(f"Landmark tensor device: {landmark_tensor.device}")
        else:
            landmark_tensor = None
            print("No landmarks detected")
        
        # Visualize if requested
        if visualize and unified_landmarks:
            annotated_image = self.draw_landmarks(
                image, results, extract_face=extract_face
            )
            cv2.imwrite('Landmarks_Detection.png', annotated_image)
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'original_results': results,
            'unified_landmarks': unified_landmarks,
            'landmark_tensor': landmark_tensor
        }
    
    
    def process_video(
        self,
        video_path: str,
        output_path: str = "output_video.mp4",
        extract_face: bool = False,
        visualize: bool = True,
        fps: int = 30,
        skip_frames: int = 2,
        verbose: bool = True,
        progress_interval: int = 10,
        lightweight_viz: bool = True,
        store_tensors: bool = True
    ) -> Dict:
        """
        Process a video and extract landmarks for each frame

        Args:
            video_path: Path to input video file
            output_path: Path to save output video with landmarks
            extract_face: Whether to extract face landmarks
            visualize: Whether to draw landmarks on output video
            fps: Frames per second for output video
            skip_frames: Process every (skip_frames+1)th frame (0=all frames, 1=every 2nd, etc.)
            verbose: Whether to print progress messages
            progress_interval: Print progress every N frames (only if verbose=True)
            lightweight_viz: Skip text labels for 30-40% faster drawing
            store_tensors: Whether to store landmark tensors (disable for less memory/faster)

        Returns:
            Dictionary containing unified landmarks for each frame
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Adjust fps if skipping frames
        output_fps = fps if skip_frames == 0 else fps / (skip_frames + 1)

        # Initialize video writer if visualization is enabled
        if visualize:
            # Try h264 (faster), fallback to mp4v if unavailable
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec - faster
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
                if not out.isOpened():
                    raise Exception("H.264 codec not available")
            except:
                if verbose:
                    print("H.264 codec unavailable, using mp4v (slower)")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

        # Store results for all frames
        video_results = {
            'frames': [],
            'video_properties': {
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'original_fps': original_fps,
                'output_fps': output_fps,
                'skip_frames': skip_frames
            }
        }

        if verbose:
            print(f"Processing video: {video_path}")
            print(f"Total frames: {total_frames}")
            if skip_frames > 0:
                print(f"Skipping every {skip_frames} frames (processing ~{total_frames//(skip_frames+1)} frames)")

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if configured
            if skip_frames > 0 and (frame_count - 1) % (skip_frames + 1) != 0:
                continue

            processed_count += 1

            # Print progress at intervals
            if verbose and processed_count % progress_interval == 0:
                percentage = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {frame_count}/{total_frames} frames ({percentage:.1f}%)")

            # Process current frame
            results = self.process_image(frame, extract_face=extract_face)
            unified_landmarks = self.unify_all_normalized_landmarks_to_shoulder_center(
                results, extract_face=extract_face
            )

            # Convert to tensor only if needed (saves time and memory)
            landmark_tensor = None
            if store_tensors and unified_landmarks:
                landmark_tensor = self.landmarks_to_tensor(unified_landmarks, extract_face=extract_face)

            # Store frame results
            frame_data = {
                'frame_number': frame_count,
                'unified_landmarks': unified_landmarks,
                'landmark_tensor': landmark_tensor
            }
            video_results['frames'].append(frame_data)

            # Draw landmarks and write to output video
            if visualize:
                if unified_landmarks:
                    # Use inplace drawing + lightweight mode for best performance
                    annotated_frame = self.draw_landmarks(
                        frame, results,
                        extract_face=extract_face,
                        inplace=True,
                        lightweight=lightweight_viz
                    )
                    out.write(annotated_frame)
                else:
                    out.write(frame)  # Write original frame if no landmarks

        # Release resources
        cap.release()
        if visualize:
            out.release()
            if verbose:
                print(f"Output video saved to: {output_path}")

        if verbose:
            print(f"Video processing completed. Processed {processed_count}/{frame_count} frames.")

        return video_results

    def process_video_to_tensor(
        self,
        video_path: str,
        extract_face: bool = False,
        skip_frames: int = 2,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Process video and return all landmarks as a single tensor
        Shape: [num_frames, num_landmarks, 3]

        Args:
            video_path: Path to input video file
            extract_face: Whether to extract face landmarks
            skip_frames: Process every (skip_frames+1)th frame for faster processing
            verbose: Whether to print progress messages

        Returns:
            Tensor of shape [num_frames, num_landmarks, 3]
        """
        video_results = self.process_video(
            video_path,
            visualize=False,  # No visualization for tensor extraction
            extract_face=extract_face,
            skip_frames=skip_frames,
            verbose=verbose
        )

        # Collect all tensors from frames that have landmarks
        frame_tensors = []
        for frame_data in video_results['frames']:
            if frame_data['landmark_tensor'] is not None:
                frame_tensors.append(frame_data['landmark_tensor'])

        if not frame_tensors:
            if verbose:
                print("No landmarks detected in any frame")
            return torch.tensor([])

        # Stack all frame tensors
        video_tensor = torch.stack(frame_tensors)
        if verbose:
            print(f"Video tensor shape: {video_tensor.shape}")  # [frames, landmarks, 3]

        return video_tensor

    # Add these methods to your MediaPipeLandmarkExtractor class
    # Just add them inside the class definition

    # Example usage for video processing:
    def process_video_example():
        extractor = MediaPipeLandmarkExtractor(use_gpu=False, extract_face=False)
        
        # Process video with visualization
        video_results = extractor.process_video(
            video_path="input_video.mp4",
            output_path="output_with_landmarks.mp4",
            extract_face=False,
            visualize=True,
            fps=30
        )
        
        # Or get as tensor for ML models
        video_tensor = extractor.process_video_to_tensor(
            video_path="input_video.mp4",
            extract_face=False
        )
        
        return video_results, video_tensor


# Example usage
def main(extract_face = False, video = False):
    """
    Main function demonstrating image and video processing

    Args:
        extract_face: Whether to extract face landmarks
        video: If True, process video. If False, process single image.
    """
    try:
        if not video:
            # Process a single image - use static_image_mode=True (default)
            extractor = MediaPipeLandmarkExtractor(
                use_gpu=False,
                extract_face=extract_face,
                static_image_mode=True  # Best for images
            )

            # Replace with your image path
            image_path = "pics_and_videos/full_body.jpg.webp"
            results = extractor.process_single_image(image_path, visualize=True)

            # Print some information
            if results['unified_landmarks']:
                unified = results['unified_landmarks']
                print(f"Pose landmarks: {unified['pose']}")
                if extract_face:
                    print(f"Face landmarks: {unified['face']}")
                print(f"Left hand landmarks: {unified['left_hand']}")
                print(f"Right hand landmarks: {unified['right_hand']}")
                return unified
        else:
            # Process video - optimized for SLT (Sign Language Translation)
            extractor = MediaPipeLandmarkExtractor(
                use_gpu=False,
                extract_face=extract_face,
                static_image_mode=False,  # Better tracking for videos
                pose_complexity=1,  # 0=fastest, 1=balanced/recommended for SLT, 2=most accurate but slow
                hand_complexity=1   # 0=fastest, 1=recommended for SLT (hands are critical!)
            )

            # ===== MODE 1: SLT Dataset Preprocessing (FASTEST - no visualization) =====
            # Use this to extract landmarks from your sign language videos for training/inference
            results = extractor.process_video(
                video_path="pics_and_videos/dancing.mp4",
                output_path="pics_and_videos/dancing_landmarked.mp4",
                visualize=False,       # No video output = much faster
                skip_frames=0,         # Process all frames for SLT
                verbose=True,
                progress_interval=30,
                lightweight_viz=False, # Doesn't matter when visualize=False
                store_tensors=True     # CRITICAL for SLT - need the tensor data!
            )
            print(f"Processed {len(results['frames'])} frames for SLT")

            # ===== MODE 2: Debug/Visualization Mode =====
            # Use this to check if landmark extraction is working correctly
            # results = extractor.process_video(
            #     video_path="pics_and_videos/dancing.mp4",
            #     output_path="pics_and_videos/dancing_landmarked.mp4",
            #     visualize=True,
            #     skip_frames=2,         # Can skip frames when just checking
            #     verbose=True,
            #     lightweight_viz=True,  # Faster drawing
            #     store_tensors=False    # Don't need tensors for visualization
            # )

            # ===== MODE 3: Fast tensor extraction (no video output) =====
            # tensor_data = extractor.process_video_to_tensor(
            #     "pics_and_videos/dancing.mp4",
            #     skip_frames=0,  # Process all frames for SLT
            #     verbose=True
            # )
            # print(f"Tensor shape: {tensor_data.shape}")  # [frames, landmarks, 3]
            # # Save for later use:
            # torch.save(tensor_data, "landmarks.pt")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main(extract_face=False, video=True)