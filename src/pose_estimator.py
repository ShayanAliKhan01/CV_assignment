"""
Pose Estimation Module
======================
Handles MediaPipe Pose keypoint extraction and smoothing.
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter


class PoseEstimator:
    """Extract and process pose landmarks from video frames."""
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            confidence_threshold (float): Minimum confidence score for landmarks
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=False  # Manual smoothing applied later
        )
        self.confidence_threshold = confidence_threshold
        
    def extract_landmarks_from_video(self, video_path):
        """
        Extract raw landmarks frame-by-frame from video.
        
        Args:
            video_path (str): Path to input video file
            
        Returns:
            landmarks_list: list of tuples (frame_idx, landmarks_array, frame_bgr)
            fps (float): Video frames per second
            total_frames (int): Total number of frames
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_list = []
        frame_idx = 0
        
        print(f"Processing {total_frames} frames at {fps} FPS...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.pose.process(frame_rgb)
            
            # Extract landmarks: shape (33, 4) -> [x, y, z, visibility]
            if results.pose_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility] 
                    for lm in results.pose_landmarks.landmark
                ])
            else:
                landmarks = np.full((33, 4), np.nan)
            
            landmarks_list.append((frame_idx, landmarks, frame))
            frame_idx += 1
            
            if frame_idx % 50 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        print(f"[OK] Extraction complete: {len(landmarks_list)} frames")
        
        return landmarks_list, fps, total_frames
    
    @staticmethod
    def smooth_landmarks(landmarks_list, window=5, polyorder=2):
        """
        Apply Savitzky-Golay filter to smooth keypoint coordinates.
        
        Args:
            landmarks_list: list of (frame_idx, landmarks_array, frame_bgr)
            window (int): Filter window length (must be odd)
            polyorder (int): Polynomial order
            
        Returns:
            smoothed_landmarks (ndarray): shape (num_frames, 33, 4)
        """
        # Extract only landmarks array
        landmarks_array = np.array([lm[1] for lm in landmarks_list])
        num_frames, num_keypoints, num_dims = landmarks_array.shape
        
        smoothed = np.zeros_like(landmarks_array)
        
        # Apply filter to each keypoint dimension
        for kp_idx in range(num_keypoints):
            for dim_idx in range(num_dims):
                signal = landmarks_array[:, kp_idx, dim_idx]
                
                # Handle NaN by interpolation
                mask = ~np.isnan(signal)
                if mask.sum() < window:
                    smoothed[:, kp_idx, dim_idx] = signal
                    continue
                
                # Interpolate missing values
                indices = np.arange(len(signal))
                signal_interp = np.interp(indices, indices[mask], signal[mask])
                
                # Apply Savitzky-Golay filter
                try:
                    smoothed[:, kp_idx, dim_idx] = savgol_filter(
                        signal_interp, window, polyorder
                    )
                except ValueError:
                    # If filter fails, use original signal
                    smoothed[:, kp_idx, dim_idx] = signal
        
        print(f"[OK] Smoothing complete: {smoothed.shape}")
        return smoothed
    
    @staticmethod
    def visualize_skeleton(frame, landmarks, joint_pairs=None):
        """
        Draw skeleton keypoints and connections on frame.
        
        Args:
            frame (ndarray): BGR image frame
            landmarks (ndarray): shape (33, 4) or (33, 3)
            joint_pairs (list): List of (idx1, idx2) tuples for connections
            
        Returns:
            annotated_frame (ndarray): Frame with skeleton overlay
        """
        if joint_pairs is None:
            # Default MediaPipe skeleton connections
            joint_pairs = [
                # Arms
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                # Torso
                (11, 23), (12, 24),  # Shoulders to hips
                # Legs
                (23, 25), (25, 27),  # Left leg
                (24, 26), (26, 28),  # Right leg
            ]
        
        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()
        
        # Draw keypoints
        for i, landmark in enumerate(landmarks[:25]):  # Use only main body points
            if np.isnan(landmark[0]) or np.isnan(landmark[1]):
                continue
            
            x = int(landmark[0] * frame_w)
            y = int(landmark[1] * frame_h)
            
            # Color based on visibility
            if landmark[3] > 0.5:  # High confidence
                cv2.circle(annotated, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
            else:
                cv2.circle(annotated, (x, y), radius=3, color=(0, 165, 255), thickness=1)
        
        # Draw connections
        for idx1, idx2 in joint_pairs:
            if (np.isnan(landmarks[idx1, 0]) or np.isnan(landmarks[idx2, 0]) or
                landmarks[idx1, 3] < 0.5 or landmarks[idx2, 3] < 0.5):
                continue
            
            x1 = int(landmarks[idx1, 0] * frame_w)
            y1 = int(landmarks[idx1, 1] * frame_h)
            x2 = int(landmarks[idx2, 0] * frame_w)
            y2 = int(landmarks[idx2, 1] * frame_h)
            
            cv2.line(annotated, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        
        return annotated
    
    def close(self):
        """Cleanup MediaPipe resources."""
        self.pose.close()



