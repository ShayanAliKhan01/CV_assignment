"""
Angle Calculator Module
=======================
Calculates joint angles using robust vector geometry.
"""

import numpy as np


class AngleCalculator:
    """Calculate joint angles from landmark coordinates."""
    
    @staticmethod
    def calculate_angle(p_a, p_b, p_c):
        """
        Calculate angle at point B (between BA and BC vectors) in degrees [0, 180].
        
        Args:
            p_a, p_b, p_c (array): Points as [x, y, z, confidence] or [x, y]
            
        Returns:
            float: Angle in degrees, or np.nan if invalid
        """
        try:
            a = np.array(p_a[:2])
            b = np.array(p_b[:2])
            c = np.array(p_c[:2])
            
            if np.any(np.isnan([a, b, c])):
                return np.nan
            
            # Compute vectors
            v1 = a - b
            v2 = c - b
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-6 or norm2 < 1e-6:
                return np.nan
            
            # Cosine formula from dot product
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
            
            return np.degrees(np.arccos(cos_angle))
        except Exception:
            return np.nan
    
    @staticmethod
    def extract_joint_angles(landmarks_frame, confidence_threshold=0.5):
        """
        Extract knee, hip, and elbow joint angles from a single frame.
        
        Args:
            landmarks_frame (ndarray): Shape (33, 4) - [x, y, z, confidence]
            confidence_threshold (float): Minimum visibility score
            
        Returns:
            dict: Keys 'elbow', 'knee', 'hip' with angle values in degrees or np.nan
        """
        # Joint indices
        shoulder_idx = 11  # Left shoulder
        elbow_idx = 13     # Left elbow
        wrist_idx = 15     # Left wrist
        hip_idx = 23       # Left hip
        knee_idx = 25      # Left knee
        ankle_idx = 27     # Left ankle
        
        angles = {}
        
        def compute_joint_angle(idx1, idx2, idx3):
            if (landmarks_frame[idx1, 3] > confidence_threshold and
                landmarks_frame[idx2, 3] > confidence_threshold and
                landmarks_frame[idx3, 3] > confidence_threshold):
                return AngleCalculator.calculate_angle(
                    landmarks_frame[idx1],
                    landmarks_frame[idx2],
                    landmarks_frame[idx3]
                )
            return np.nan
            
        angles['elbow'] = compute_joint_angle(shoulder_idx, elbow_idx, wrist_idx)
        angles['knee'] = compute_joint_angle(hip_idx, knee_idx, ankle_idx)
        angles['hip'] = compute_joint_angle(shoulder_idx, hip_idx, knee_idx)
        
        return angles
    
    @staticmethod
    def calculate_all_angles(smoothed_landmarks, confidence_threshold=0.5):
        """
        Calculate joint angles for all frames.
        
        Args:
            smoothed_landmarks (ndarray): Shape (num_frames, 33, 4)
            confidence_threshold (float): Minimum visibility score
            
        Returns:
            list: List of angle dicts, one per frame
        """
        return [
            AngleCalculator.extract_joint_angles(frame, confidence_threshold)
            for frame in smoothed_landmarks
        ]
