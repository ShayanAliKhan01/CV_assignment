"""
Data Processor Module
=====================
Organizes pose tracking data into structured formats and handles exports.
"""

import pandas as pd
import numpy as np


class DataProcessor:
    """Process and organize pose estimation data."""
    
    @staticmethod
    def build_tracking_dataframe(landmarks_list, smoothed_landmarks, all_angles, fps):
        """
        Create comprehensive DataFrame with joint angles and features.
        
        Args:
            landmarks_list: list of (frame_idx, landmarks_array, frame_bgr)
            smoothed_landmarks (ndarray): Shape (num_frames, 33, 4)
            all_angles (list): List of angle dicts
            fps (float): Video frames per second
            
        Returns:
            pd.DataFrame: Frame-by-frame tracking data
        """
        rows = []
        for frame_idx in range(len(smoothed_landmarks)):
            landmarks = smoothed_landmarks[frame_idx]
            angles = all_angles[frame_idx]
            
            shoulder_y = landmarks[11, 1]  # Left shoulder y (normalized)
            hip_y = landmarks[23, 1]       # Left hip y
            
            rows.append({
                'frame_id': frame_idx,
                'timestamp_sec': frame_idx / fps,
                'elbow_angle_deg': angles.get('elbow', np.nan),
                'knee_angle_deg': angles.get('knee', np.nan),
                'hip_angle_deg': angles.get('hip', np.nan),
                'shoulder_height_norm': shoulder_y,
                'hip_height_norm': hip_y,
                'shoulder_hip_distance': abs(shoulder_y - hip_y),
            })
        
        df = pd.DataFrame(rows)
        print(f"[OK] DataFrame created: {df.shape[0]} frames, {df.shape[1]} columns")
        return df
    
    @staticmethod
    def add_ground_truth_labels(df, ground_truth_array):
        """
        Add manual ground-truth labels to DataFrame.
        
        Args:
            df (pd.DataFrame): Tracking DataFrame
            ground_truth_array (ndarray): Array of labels [0=sitting, 1=standing]
            
        Returns:
            pd.DataFrame: Updated DataFrame with 'ground_truth_activity' column
        """
        if len(ground_truth_array) != len(df):
            raise ValueError(f"Ground truth length {len(ground_truth_array)} != DataFrame length {len(df)}")
        
        df['ground_truth_activity'] = ground_truth_array
        print("[OK] Ground truth labels added")
        return df
    
    @staticmethod
    def get_angle_statistics(df, ground_truth=None):
        """
        Compute mean, std, min, and max for knee, hip, and elbow angles,
        optionally grouped by ground truth activity labels.
        
        Args:
            df (pd.DataFrame): Tracking DataFrame
            ground_truth (ndarray): Optional ground truth labels
            
        Returns:
            dict: Statistics for each angle
        """
        stats = {}
        angles = ['elbow_angle_deg', 'knee_angle_deg', 'hip_angle_deg']
        
        for col in angles:
            if ground_truth is not None:
                stats[col] = {
                    'sitting_mean': df.loc[ground_truth == 0, col].mean(),
                    'sitting_std': df.loc[ground_truth == 0, col].std(),
                    'sitting_min': df.loc[ground_truth == 0, col].min(),
                    'sitting_max': df.loc[ground_truth == 0, col].max(),
                    'standing_mean': df.loc[ground_truth == 1, col].mean(),
                    'standing_std': df.loc[ground_truth == 1, col].std(),
                    'standing_min': df.loc[ground_truth == 1, col].min(),
                    'standing_max': df.loc[ground_truth == 1, col].max(),
                }
            else:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                }
        return stats
    
    @staticmethod
    def export_to_csv(df, output_path):
        """
        Export DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): Tracking DataFrame
            output_path (str): Output file path
        """
        df.to_csv(output_path, index=False)
        print(f"[OK] Exported to {output_path}")
