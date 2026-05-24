"""Annotate the input video with MediaPipe pose tracking lines and save the result."""

import os
from src import PoseEstimator
import cv2

INPUT_VIDEO = 'input/video.mp4'
OUTPUT_VIDEO = 'output/annotated_video.mp4'

os.makedirs('output', exist_ok=True)

print('Loading pose estimator...')
estimator = PoseEstimator(confidence_threshold=0.5)
print(f'Processing video: {INPUT_VIDEO}')
landmarks_list, fps, total_frames = estimator.extract_landmarks_from_video(INPUT_VIDEO)

if len(landmarks_list) == 0:
    raise RuntimeError('No frames were processed from the input video.')

# Use the first frame to obtain frame dimensions
frame_height, frame_width = landmarks_list[0][2].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

print(f'Writing annotated video to: {OUTPUT_VIDEO}')
for frame_idx, landmarks, frame in landmarks_list:
    annotated = estimator.visualize_skeleton(frame, landmarks)
    writer.write(annotated)
    if (frame_idx + 1) % 50 == 0:
        print(f'  Annotated {frame_idx + 1}/{total_frames} frames')

writer.release()
estimator.close()
print('Annotation complete!')
print(f'Output saved at: {OUTPUT_VIDEO}')
