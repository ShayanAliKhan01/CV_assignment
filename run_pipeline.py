"""
Run the pose estimation -> classification -> evaluation pipeline as a script.
Saves outputs to the output/ folder.
"""

from src import PoseEstimator, AngleCalculator, DataProcessor, ActivityClassifier, Evaluator
import numpy as np
import os

INPUT_VIDEO = 'input/video.mp4'
GT_PATH = 'data/ground_truth_labels.npy'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
CONFIDENCE_THRESHOLD = 0.5
SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2

# 1. Extract landmarks
print('1/9: Extracting landmarks...')
estimator = PoseEstimator(confidence_threshold=CONFIDENCE_THRESHOLD)
landmarks_list, fps, total_frames = estimator.extract_landmarks_from_video(INPUT_VIDEO)

# 2. Smooth
print('2/9: Smoothing landmarks...')
smoothed_landmarks = estimator.smooth_landmarks(landmarks_list, window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER)

# 3. Calculate angles
print('3/9: Calculating angles...')
all_angles = AngleCalculator.calculate_all_angles(smoothed_landmarks)

# 4. Build DataFrame
print('4/9: Building DataFrame...')
df = DataProcessor.build_tracking_dataframe(landmarks_list, smoothed_landmarks, all_angles, fps)

# 5. Load or create ground truth
print('5/9: Loading ground truth...')
if os.path.exists(GT_PATH):
    ground_truth = np.load(GT_PATH)
    print(f'Loaded ground truth with {len(ground_truth)} frames')
    if len(ground_truth) != len(df):
        print(f'Warning: Ground truth length {len(ground_truth)} != frame count {len(df)}. Regenerating default all-sitting labels.')
        ground_truth = np.zeros(len(df), dtype=int)
        np.save(GT_PATH, ground_truth)
else:
    print('No ground truth found. Creating default all-sitting labels (0).')
    ground_truth = np.zeros(len(df), dtype=int)
    np.save(GT_PATH, ground_truth)

# Add to df
df = DataProcessor.add_ground_truth_labels(df, ground_truth)

# 6. Grid search thresholds
print('6/9: Grid searching thresholds (may take a while)...')
best_params, results_df = ActivityClassifier.grid_search_thresholds(df, ground_truth)
print('Best params:', best_params)

# 7. Classify with best params
print('7/9: Classifying frames...')
classifier = ActivityClassifier(knee_threshold=best_params.get('knee', 135), hip_threshold=best_params.get('hip', 120))
df = classifier.classify_all_frames(df)

# 8. Evaluate
print('8/9: Evaluating...')
metrics = Evaluator.compute_metrics(ground_truth, df['predicted_activity'].values)
Evaluator.print_metrics(metrics)
Evaluator.plot_confusion_matrix(ground_truth, df['predicted_activity'].values, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
Evaluator.plot_predictions_vs_ground_truth(df, ground_truth, save_path=os.path.join(OUTPUT_DIR, 'predictions_vs_ground_truth.png'))
Evaluator.plot_angle_distributions(df, ground_truth, save_path=os.path.join(OUTPUT_DIR, 'angle_distributions.png'))
Evaluator.plot_angles_over_time(df, ground_truth=ground_truth, thresholds=best_params, save_path=os.path.join(OUTPUT_DIR, 'angle_analysis.png'))

# 9. Export
print('9/9: Exporting results...')
DataProcessor.export_to_csv(df, os.path.join(OUTPUT_DIR, 'results.csv'))
print('Done. Outputs in', OUTPUT_DIR)

estimator.close()
