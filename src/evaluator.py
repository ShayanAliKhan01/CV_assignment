"""
Evaluation Module
=================
Computes performance metrics and generates visual reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evaluator:
    """Evaluate classification performance and generate visual reports."""
    
    @staticmethod
    def compute_confusion_matrix(ground_truth, predictions):
        """
        Compute confusion matrix components (TP, TN, FP, FN).
        """
        valid_mask = ~pd.isna(predictions)
        gt_valid = ground_truth[valid_mask]
        pred_valid = predictions[valid_mask]
        
        tp = np.sum((pred_valid == 1) & (gt_valid == 1))
        tn = np.sum((pred_valid == 0) & (gt_valid == 0))
        fp = np.sum((pred_valid == 1) & (gt_valid == 0))
        fn = np.sum((pred_valid == 0) & (gt_valid == 1))
        
        return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}
    
    @staticmethod
    def compute_metrics(ground_truth, predictions):
        """
        Compute standard classification metrics (accuracy, precision, recall, f1).
        """
        cm = Evaluator.compute_confusion_matrix(ground_truth, predictions)
        tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    @staticmethod
    def print_metrics(metrics):
        """
        Print formatted table of metrics in terminal.
        """
        cm = metrics['confusion_matrix']
        print("\n" + "="*50)
        print("              CLASSIFICATION METRICS")
        print("="*50)
        print(f"Accuracy:   {metrics['accuracy']:.3f}")
        print(f"Precision:  {metrics['precision']:.3f}")
        print(f"Recall:     {metrics['recall']:.3f}")
        print(f"F1-Score:   {metrics['f1_score']:.3f}")
        print("-"*50)
        print(f"True Positives (Standing):  {cm['TP']}")
        print(f"True Negatives (Sitting):   {cm['TN']}")
        print(f"False Positives:            {cm['FP']}")
        print(f"False Negatives:            {cm['FN']}")
        print("="*50 + "\n")
    
    @staticmethod
    def plot_confusion_matrix(ground_truth, predictions, save_path=None):
        """
        Visualize confusion matrix as heatmap.
        """
        cm_dict = Evaluator.compute_confusion_matrix(ground_truth, predictions)
        cm = np.array([
            [cm_dict['TN'], cm_dict['FP']],
            [cm_dict['FN'], cm_dict['TP']]
        ])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=['Predicted Sitting', 'Predicted Standing'],
            yticklabels=['True Sitting', 'True Standing'],
            ylabel='True Label', xlabel='Predicted Label'
        )
        
        # Add labels to heatmap cells
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight='bold')
        
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_predictions_vs_ground_truth(df, ground_truth, save_path=None):
        """
        Visualize timeline comparing predictions against ground-truth activities.
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        
        ax.fill_between(df['timestamp_sec'], 0, ground_truth, alpha=0.3, color='green', label='Ground Truth: Standing')
        
        pred = df['predicted_activity'].values
        ax.fill_between(df['timestamp_sec'], 0, pred, alpha=0.3, color='blue', label='Predicted: Standing')
        
        # Draw red lines at mismatching timestamps
        valid = ~pd.isna(pred)
        mismatch = pred[valid] != ground_truth[valid]
        mismatch_times = df.loc[valid].iloc[mismatch]['timestamp_sec'].values
        for t in mismatch_times:
            ax.axvline(x=t, color='red', alpha=0.2, linewidth=0.8)
            
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Activity (0=Sitting, 1=Standing)', fontsize=10)
        ax.set_title('Predictions vs Ground Truth Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_angle_distributions(df, ground_truth, save_path=None):
        """
        Visualize histograms comparing angle values during sitting vs. standing.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        cols = ['knee_angle_deg', 'hip_angle_deg', 'elbow_angle_deg']
        names = ['Knee Angle Distribution', 'Hip Angle Distribution', 'Elbow Angle Distribution']
        
        for idx, col in enumerate(cols):
            axes[idx].hist(df.loc[ground_truth == 0, col], bins=25, alpha=0.5, label='Sitting', color='blue')
            axes[idx].hist(df.loc[ground_truth == 1, col], bins=25, alpha=0.5, label='Standing', color='orange')
            axes[idx].set_xlabel('Angle (degrees)')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(names[idx], fontweight='bold', fontsize=11)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_angles_over_time(df, ground_truth=None, thresholds=None, save_path=None):
        """
        Plot angle variations over time with classification threshold boundaries.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        cols = ['knee_angle_deg', 'hip_angle_deg', 'elbow_angle_deg']
        colors = ['blue', 'green', 'purple']
        names = ['Knee Angle', 'Hip Angle', 'Elbow Angle']
        
        for idx, col in enumerate(cols):
            axes[idx].plot(df['timestamp_sec'], df[col], label=names[idx], color=colors[idx], linewidth=1.5)
            
            # Draw threshold line if specified
            key = col.split('_')[0]
            if thresholds and key in thresholds:
                axes[idx].axhline(y=thresholds[key], color='red', linestyle='--', label=f'Threshold: {thresholds[key]}°')
                
            axes[idx].set_ylabel('Angle (degrees)')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_title(f"{names[idx]} Over Time", fontweight='bold', fontsize=11)
            
        axes[2].set_xlabel('Time (seconds)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved plot: {save_path}")
        plt.show()
