from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import datetime

class Logger:
    def __init__(self, log_dir='runs'):
        # Create unique run directory with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / timestamp
        self.writer = SummaryWriter(self.log_dir)
        
    def log_metrics(self, metrics, step, prefix='train'):
        """Log metrics to tensorboard"""
        for k, v in metrics.items():
            self.writer.add_scalar(f'{prefix}/{k}', v, step)
    
    def log_images(self, images, targets, predictions, step, max_imgs=5):
        """Log images with bounding boxes to tensorboard"""
        images = images[:max_imgs].cpu().numpy()
        
        for idx, (img, target, pred) in enumerate(zip(images, targets[:max_imgs], predictions[:max_imgs])):
            fig, ax = plt.subplots(1)
            
            # Show image
            img = np.transpose(img, (1, 2, 0))  # CHW to HWC
            img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            
            # Draw ground truth boxes in green
            self._draw_boxes(ax, target, color='green', label='Ground Truth')
            
            # Draw predicted boxes in red
            self._draw_boxes(ax, pred, color='red', label='Prediction')
            
            ax.legend()
            self.writer.add_figure(f'predictions/image_{idx}', fig, step)
            plt.close(fig)
    
    def log_model_graph(self, model, images):
        """Log model architecture"""
        self.writer.add_graph(model, images)
    
    def log_pr_curve(self, labels, predictions, step, prefix='val'):
        """Log precision-recall curve"""
        self.writer.add_pr_curve(f'{prefix}/PR_curve', labels, predictions, step)
    
    def log_confusion_matrix(self, cm, class_names, step, prefix='val'):
        """Log confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm)
        
        # Add labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add colorbar
        plt.colorbar(im)
        
        self.writer.add_figure(f'{prefix}/confusion_matrix', fig, step)
        plt.close(fig)
    
    def _draw_boxes(self, ax, boxes, color='green', label=None):
        """Helper function to draw bounding boxes"""
        for box in boxes:
            class_id, x, y, w, h = box[:5]
            rect = plt.Rectangle((x-w/2, y-h/2), w, h, fill=False, color=color, label=label)
            ax.add_patch(rect)
    
    def close(self):
        self.writer.close() 