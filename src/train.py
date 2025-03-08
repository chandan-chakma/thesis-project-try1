import torch
from torch.utils.data import DataLoader
from models.yolo_world import YOLOWorld
from utils.data_loading import UnifiedDataset, custom_collate_fn
from utils.logger import Logger
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import average_precision_score, confusion_matrix
import argparse
import yaml
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    return parser.parse_args()

def validate(model, val_loader, device, logger, epoch, config):
    """Validation loop with metrics calculation"""
    model.set_eval_mode()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move data to GPU
            images = batch['image'].to(device, non_blocking=True)
            labels = [label.to(device, non_blocking=True) for label in batch['labels']]
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(images, labels)
                loss = outputs['loss']
                preds = outputs['pred']
            
            val_loss += loss.item()
            
            # Collect predictions and targets for metrics
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend([l.cpu().numpy() for l in labels])
            
            # Log sample predictions periodically
            if batch_idx % 100 == 0:
                try:
                    logger.log_images(
                        images=images,
                        predictions=preds
                    )
                except Exception as e:
                    print(f"Warning: Failed to log images: {str(e)}")
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate metrics
    val_loss = val_loss / len(val_loader)
    
    # Calculate mAP and other metrics
    metrics = calculate_metrics(all_preds, all_targets, config['class_names'])
    
    # Log all metrics
    logger.log_metrics({
        'val/loss': val_loss,
        'val/mAP': metrics['mAP'],
        'val/precision': metrics['precision'],
        'val/recall': metrics['recall'],
    }, epoch)
    
    # Log confusion matrix
    try:
        logger.log_confusion_matrix(
            metrics['confusion_matrix'],
            config['class_names'],
            global_step=epoch
        )
    except Exception as e:
        print(f"Warning: Failed to log confusion matrix: {str(e)}")
    
    return val_loss, metrics

def calculate_metrics(predictions, targets, class_names):
    """Calculate various metrics for object detection"""
    metrics = {}
    
    # Convert predictions and targets to appropriate format
    pred_boxes = np.concatenate(predictions)
    target_boxes = np.concatenate(targets)
    
    # Calculate mAP
    ap_per_class = []
    for class_idx in range(len(class_names)):
        mask_pred = pred_boxes[:, 0] == class_idx
        mask_target = target_boxes[:, 0] == class_idx
        
        if mask_target.sum() > 0:
            ap = average_precision_score(
                mask_target,
                mask_pred
            )
            ap_per_class.append(ap)
    
    metrics['mAP'] = np.mean(ap_per_class)
    
    # Calculate confusion matrix
    cm = confusion_matrix(
        target_boxes[:, 0],
        pred_boxes[:, 0],
        labels=range(len(class_names))
    )
    metrics['confusion_matrix'] = cm
    
    # Calculate precision and recall
    metrics['precision'] = np.mean([p.mean() for p in predictions])
    metrics['recall'] = np.mean([t.mean() for t in targets])
    
    return metrics

def train():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print CUDA information once
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA GPU available. Training will be slow on CPU.")
        print("To enable GPU support, install PyTorch with CUDA:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # Optimize CUDA settings
    if torch.cuda.is_available():
        # Set CUDA device
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # Enable TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Print GPU info
        print("\nGPU Setup:")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Optimize batch size based on GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem >= 6:  # For GPUs with 6GB or more
            config['batch_size'] = 32
            config['accumulation_steps'] = 2
        else:  # For GPUs with less memory
            config['batch_size'] = 16
            config['accumulation_steps'] = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories and logger
    save_dir = Path('runs/yolo_world')
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(save_dir)
    
    # Initialize model
    model = YOLOWorld(
        yolo_model=config['yolo_model'],
        text_encoder=config['text_encoder']
    ).to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    # Optimize model parameters
    model = model.float()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Create data loaders with optimized settings
    data_dir = Path(config['data_dir']).resolve()  # Get absolute path
    print(f"Loading dataset from: {data_dir}")
    
    train_dataset = UnifiedDataset(data_dir, split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,  # Increased worker threads
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=3,  # Prefetch more batches
        collate_fn=custom_collate_fn
    )
    
    # Create validation loader
    val_dataset = UnifiedDataset(data_dir, split='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Initialize optimizer with better parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-7,
        betas=(0.9, 0.999),
        amsgrad=True  # Enable AMSGrad
    )
    
    # Learning rate scheduler with better warmup
    num_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * 5  # 5 epochs of warmup
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=num_steps,
        pct_start=warmup_steps/num_steps,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=True  # Enable three-phase schedule
    )
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_map = 0
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config['epochs']):
        model.set_train_mode()  # Use custom training mode
        train_loss = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, batch in pbar:
            try:
                # Clear cache periodically
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
                # Forward pass with automatic mixed precision
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    loss = model.train_step(batch)
                
                if not isinstance(loss, torch.Tensor) or torch.isnan(loss).any():
                    print(f"Warning: Invalid loss at batch {batch_idx}")
                    continue
                
                # Backward pass with gradient scaling
                scaled_loss = loss / config['accumulation_steps']
                
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                    if (batch_idx + 1) % config['accumulation_steps'] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                else:
                    scaled_loss.backward()
                    if (batch_idx + 1) % config['accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                gpu_memory = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'gpu_mem': f"{gpu_memory:.2f}GB",
                    'lr': f"{current_lr:.6f}"
                })
                
                # Log metrics
                if batch_idx % 10 == 0:
                    logger.log_metrics({
                        'train/loss': loss.item(),
                        'train/lr': current_lr,
                        'train/gpu_memory': gpu_memory,
                    }, epoch * len(train_loader) + batch_idx)
                    
            except Exception as e:
                print(f"Warning: Error in training step: {str(e)}")
                continue
        
        # Validation
        model.set_eval_mode()  # Use custom eval mode
        val_loss, val_metrics = validate(model, val_loader, device, logger, epoch, config)
        
        # Save best model
        if val_metrics['mAP'] > best_map:
            best_map = val_metrics['mAP']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map,
            }, save_dir / 'best_model.pt')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"mAP: {val_metrics['mAP']:.4f}")
        print(f"Best mAP: {best_map:.4f}")

if __name__ == '__main__':
    train()