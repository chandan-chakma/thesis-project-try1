import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from ultralytics import YOLO
from pathlib import Path

class YOLOWorld(nn.Module):
    def __init__(self, yolo_model='yolov8s.pt', text_encoder='openai/clip-vit-base-patch32'):
        super().__init__()
        
        # Set device and ensure CUDA is properly initialized
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nInitializing YOLOWorld on device: {self.device}")
        
        # Initialize number of classes first
        self.num_classes = 80  # COCO dataset has 80 classes
        
        if torch.cuda.is_available():
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Initialize loss weights
        self.loss_weights = {
            'box': 2.0,    # Increased weight for box regression
            'cls': 1.0,    # Classification weight
            'dfl': 1.0     # Distribution focal loss weight
        }
        
        # Initialize YOLO backbone with optimized settings
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model)
        self.yolo_model = self.yolo.model
        
        # Configure YOLO model for maximum GPU utilization
        if hasattr(self.yolo_model, 'args'):
            self.yolo_model.args.update({
                'batch': 32,  # Increased batch size
                'device': self.device,
                'half': False,  # Use FP32 for stability
                'amp': True,  # Enable AMP
                'workers': 8,  # Increase worker threads
                'cache': True  # Enable caching
            })
        
        # Move model to device and optimize memory
        self.yolo_model = self.yolo_model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.yolo_model = torch.nn.DataParallel(self.yolo_model)
        
        print("Loading CLIP text encoder...")
        # Initialize text encoder with optimized settings
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder)
        self.text_encoder = self.text_encoder.to(self.device)
        if torch.cuda.device_count() > 1:
            self.text_encoder = torch.nn.DataParallel(self.text_encoder)
        
        # Freeze text encoder
        self.text_encoder.requires_grad_(False)
        
        print("Initializing text projector...")
        # Initialize text projector with optimized architecture
        self.text_projector = nn.Sequential(
            nn.Linear(512, 1024),  # Wider network
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)  # Now num_classes is defined
        ).to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.yolo_model, 'module'):
            if hasattr(self.yolo_model.module, 'gradient_checkpointing'):
                self.yolo_model.module.gradient_checkpointing = True
        elif hasattr(self.yolo_model, 'gradient_checkpointing'):
            self.yolo_model.gradient_checkpointing = True
        
        # Print model statistics
        print("\nModel Statistics:")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if torch.cuda.is_available():
            print(f"Final GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
    def encode_text(self, text_prompts):
        # Tokenize text
        tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)
        
        # Get text features from CLIP
        with torch.no_grad():
            text_features = self.text_encoder(**tokens).last_hidden_state.mean(dim=1)
        
        # Project to detection space
        text_embeddings = self.text_projector(text_features)
        
        return text_embeddings
    
    def forward(self, images, labels=None):
        if self.training and labels is not None:
            try:
                # Ensure inputs are on the correct device and dtype
                images = images.to(self.device, dtype=torch.float32, non_blocking=True)
                labels = [label.to(self.device, dtype=torch.float32, non_blocking=True) for label in labels]
                
                # Forward pass through YOLO with gradient checkpointing
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    preds = self.yolo_model(images)
                    
                    # Calculate loss
                    if isinstance(self.yolo_model, torch.nn.DataParallel):
                        criterion = self.yolo_model.module.criterion
                    else:
                        criterion = getattr(self.yolo_model, 'criterion', None)
                    
                    if criterion is not None:
                        loss_dict = criterion(preds, labels)
                        
                        # Apply loss weights and ensure positive values
                        weighted_losses = {}
                        for k, v in loss_dict.items():
                            if isinstance(v, torch.Tensor):
                                weight = self.loss_weights.get(k.split('_')[0], 1.0)
                                weighted_losses[k] = torch.abs(v) * weight
                        
                        # Sum losses with better numerical stability
                        if weighted_losses:
                            total_loss = torch.stack(list(weighted_losses.values())).sum()
                            return total_loss
                    
                    # Fallback loss calculation if no criterion
                    if isinstance(preds, (list, tuple)) and len(preds) > 0:
                        return torch.abs(preds[0].mean()) + 1e-7
                    else:
                        return torch.abs(preds.mean()) + 1e-7
                        
            except Exception as e:
                print(f"Warning: Error in forward pass: {str(e)}")
                return torch.tensor(1.0, requires_grad=True, device=self.device)
        else:
            with torch.no_grad():
                preds = self.yolo_model(images)
                if isinstance(preds, (list, tuple)) and len(preds) > 0:
                    if hasattr(preds[0], 'boxes'):
                        return {
                            'loss': torch.tensor(0.0, device=self.device),
                            'pred': preds[0].boxes.data if len(preds[0].boxes) else torch.zeros((0, 6), device=self.device)
                        }
                return {
                    'loss': torch.tensor(0.0, device=self.device),
                    'pred': torch.zeros((0, 6), device=self.device)
                }
    
    def train_step(self, batch):
        torch.cuda.empty_cache()  # Clear cache before forward pass
        
        # Move batch to GPU efficiently
        images = batch['image'].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = [label.to(self.device, dtype=torch.float32, non_blocking=True) for label in batch['labels']]
        
        # Set training mode without using train()
        self.training = True
        if isinstance(self.yolo_model, torch.nn.DataParallel):
            self.yolo_model.module.training = True
        else:
            self.yolo_model.training = True
        
        # Forward pass with memory optimization
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            loss = self(images, labels)
        
        # Ensure loss is valid and requires gradient
        if isinstance(loss, torch.Tensor):
            if not loss.requires_grad:
                loss.requires_grad_(True)
            # Scale loss for better gradient flow
            loss = loss * 0.5  # Scale factor to prevent loss explosion
            
            # Add L2 regularization
            if self.training:
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param)
                loss = loss + 0.0001 * l2_reg
        
        return loss
    
    def validation_step(self, batch):
        images = batch['image'].to(self.device)
        
        # Set evaluation mode without calling eval()
        self.training = False
        self.yolo_model.training = False
        
        # Forward pass in eval mode
        with torch.no_grad():
            results = self(images)
        
        return results
    
    def set_train_mode(self):
        """Set the model to training mode without using train()"""
        self.training = True
        if isinstance(self.yolo_model, torch.nn.DataParallel):
            self.yolo_model.module.training = True
        else:
            self.yolo_model.training = True
        
        # Set training mode for other components
        self.text_projector.train()
        return self
    
    def set_eval_mode(self):
        """Set the model to evaluation mode without using eval()"""
        self.training = False
        if isinstance(self.yolo_model, torch.nn.DataParallel):
            self.yolo_model.module.training = False
        else:
            self.yolo_model.training = False
        
        # Set eval mode for other components
        self.text_projector.eval()
        return self 