import torch
import torch.nn as nn
import kornia.augmentation as K

class EoTLayer(nn.Module):
    """
    Expectation over Transformation (EoT) Layer.
    Applies differentiable transformations during the forward pass to ensure
    robustness against digital compression (e.g., JPEG, cropping).
    
    Supports a paired mode where the SAME random transform parameters are
    applied to two inputs (clean + poisoned) to maintain spatial alignment.
    """
    def __init__(self, jpeg_quality=(50, 100), crop_scale=(0.8, 1.0), image_size=512):
        super().__init__()
        self.image_size = image_size
        # Differentiable JPEG compression
        self.jpeg = K.RandomJPEG(jpeg_quality=jpeg_quality, p=0.5)
        # Random resized crop for robustness against cropping/resizing
        self.crop = K.RandomResizedCrop(
            size=(image_size, image_size), scale=crop_scale, p=0.5
        )

    def forward(self, x):
        """
        Apply random augmentations to a single input.
        
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W) in range [-1, 1].
        Returns:
            Augmented tensor in [-1, 1].
        """
        # Convert [-1, 1] to [0, 1] for Kornia
        x_norm = (x + 1.0) / 2.0
        
        # Apply augmentations
        x_aug = self.crop(x_norm)
        x_aug = self.jpeg(x_aug)
        
        # Convert back to [-1, 1]
        x_out = (x_aug * 2.0) - 1.0
        return x_out

    def forward_paired(self, x_clean, x_poisoned):
        """
        Apply the SAME random augmentation to both clean and poisoned images.
        This ensures spatial alignment is maintained, so losses measure the
        actual perturbation effect rather than augmentation misalignment.
        
        Args:
            x_clean (Tensor): Clean image tensor (B, C, H, W) in [-1, 1].
            x_poisoned (Tensor): Poisoned image tensor (B, C, H, W) in [-1, 1].
        Returns:
            Tuple of (augmented_clean, augmented_poisoned), both in [-1, 1].
        """
        # Convert [-1, 1] to [0, 1] for Kornia
        clean_01 = (x_clean + 1.0) / 2.0
        poison_01 = (x_poisoned + 1.0) / 2.0
        
        # Concatenate along batch dimension so the SAME random params apply to both
        batch = torch.cat([clean_01, poison_01], dim=0)
        
        # Apply crop — Kornia applies the same transform to the entire batch
        batch_aug = self.crop(batch)
        # Apply JPEG
        batch_aug = self.jpeg(batch_aug)
        
        # Split back
        B = x_clean.shape[0]
        clean_aug = batch_aug[:B]
        poison_aug = batch_aug[B:]
        
        # Convert back to [-1, 1]
        clean_out = (clean_aug * 2.0) - 1.0
        poison_out = (poison_aug * 2.0) - 1.0
        
        return clean_out, poison_out
