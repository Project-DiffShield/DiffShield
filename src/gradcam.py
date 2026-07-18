import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from diffusers import StableDiffusionPipeline
import os


class GradCAMDiffusion:
    """
    Grad-CAM for Stable Diffusion UNet cross-attention layers.
    
    Extracts gradients from the cross-attention layers of the UNet to generate
    heatmaps showing where the model focuses spatially. For a successfully
    immunized image, the heatmap should show scattered/incoherent attention
    rather than focused attention on facial features.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.pipe = None
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def _load_pipeline(self):
        """Load SD 1.5 pipeline if not already loaded."""
        if self.pipe is None:
            print("Loading Stable Diffusion pipeline for Grad-CAM...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
            )
            self.pipe.to(self.device)
            self.pipe.unet.eval()
            # Keep VAE and text encoder frozen
            for param in self.pipe.unet.parameters():
                param.requires_grad = False
    
    def _register_hooks(self, target_layer_name="mid_block"):
        """
        Register forward and backward hooks on cross-attention layers.
        
        Args:
            target_layer_name: Which UNet block to target. Options:
                - "mid_block": Middle block (default, good spatial resolution)
                - "up_blocks.3": Last upsampling block (highest resolution)
        """
        self._clear_hooks()
        
        unet = self.pipe.unet
        
        # Find cross-attention layers in the target block
        target_module = None
        for name, module in unet.named_modules():
            if target_layer_name in name and 'attn2' in name and hasattr(module, 'to_q'):
                # attn2 is the cross-attention layer (text-to-image)
                target_module = module
                layer_name = name
                break
        
        if target_module is None:
            # Fallback: use the mid block's main attention
            for name, module in unet.named_modules():
                if 'mid_block' in name and 'attn' in name and hasattr(module, 'to_q'):
                    target_module = module
                    layer_name = name
                    break
        
        if target_module is None:
            print("Warning: Could not find cross-attention layer for Grad-CAM. "
                  "Using the mid_block output instead.")
            target_module = unet.mid_block
            layer_name = "mid_block"
        
        print(f"Grad-CAM targeting layer: {layer_name}")
        
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations['target'] = output[0].detach()
            else:
                self.activations['target'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients['target'] = grad_output[0].detach()
            else:
                self.gradients['target'] = grad_output.detach()
        
        hook_fwd = target_module.register_forward_hook(forward_hook)
        hook_bwd = target_module.register_full_backward_hook(backward_hook)
        self.hooks.extend([hook_fwd, hook_bwd])
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def generate_heatmap(self, image_tensor, prompt="A photo of a person", 
                         num_inference_steps=5, target_layer="mid_block"):
        """
        Generate a Grad-CAM heatmap for a given image through the UNet.
        
        This works by:
        1. Encoding the image to VAE latent space
        2. Running a single denoising step with the UNet
        3. Computing gradients of the predicted noise w.r.t. cross-attention activations
        4. Generating the Grad-CAM heatmap from activations and gradients
        
        Args:
            image_tensor: Tensor (1, 3, H, W) in [-1, 1]
            prompt: Text prompt to condition the UNet
            num_inference_steps: Not used for single-step (kept for API compat)
            target_layer: Which UNet block to visualize
            
        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
        """
        self._load_pipeline()
        self._register_hooks(target_layer)
        
        image_tensor = image_tensor.to(self.device)
        
        # 1. Encode image to latent space
        with torch.no_grad():
            latents = self.pipe.vae.encode(image_tensor).latent_dist.mean
            latents = latents * self.pipe.vae.config.scaling_factor
        
        # 2. Encode text prompt
        text_inputs = self.pipe.tokenizer(
            prompt, return_tensors="pt", padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_inputs.input_ids)[0]
        
        # 3. Add small noise and run a single denoising step
        # We need gradients to flow through the UNet
        self.pipe.unet.eval()
        
        # Temporarily enable gradients for the target layer
        latents_input = latents.clone().detach().requires_grad_(True)
        
        # Use a small timestep (near the end of denoising)
        timestep = torch.tensor([50], device=self.device)
        
        # Add noise at this timestep
        noise = torch.randn_like(latents_input)
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(1000)
        
        noisy_latents = scheduler.add_noise(latents_input, noise, timestep)
        noisy_latents = noisy_latents.detach().requires_grad_(True)
        
        # Forward pass through UNet with gradient tracking
        for param in self.pipe.unet.parameters():
            param.requires_grad = True
            
        noise_pred = self.pipe.unet(
            noisy_latents, timestep, encoder_hidden_states=text_embeddings
        ).sample
        
        # 4. Compute loss (we use the L2 norm of predicted noise as the target)
        loss = noise_pred.pow(2).sum()
        loss.backward()
        
        # Disable gradients again
        for param in self.pipe.unet.parameters():
            param.requires_grad = False
        
        # 5. Compute Grad-CAM
        if 'target' in self.activations and 'target' in self.gradients:
            activations = self.activations['target']
            gradients = self.gradients['target']
            
            # Global average pooling of gradients to get channel weights
            if gradients.dim() == 4:
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True)
            elif gradients.dim() == 3:
                # Attention output: (B, SeqLen, Dim)
                weights = gradients.mean(dim=1, keepdim=True)
                cam = (weights * activations).sum(dim=-1)
                # Reshape to 2D
                seq_len = cam.shape[-1]
                h = w = int(seq_len ** 0.5)
                if h * w == seq_len:
                    cam = cam.view(1, 1, h, w)
                else:
                    cam = cam.unsqueeze(0).unsqueeze(0)
            else:
                weights = gradients.mean(dim=tuple(range(1, gradients.dim())), keepdim=True)
                cam = (weights * activations).sum(dim=1, keepdim=True)
            
            # ReLU — only positive contributions
            cam = F.relu(cam)
            
            # Normalize to [0, 1]
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to original image size
            H, W = image_tensor.shape[2], image_tensor.shape[3]
            heatmap = F.interpolate(cam.float(), size=(H, W), mode='bilinear', align_corners=False)
            heatmap = heatmap.squeeze().cpu().numpy()
        else:
            print("Warning: Could not capture activations/gradients. Returning blank heatmap.")
            H, W = image_tensor.shape[2], image_tensor.shape[3]
            heatmap = np.zeros((H, W))
        
        self._clear_hooks()
        return heatmap
    
    def visualize_comparison(self, clean_tensor, immunized_tensor, 
                              prompt="A photo of a person", 
                              save_path="outputs/gradcam_comparison.png"):
        """
        Generate side-by-side Grad-CAM comparison for clean vs immunized image.
        
        A successful immunization should show:
        - Clean image: Focused attention on facial features (eyes, nose, mouth)
        - Immunized image: Scattered/incoherent attention across the image
        
        Args:
            clean_tensor: Clean face image (1, 3, H, W) in [-1, 1]
            immunized_tensor: Immunized face image (1, 3, H, W) in [-1, 1]
            prompt: Text prompt for conditioning
            save_path: Where to save the comparison figure
        """
        print("Generating Grad-CAM for clean image...")
        heatmap_clean = self.generate_heatmap(clean_tensor, prompt)
        
        print("Generating Grad-CAM for immunized image...")
        heatmap_immunized = self.generate_heatmap(immunized_tensor, prompt)
        
        # Convert tensors to displayable images [0, 1]
        clean_img = ((clean_tensor.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
        immun_img = ((immunized_tensor.squeeze().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Grad-CAM Analysis: UNet Cross-Attention Focus", fontsize=16, fontweight='bold')
        
        # Row 1: Clean image
        axes[0, 0].imshow(clean_img)
        axes[0, 0].set_title("Clean Image")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(heatmap_clean, cmap='jet')
        axes[0, 1].set_title("Grad-CAM (Clean)")
        axes[0, 1].axis('off')
        
        # Overlay
        heatmap_color_clean = cm.jet(heatmap_clean)[:, :, :3]
        overlay_clean = 0.5 * clean_img + 0.5 * heatmap_color_clean
        axes[0, 2].imshow(overlay_clean.clip(0, 1))
        axes[0, 2].set_title("Overlay (Clean)")
        axes[0, 2].axis('off')
        
        # Row 2: Immunized image
        axes[1, 0].imshow(immun_img)
        axes[1, 0].set_title("Immunized Image")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(heatmap_immunized, cmap='jet')
        axes[1, 1].set_title("Grad-CAM (Immunized)")
        axes[1, 1].axis('off')
        
        # Overlay
        heatmap_color_immun = cm.jet(heatmap_immunized)[:, :, :3]
        overlay_immun = 0.5 * immun_img + 0.5 * heatmap_color_immun
        axes[1, 2].imshow(overlay_immun.clip(0, 1))
        axes[1, 2].set_title("Overlay (Immunized)")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Grad-CAM comparison saved to: {save_path}")
        
        return heatmap_clean, heatmap_immunized
