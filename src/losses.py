import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPProcessor, CLIPTokenizer
import kornia.filters as KF

class DiffShieldLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # Load VAE for Visual Loss
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Load CLIP Vision Encoder for Semantic Loss (image side)
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_vision.to(self.device)
        self.clip_vision.eval()
        for param in self.clip_vision.parameters():
            param.requires_grad = False
        
        # Load CLIP Text Encoder for Semantic Loss (text side)
        self.clip_text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text.to(self.device)
        self.clip_text.eval()
        for param in self.clip_text.parameters():
            param.requires_grad = False
            
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Canny edge detector is natively differentiable via Kornia
        # We use the gradient magnitude (first return value), NOT the thresholded binary edges,
        # because binary thresholding is non-differentiable and blocks gradient flow.
        
        self.mse_loss = nn.MSELoss()
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def encode_target_text(self, text_prompts):
        """
        Properly encode target text using CLIPTextModelWithProjection.
        Returns text embeddings in the same space as image embeddings (projected).
        
        Args:
            text_prompts: List of strings, e.g. ["a potted plant"]
        Returns:
            text_embeds: Tensor of shape (B, 512) — projected text embeddings
        """
        inputs = self.clip_tokenizer(text_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_outputs = self.clip_text(**inputs)
        # text_embeds is the projected output (same space as image_embeds)
        return text_outputs.text_embeds

    def compute_visual_loss(self, clean_image, poisoned_image):
        """
        L_vis = ||E_v(X_aug) - E_v(X)||^2_2
        
        Computes MSE in VAE latent space to destroy the spatial blueprint
        that Stable Diffusion uses for image generation.
        """
        # Encode clean and poisoned images to latent space
        # VAE expects inputs in [-1, 1]
        clean_latents = self.vae.encode(clean_image).latent_dist.mean
        poison_latents = self.vae.encode(poisoned_image).latent_dist.mean
        return self.mse_loss(poison_latents, clean_latents)

    def compute_semantic_loss(self, poisoned_image, target_concept_embedding):
        """
        L_sem = 1 - cos(E_img, E_txt)
        
        Measures how far the poisoned image's CLIP embedding is from the target
        concept ("a potted plant"). We want to MINIMIZE this value to MAXIMIZE
        cosine similarity, severing the identity-language link.
        """
        # Convert [-1, 1] to [0, 1] then normalize for CLIP
        poisoned_image_01 = (poisoned_image + 1.0) / 2.0
        
        # Resize to 224x224 for CLIP ViT-B/32
        clip_resized = torch.nn.functional.interpolate(
            poisoned_image_01, size=(224, 224), mode='bicubic', align_corners=False
        )
        
        # Normalize using CLIP's expected ImageNet stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        clip_input = (clip_resized - mean) / std
        
        image_embeddings = self.clip_vision(clip_input).image_embeds
        
        cos_sim = self.cosine_sim(image_embeddings, target_concept_embedding)
        return (1 - cos_sim).mean()

    def compute_structure_loss(self, clean_image, poisoned_image):
        """
        L_str = ||Canny_mag(X_aug) - Canny_mag(X)||^2_2
        
        Uses the gradient MAGNITUDE from Kornia's Canny detector (not the binary
        thresholded edges) because:
        - Gradient magnitude is fully differentiable (smooth function of pixel values)
        - Binary thresholding involves non-differentiable operations (NMS + hysteresis)
        - The magnitude captures edge strength continuously, giving useful gradients
        
        Maximizing this MSE injects fake micro-edges that disrupt ControlNet's
        geometric constraints.
        """
        # Kornia Canny expects [0, 1] range
        clean_01 = (clean_image + 1.0) / 2.0
        poison_01 = (poisoned_image + 1.0) / 2.0
        
        # canny() returns (magnitude, edges) — we use magnitude for differentiability
        clean_magnitude, _ = KF.canny(clean_01)
        poison_magnitude, _ = KF.canny(poison_01)
        
        return self.mse_loss(poison_magnitude, clean_magnitude)

    def forward(self, clean_image, poisoned_image, target_concept_embedding, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Tri-Modal Loss: max_δ (α·L_vis - β·L_sem + γ·L_str)
        
        Sign convention explanation:
        - L_vis (MSE in VAE latent): We MAXIMIZE this → destroys spatial blueprint
        - L_sem (1 - cos_sim):       We MINIMIZE this → pushes cos_sim toward 1.0
                                      (image embedding → "a potted plant")
        - L_str (MSE of edge magnitudes): We MAXIMIZE this → injects fake edges
        
        Since PGD performs gradient ASCENT on total_loss:
        - +α·L_vis: ascending maximizes latent divergence ✓
        - -β·L_sem: ascending minimizes (1-cos), i.e. maximizes cos_sim ✓
        - +γ·L_str: ascending maximizes edge divergence ✓
        """
        l_vis = self.compute_visual_loss(clean_image, poisoned_image)
        l_sem = self.compute_semantic_loss(poisoned_image, target_concept_embedding)
        l_str = self.compute_structure_loss(clean_image, poisoned_image)
        
        # Objective to MAXIMIZE via PGD gradient ascent
        total_loss = (alpha * l_vis) - (beta * l_sem) + (gamma * l_str)
        return total_loss, l_vis, l_sem, l_str
