import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPProcessor, CLIPTokenizer
import kornia.filters as KF
import torch.nn.functional as F

class DiffShieldLoss(nn.Module):
    def __init__(self, device='cpu', sd_model_id="runwayml/stable-diffusion-v1-5",
                 clip_model_id="openai/clip-vit-base-patch32", revision=None,
                 vae=None, clip_vision=None, clip_text=None, clip_tokenizer=None):
        """
        Defaults reproduce v1 exactly (same model ids, fp32, no pinned revision).
        Any component passed in (vae, clip_vision, clip_text, clip_tokenizer) is used
        as-is and frozen; nothing is downloaded for it. This allows CPU unit tests
        with tiny random models and pinning a model revision for reproducibility.
        """
        super().__init__()
        self.device = device
        rev = {"revision": revision} if revision else {}

        # Load VAE for Visual Loss
        if vae is None:
            vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", **rev)
        self.vae = vae
        self.vae.to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Load CLIP Vision Encoder for Semantic Loss (image side)
        if clip_vision is None:
            clip_vision = CLIPVisionModelWithProjection.from_pretrained(clip_model_id)
        self.clip_vision = clip_vision
        self.clip_vision.to(self.device)
        self.clip_vision.eval()
        for param in self.clip_vision.parameters():
            param.requires_grad = False

        # Load CLIP Text Encoder for Semantic Loss (text side)
        if clip_text is None:
            clip_text = CLIPTextModelWithProjection.from_pretrained(clip_model_id)
        self.clip_text = clip_text
        self.clip_text.to(self.device)
        self.clip_text.eval()
        for param in self.clip_text.parameters():
            param.requires_grad = False

        if clip_tokenizer is None:
            try:  # unused by the loss; tolerate processor-loading differences across transformers versions
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
            except Exception:
                self.clip_processor = None
            clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)
        else:
            self.clip_processor = None
        self.clip_tokenizer = clip_tokenizer

        # L_str is a differentiable windowed-SSIM term (see compute_structure_loss).

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

    def _gaussian_window(self, window_size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
        coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        window_2d = g @ g.t()
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def compute_structure_loss(self, clean_image, poisoned_image):
        """
        L_str = 1 - SSIM_windowed(X, X_aug)   (11x11 Gaussian window, sigma=1.5)

        FIDELITY-PRESERVATION term (unlike L_vis / L_sem, which are disruption
        terms). We want PGD to MINIMIZE this quantity, i.e. keep windowed SSIM
        high, so it must be SUBTRACTED in forward(), not added.
        """
        clean_01 = (clean_image + 1.0) / 2.0
        poison_01 = (poisoned_image + 1.0) / 2.0

        C = clean_01.shape[1]
        window = self._gaussian_window(11, 1.5, C, clean_01.device, clean_01.dtype)
        pad = 11 // 2

        mu_x = F.conv2d(clean_01, window, padding=pad, groups=C)
        mu_y = F.conv2d(poison_01, window, padding=pad, groups=C)
        mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

        sigma_x_sq = F.conv2d(clean_01 * clean_01, window, padding=pad, groups=C) - mu_x_sq
        sigma_y_sq = F.conv2d(poison_01 * poison_01, window, padding=pad, groups=C) - mu_y_sq
        sigma_xy = F.conv2d(clean_01 * poison_01, window, padding=pad, groups=C) - mu_xy

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /                    ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        return 1.0 - ssim_map.mean()

    def forward(self, clean_image, poisoned_image, target_concept_embedding, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Tri-Modal Loss: max_delta (alpha*L_vis - beta*L_sem - gamma*L_str)

        Sign convention explanation:
        - L_vis (MSE in VAE latent):  We MAXIMIZE this -> destroys spatial blueprint
        - L_sem (1 - cos_sim):        We MINIMIZE this -> pushes cos_sim toward 1.0
                                       (image embedding -> "a potted plant")
        - L_str (1 - windowed SSIM):  We MINIMIZE this -> PRESERVES structural fidelity

        Since PGD performs gradient ASCENT on total_loss:
        - +alpha*L_vis: ascending maximizes latent divergence (attack)
        - -beta*L_sem:  ascending minimizes (1-cos), i.e. maximizes cos_sim (attack)
        - -gamma*L_str: ascending minimizes (1-SSIM), i.e. keeps SSIM high (preservation)
        """
        l_vis = self.compute_visual_loss(clean_image, poisoned_image)
        l_sem = self.compute_semantic_loss(poisoned_image, target_concept_embedding)
        l_str = self.compute_structure_loss(clean_image, poisoned_image)

        # Objective to MAXIMIZE via PGD gradient ascent
        total_loss = (alpha * l_vis) - (beta * l_sem) - (gamma * l_str)
        return total_loss, l_vis, l_sem, l_str
