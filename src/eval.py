import torch
import torch.nn as nn
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import cv2
import numpy as np
import math

def calculate_psnr(img1, img2):
    # img1 and img2 are expected to be numpy arrays in [0, 255]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.pipe = None

    def _load_models(self):
        if self.pipe is None:
            print("Loading ControlNet and Stable Diffusion models...")
            # Load ControlNet Pipeline for testing attacks
            controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
            )
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.enable_model_cpu_offload() # memory efficiency

    def run_controlnet_attack(self, image_pil, prompt="A photo of a person"):
        """
        Runs the immunized image through ControlNet to see if the identity is preserved (attack fails)
        or if the geometric override was successful (attack succeeds).
        """
        self._load_models()
        # Get Canny edges from the PIL image
        image_np = np.array(image_pil)
        low_threshold = 100
        high_threshold = 200
        canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        from PIL import Image
        canny_image_pil = Image.fromarray(canny_image)

        # Generate image
        output = self.pipe(
            prompt,
            image=canny_image_pil,
            num_inference_steps=20,
        ).images[0]
        
        return output
