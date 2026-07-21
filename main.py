import torch
import torchvision.transforms as T
import numpy as np
import os
import math
import argparse
from PIL import Image

from src.data import get_dataloader
from src.optimizer import PGDOptimizer
from src.eval import Evaluator, calculate_psnr


def calculate_ssim(img1, img2, window_size=11):
    """
    Calculate SSIM between two numpy arrays in [0, 255].
    Simplified implementation without external dependencies.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def main():
    parser = argparse.ArgumentParser(description="Run DiffShield Benchmark")
    parser.add_argument('--epsilon', type=float, default=16.0, help='Max perturbation (out of 255)')
    parser.add_argument('--iters', type=int, default=100, help='Number of PGD iterations')
    parser.add_argument('--alpha', type=float, default=1.5, help='PGD step size (out of 255)')
    parser.add_argument('--w-vis', type=float, default=1.0, help='Visual loss weight (w_alpha)')
    parser.add_argument('--w-sem', type=float, default=1.5, help='Semantic loss weight (w_beta)')
    parser.add_argument('--w-str', type=float, default=2.5, help='Structural loss weight (w_gamma)')
    parser.add_argument('--concept', type=str, default="a potted plant", help='Target semantic concept to shift towards')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running DiffShield on device: {device}")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load dataset
    dataloader = get_dataloader(root_dir='celeba_hq_256', batch_size=1)
    
    # Initialize optimizer (PGD) and evaluator
    print(f"Hyperparameters: eps={args.epsilon}/255, iters={args.iters}, alpha={args.alpha}/255")
    print(f"Loss weights: visual={args.w_vis}, semantic={args.w_sem}, structural={args.w_str}")
    optimizer = PGDOptimizer(epsilon=args.epsilon/255, alpha=args.alpha/255, iters=args.iters, device=device)
    evaluator = Evaluator(device=device)
    
    # Generate proper target concept embedding using CLIP Text Encoder
    target_text = [args.concept]
    print(f"Encoding target concept: {target_text}")
    target_concept_embedding = optimizer.loss_fn.encode_target_text(target_text)
    print(f"Target embedding shape: {target_concept_embedding.shape}")
    
    # Process images from dataset (up to 50 for benchmark)
    num_images = min(50, len(dataloader.dataset))
    print(f"\nProcessing {num_images} images...")
    
    psnr_values = []
    ssim_values = []
    
    for idx, (clean_image_tensor, img_path) in enumerate(dataloader):
        if idx >= num_images:
            break
            
        clean_image_tensor = clean_image_tensor.to(device)
        img_name = os.path.basename(img_path[0])
        
        print(f"\n{'='*60}")
        print(f"Image {idx+1}/{num_images}: {img_name}")
        print(f"{'='*60}")
        
        print("Starting optimization...")
        immunized_image_tensor = optimizer.optimize(
            clean_image_tensor, 
            target_concept_embedding,
            w_alpha=args.w_vis, 
            w_beta=args.w_sem, 
            w_gamma=args.w_str
        )
        print("Optimization complete.")
        
        # Convert tensors to PIL for evaluation and saving
        transform_to_pil = T.ToPILImage()
        # Convert from [-1, 1] to [0, 1]
        clean_img_pil = transform_to_pil((clean_image_tensor.squeeze().cpu() + 1.0) / 2.0)
        immunized_img_pil = transform_to_pil((immunized_image_tensor.squeeze().cpu() + 1.0) / 2.0)
        
        # Calculate quality metrics
        clean_np = np.array(clean_img_pil)
        immunized_np = np.array(immunized_img_pil)
        
        psnr_val = calculate_psnr(clean_np, immunized_np)
        ssim_val = calculate_ssim(clean_np, immunized_np)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        
        print(f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        
        # Save outputs for first few images
        if idx < 5:
            clean_img_pil.save(f"outputs/clean_{img_name}")
            immunized_img_pil.save(f"outputs/immunized_{img_name}")
            
            # Save amplified difference map
            diff = np.abs(immunized_np.astype(np.float32) - clean_np.astype(np.float32))
            diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)
            Image.fromarray(diff_amplified).save(f"outputs/diff_{img_name}")
            
            print(f"Saved: outputs/clean_{img_name}, outputs/immunized_{img_name}, outputs/diff_{img_name}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY ({num_images} images)")
    print(f"{'='*60}")
    print(f"PSNR  - Mean: {np.mean(psnr_values):.2f} dB | Min: {np.min(psnr_values):.2f} | Max: {np.max(psnr_values):.2f}")
    print(f"SSIM  - Mean: {np.mean(ssim_values):.4f} | Min: {np.min(ssim_values):.4f} | Max: {np.max(ssim_values):.4f}")
    print(f"Target: PSNR > 40 dB, SSIM > 0.95")
    print(f"Result: PSNR {'PASS' if np.mean(psnr_values) > 40 else 'FAIL'} | "
          f"SSIM {'PASS' if np.mean(ssim_values) > 0.95 else 'FAIL'}")
    
    # Optional: Run Grad-CAM analysis on first image
    print(f"\n{'='*60}")
    print("Running Grad-CAM XAI analysis on first image...")
    print(f"{'='*60}")
    try:
        from src.gradcam import GradCAMDiffusion
        gradcam = GradCAMDiffusion(device=device)
        
        # Reload first image
        first_clean = next(iter(dataloader))[0].to(device)
        first_immunized = optimizer.optimize(
            first_clean, target_concept_embedding,
            w_alpha=args.w_vis, w_beta=args.w_sem, w_gamma=args.w_str
        )
        
        gradcam.visualize_comparison(
            first_clean, first_immunized,
            prompt="A photo of a person",
            save_path="outputs/gradcam_comparison.png"
        )
    except Exception as e:
        print(f"Grad-CAM analysis skipped due to error: {e}")
        print("(This is normal on CPU with limited memory)")


if __name__ == "__main__":
    main()
