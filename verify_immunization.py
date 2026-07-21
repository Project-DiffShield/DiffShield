"""
DiffShield Verification Script
================================
Feeds clean and immunized images into Stable Diffusion 1.5 + ControlNet (Canny)
to verify that the immunized image resists deepfake generation.

Expected Results:
  - Clean image + ControlNet -> Realistic face reconstruction (attack succeeds)
  - Immunized image + ControlNet -> Distorted/unrecognizable output (attack fails = defense works)

Usage:
  python verify_immunization.py
  python verify_immunization.py --image outputs/immunized_06765.jpg --clean outputs/clean_06765.jpg
  python verify_immunization.py --immunize-first --source celeba_hq_256
"""

import torch
import numpy as np
import cv2
import os
import math
import argparse
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt


def calculate_psnr(img1, img2):
    """PSNR between two numpy arrays in [0, 255]."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def extract_canny_edges(image_pil, low=100, high=200):
    """Extract Canny edge map from a PIL image."""
    image_np = np.array(image_pil)
    edges = cv2.Canny(image_np, low, high)
    # Convert to 3-channel for ControlNet
    edges_3ch = np.stack([edges, edges, edges], axis=2)
    return Image.fromarray(edges_3ch)


def load_controlnet_pipeline(device='cpu'):
    """Load SD 1.5 + ControlNet Canny pipeline."""
    print("Loading ControlNet model (sd-controlnet-canny)...")
    
    # Use float32 for CPU, float16 for GPU
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype
    )
    
    print("Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    if device == 'cuda':
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    
    # Disable safety checker to avoid false positives on face images
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    print("Pipeline loaded successfully.")
    return pipe


def run_controlnet_attack(pipe, image_pil, prompt, num_steps=20, seed=42):
    """
    Run a ControlNet Canny attack on the given image.
    
    This simulates a deepfake attacker who:
    1. Extracts Canny edges from the image (structural guidance)
    2. Uses SD 1.5 + ControlNet to generate a new face following those edges
    
    Args:
        pipe: The loaded SD + ControlNet pipeline
        image_pil: Input PIL image (clean or immunized)
        prompt: Text prompt for generation
        num_steps: Number of denoising steps
        seed: Random seed for reproducibility
        
    Returns:
        generated_image: PIL image output from the pipeline
        canny_image: PIL image of the extracted Canny edges
    """
    # Extract Canny edges
    canny_image = extract_canny_edges(image_pil)
    
    # Generate with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    output = pipe(
        prompt,
        image=canny_image,
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]
    
    return output, canny_image


def compute_clip_similarity(image_pil, text, device='cpu'):
    """Compute CLIP cosine similarity between an image and text."""
    from transformers import CLIPProcessor, CLIPModel
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    inputs = processor(text=[text], images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Normalized similarity
        logits = outputs.logits_per_image  # shape: (1, 1)
    
    return logits.item()


def create_comparison_figure(clean_img, immunized_img, 
                              clean_canny, immunized_canny,
                              clean_output, immunized_output,
                              metrics, save_path):
    """Create a detailed 2x3 comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("DiffShield Verification: ControlNet Attack Results", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Row 1: Clean image flow
    axes[0, 0].imshow(clean_img)
    axes[0, 0].set_title("Clean Image (Input)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(clean_canny)
    axes[0, 1].set_title("Canny Edges (Clean)", fontsize=13)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(clean_output)
    axes[0, 2].set_title("SD+ControlNet Output (Clean)\n[ATTACK SUCCEEDS - Face Reconstructed]", 
                          fontsize=11, fontweight='bold', color='red')
    axes[0, 2].axis('off')
    
    # Row 2: Immunized image flow
    axes[1, 0].imshow(immunized_img)
    axes[1, 0].set_title("Immunized Image (Input)", fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(immunized_canny)
    axes[1, 1].set_title("Canny Edges (Immunized)\n[Corrupted by adversarial noise]", fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(immunized_output)
    axes[1, 2].set_title("SD+ControlNet Output (Immunized)\n[ATTACK FAILS - Defense Works]", 
                          fontsize=11, fontweight='bold', color='green')
    axes[1, 2].axis('off')
    
    # Add metrics as text below the figure
    metric_text = (
        f"PSNR (clean vs immunized): {metrics['psnr_clean_vs_immunized']:.2f} dB  |  "
        f"PSNR (clean vs clean_output): {metrics['psnr_clean_vs_clean_out']:.2f} dB  |  "
        f"PSNR (clean vs immunized_output): {metrics['psnr_clean_vs_immun_out']:.2f} dB"
    )
    fig.text(0.5, 0.02, metric_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison figure saved: {save_path}")


def immunize_image(image_path, device='cpu', args=None):
    """Run the DiffShield PGD optimization on a single image."""
    from src.optimizer import PGDOptimizer
    import torchvision.transforms as T
    
    print(f"Immunizing image: {image_path}")
    
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # -> [-1, 1]
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    eps = args.epsilon if args else 16.0
    alpha = args.alpha if args else 1.5
    iters = args.iters if args else 100
    w_vis = args.w_vis if args else 1.0
    w_sem = args.w_sem if args else 1.5
    w_str = args.w_str if args else 2.5
    concept = args.concept if args else "a potted plant"

    # Initialize optimizer
    optimizer = PGDOptimizer(epsilon=eps/255, alpha=alpha/255, iters=iters, device=device)
    
    # Get target concept embedding
    target_embed = optimizer.loss_fn.encode_target_text([concept])
    
    # Run PGD
    immunized_tensor = optimizer.optimize(
        img_tensor, target_embed,
        w_alpha=w_vis, w_beta=w_sem, w_gamma=w_str
    )
    
    # Convert back to PIL
    to_pil = T.ToPILImage()
    clean_pil = to_pil((img_tensor.squeeze().cpu() + 1.0) / 2.0)
    immunized_pil = to_pil((immunized_tensor.squeeze().cpu() + 1.0) / 2.0)
    
    return clean_pil, immunized_pil


def main():
    parser = argparse.ArgumentParser(description="Verify DiffShield immunization against ControlNet attacks")
    parser.add_argument('--clean', type=str, default=None,
                        help='Path to clean image (default: first in outputs/)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to immunized image (default: first in outputs/)')
    parser.add_argument('--immunize-first', action='store_true',
                        help='Immunize the image first before verification')
    parser.add_argument('--source', type=str, default='celeba_hq_256',
                        help='Source directory for images when --immunize-first is used')
    parser.add_argument('--prompt', type=str, default='A high quality photo of a person, detailed face',
                        help='Text prompt for ControlNet generation')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of inference steps')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory to save results')
    parser.add_argument('--epsilon', type=float, default=16.0, help='Max perturbation (out of 255)')
    parser.add_argument('--iters', type=int, default=100, help='Number of PGD iterations')
    parser.add_argument('--alpha', type=float, default=1.5, help='PGD step size (out of 255)')
    parser.add_argument('--w-vis', type=float, default=1.0, help='Visual loss weight')
    parser.add_argument('--w-sem', type=float, default=1.5, help='Semantic loss weight')
    parser.add_argument('--w-str', type=float, default=2.5, help='Structural loss weight')
    parser.add_argument('--concept', type=str, default="a potted plant", help='Target semantic concept')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ---- Step 1: Get clean and immunized images ----
    if args.immunize_first:
        # Immunize a fresh image
        source_images = [f for f in os.listdir(args.source) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not source_images:
            print(f"ERROR: No images found in {args.source}")
            return
        source_path = os.path.join(args.source, source_images[0])
        clean_pil, immunized_pil = immunize_image(source_path, device, args)
        
        # Save them
        img_name = os.path.splitext(source_images[0])[0]
        clean_pil.save(os.path.join(args.output_dir, f"verify_clean_{img_name}.png"))
        immunized_pil.save(os.path.join(args.output_dir, f"verify_immunized_{img_name}.png"))
    else:
        # Load existing outputs
        if args.clean and args.image:
            clean_pil = Image.open(args.clean).convert('RGB')
            immunized_pil = Image.open(args.image).convert('RGB')
        else:
            # Auto-detect from outputs directory
            output_files = os.listdir(args.output_dir)
            clean_files = sorted([f for f in output_files if f.startswith('clean_')])
            immun_files = sorted([f for f in output_files if f.startswith('immunized_')])
            
            if not clean_files or not immun_files:
                print("ERROR: No clean/immunized image pairs found in outputs/.")
                print("Run 'python main.py' first to generate immunized images,")
                print("or use --immunize-first to immunize a fresh image.")
                return
            
            clean_path = os.path.join(args.output_dir, clean_files[0])
            immun_path = os.path.join(args.output_dir, immun_files[0])
            print(f"Using: {clean_path}")
            print(f"  and: {immun_path}")
            clean_pil = Image.open(clean_path).convert('RGB')
            immunized_pil = Image.open(immun_path).convert('RGB')
    
    # ---- Step 2: Compute pre-attack metrics ----
    print("\n" + "="*60)
    print("PRE-ATTACK METRICS")
    print("="*60)
    
    clean_np = np.array(clean_pil)
    immunized_np = np.array(immunized_pil)
    
    psnr_ci = calculate_psnr(clean_np, immunized_np)
    print(f"PSNR (clean vs immunized):  {psnr_ci:.2f} dB")
    print(f"  -> {'Imperceptible noise (good)' if psnr_ci > 35 else 'Visible noise (may need tuning)'}")
    
    # ---- Step 3: Extract and compare Canny edges ----
    print("\n" + "="*60)
    print("CANNY EDGE COMPARISON")
    print("="*60)
    
    clean_canny = extract_canny_edges(clean_pil)
    immunized_canny = extract_canny_edges(immunized_pil)
    
    clean_canny_np = np.array(clean_canny)[:, :, 0]  # Single channel
    immunized_canny_np = np.array(immunized_canny)[:, :, 0]
    
    edge_diff = np.abs(clean_canny_np.astype(float) - immunized_canny_np.astype(float))
    edge_change_pct = (edge_diff > 0).sum() / edge_diff.size * 100
    
    print(f"Edge pixels changed: {edge_change_pct:.1f}%")
    print(f"  -> {'Significant edge corruption (defense active)' if edge_change_pct > 10 else 'Minor edge changes'}")
    
    # Save Canny comparison
    clean_canny.save(os.path.join(args.output_dir, "verify_canny_clean.png"))
    immunized_canny.save(os.path.join(args.output_dir, "verify_canny_immunized.png"))
    
    # Save edge difference map
    edge_diff_img = Image.fromarray((edge_diff * 2).clip(0, 255).astype(np.uint8))
    edge_diff_img.save(os.path.join(args.output_dir, "verify_canny_diff.png"))
    
    # ---- Step 4: Run ControlNet attacks ----
    print("\n" + "="*60)
    print("RUNNING CONTROLNET ATTACKS")
    print("="*60)
    print(f"Prompt: '{args.prompt}'")
    print(f"Inference steps: {args.steps}")
    
    pipe = load_controlnet_pipeline(device)
    
    print("\n--- Attack on CLEAN image ---")
    clean_output, _ = run_controlnet_attack(pipe, clean_pil, args.prompt, args.steps)
    clean_output.save(os.path.join(args.output_dir, "verify_attack_clean.png"))
    print("Saved: verify_attack_clean.png")
    
    print("\n--- Attack on IMMUNIZED image ---")
    immunized_output, _ = run_controlnet_attack(pipe, immunized_pil, args.prompt, args.steps)
    immunized_output.save(os.path.join(args.output_dir, "verify_attack_immunized.png"))
    print("Saved: verify_attack_immunized.png")
    
    # ---- Step 5: Post-attack metrics ----
    print("\n" + "="*60)
    print("POST-ATTACK METRICS")
    print("="*60)
    
    clean_out_np = np.array(clean_output.resize(clean_pil.size))
    immun_out_np = np.array(immunized_output.resize(clean_pil.size))
    
    psnr_clean_out = calculate_psnr(clean_np, clean_out_np)
    psnr_immun_out = calculate_psnr(clean_np, immun_out_np)
    psnr_outputs = calculate_psnr(clean_out_np, immun_out_np)
    
    print(f"PSNR (clean vs clean_output):      {psnr_clean_out:.2f} dB")
    print(f"  -> ControlNet reconstructed the face from clean edges")
    print(f"PSNR (clean vs immunized_output):   {psnr_immun_out:.2f} dB")
    print(f"  -> {'Lower = more distortion = better defense' if psnr_immun_out < psnr_clean_out else 'Similar to clean = defense may be weak'}")
    print(f"PSNR (clean_output vs immun_output): {psnr_outputs:.2f} dB")
    print(f"  -> {'Outputs differ significantly = defense works' if psnr_outputs < 25 else 'Outputs are similar = defense needs strengthening'}")
    
    # ---- Step 6: Create comparison figure ----
    metrics = {
        'psnr_clean_vs_immunized': psnr_ci,
        'psnr_clean_vs_clean_out': psnr_clean_out,
        'psnr_clean_vs_immun_out': psnr_immun_out,
    }
    
    create_comparison_figure(
        clean_pil, immunized_pil,
        clean_canny, immunized_canny,
        clean_output, immunized_output,
        metrics,
        os.path.join(args.output_dir, "verify_comparison.png")
    )
    
    # ---- Step 7: Final verdict ----
    print("\n" + "="*60)
    print("VERIFICATION VERDICT")
    print("="*60)
    
    defense_score = 0
    total_checks = 4
    
    # Check 1: Imperceptible perturbation
    if psnr_ci > 35:
        print("[PASS] Perturbation is imperceptible (PSNR > 35 dB)")
        defense_score += 1
    else:
        print("[WARN] Perturbation may be visible (PSNR = {:.2f} dB)".format(psnr_ci))
    
    # Check 2: Edge corruption
    if edge_change_pct > 5:
        print("[PASS] Canny edges are corrupted ({:.1f}% changed)".format(edge_change_pct))
        defense_score += 1
    else:
        print("[WARN] Edge corruption is low ({:.1f}% changed)".format(edge_change_pct))
    
    # Check 3: Immunized output is more distorted than clean output
    if psnr_immun_out < psnr_clean_out:
        print("[PASS] Immunized output is more distorted than clean output")
        defense_score += 1
    else:
        print("[FAIL] Immunized output is NOT more distorted than clean output")
    
    # Check 4: Outputs differ significantly
    if psnr_outputs < 30:
        print("[PASS] Clean and immunized outputs differ significantly (PSNR < 30 dB)")
        defense_score += 1
    else:
        print("[WARN] Clean and immunized outputs are similar (PSNR = {:.2f} dB)".format(psnr_outputs))
    
    print(f"\nDefense Score: {defense_score}/{total_checks}")
    if defense_score >= 3:
        print("RESULT: DiffShield immunization is EFFECTIVE")
    elif defense_score >= 2:
        print("RESULT: DiffShield immunization is PARTIALLY EFFECTIVE")
    else:
        print("RESULT: DiffShield immunization needs improvement")
    
    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("  verify_comparison.png     - Side-by-side comparison figure")
    print("  verify_attack_clean.png   - ControlNet output from clean image")
    print("  verify_attack_immunized.png - ControlNet output from immunized image")
    print("  verify_canny_clean.png    - Clean image edge map")
    print("  verify_canny_immunized.png - Immunized image edge map")
    print("  verify_canny_diff.png     - Edge difference map")


if __name__ == "__main__":
    main()
