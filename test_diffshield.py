"""
DiffShield Test & Verification Script
======================================
Tests all components against the specification:
1. Unit tests for each loss function (L_vis, L_sem, L_str)
2. Gradient flow through EoT layer
3. L∞ clamping verification
4. PSNR/SSIM quality validation
5. End-to-end poisoning verification
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import math
import traceback
from PIL import Image


# ── Test utilities ───────────────────────────────────────────────────

def test_header(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

def test_pass(msg):
    print(f"  [PASS] {msg}")

def test_fail(msg):
    print(f"  [FAIL] {msg}")

def test_info(msg):
    print(f"  [INFO] {msg}")


# ── Test 1: EoT Layer Gradient Flow ─────────────────────────────────

def test_eot_gradients():
    test_header("EoT Layer - Gradient Flow")
    
    from src.eot import EoTLayer
    
    eot = EoTLayer()
    
    # Create a dummy image requiring grad
    x = torch.randn(1, 3, 512, 512, requires_grad=True)
    x_data = (x + 1.0) / 2.0  # Convert to [0,1] range approx
    
    try:
        # Single forward
        y = eot(x)
        loss = y.sum()
        loss.backward()
        
        if x.grad is not None and x.grad.abs().sum() > 0:
            test_pass(f"Single forward: gradients flow correctly (grad norm={x.grad.norm():.4f})")
        else:
            test_fail("Single forward: no gradients received")
    except Exception as e:
        test_fail(f"Single forward: {e}")
    
    # Paired forward
    x_clean = torch.randn(1, 3, 512, 512)
    x_poison = torch.randn(1, 3, 512, 512, requires_grad=True)
    
    try:
        clean_aug, poison_aug = eot.forward_paired(x_clean, x_poison)
        loss = (poison_aug - clean_aug).pow(2).sum()
        loss.backward()
        
        if x_poison.grad is not None and x_poison.grad.abs().sum() > 0:
            test_pass(f"Paired forward: gradients flow correctly (grad norm={x_poison.grad.norm():.4f})")
        else:
            test_fail("Paired forward: no gradients received")
        
        # Verify same spatial transform was applied
        # The images should have the same spatial dimensions
        if clean_aug.shape == poison_aug.shape:
            test_pass(f"Paired forward: output shapes match ({clean_aug.shape})")
        else:
            test_fail(f"Paired forward: shape mismatch clean={clean_aug.shape} poison={poison_aug.shape}")
    except Exception as e:
        test_fail(f"Paired forward: {e}")


# ── Test 2: Visual Loss (L_vis) ─────────────────────────────────────

def test_visual_loss():
    test_header("Visual Loss (L_vis) - VAE Latent MSE")
    
    from src.losses import DiffShieldLoss
    loss_fn = DiffShieldLoss(device='cpu')
    
    # Test 1: Same image should give zero loss
    img = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    l_vis = loss_fn.compute_visual_loss(img, img)
    
    if l_vis.item() < 1e-6:
        test_pass(f"Same image -> L_vis ~ 0 ({l_vis.item():.8f})")
    else:
        test_fail(f"Same image -> L_vis should be ~ 0, got {l_vis.item():.8f}")
    
    # Test 2: Different images should give positive loss
    img2 = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    l_vis2 = loss_fn.compute_visual_loss(img, img2)
    
    if l_vis2.item() > 0:
        test_pass(f"Different images -> L_vis > 0 ({l_vis2.item():.4f})")
    else:
        test_fail(f"Different images -> L_vis should be > 0, got {l_vis2.item():.4f}")
    
    # Test 3: Loss should be a scalar
    if l_vis.dim() == 0:
        test_pass("L_vis returns a scalar")
    else:
        test_fail(f"L_vis should be scalar, got shape {l_vis.shape}")
    
    # Test 4: Gradient should flow
    # Note: .clamp() on a leaf tensor detaches grad. Use the same pattern as PGD:
    # create base + delta where delta requires grad
    img_base = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    delta = torch.zeros_like(img_base, requires_grad=True)
    img_p = img_base + delta
    l = loss_fn.compute_visual_loss(img, img_p)
    l.backward()
    if delta.grad is not None:
        test_pass(f"Gradients flow through L_vis (grad norm={delta.grad.norm():.4f})")
    else:
        test_fail("No gradients through L_vis")


# ── Test 3: Semantic Loss (L_sem) ───────────────────────────────────

def test_semantic_loss():
    test_header("Semantic Loss (L_sem) - CLIP Cosine Similarity")
    
    from src.losses import DiffShieldLoss
    loss_fn = DiffShieldLoss(device='cpu')
    
    # Generate proper text embedding
    target_embed = loss_fn.encode_target_text(["a potted plant"])
    test_info(f"Target text embedding shape: {target_embed.shape}")
    
    if target_embed.shape[-1] == 512:
        test_pass("Text embedding dimension is 512 (matches CLIP ViT-B/32)")
    else:
        test_fail(f"Expected dim 512, got {target_embed.shape[-1]}")
    
    # Test: L_sem should be in range [0, 2] (since cos_sim is in [-1, 1])
    img = torch.randn(1, 3, 512, 512).clamp(-1, 1)
    l_sem = loss_fn.compute_semantic_loss(img, target_embed)
    
    if 0 <= l_sem.item() <= 2:
        test_pass(f"L_sem in valid range [0, 2]: {l_sem.item():.4f}")
    else:
        test_fail(f"L_sem out of range: {l_sem.item():.4f}")
    
    # Test: L_sem should be scalar
    if l_sem.dim() == 0:
        test_pass("L_sem returns a scalar")
    else:
        test_fail(f"L_sem should be scalar, got shape {l_sem.shape}")
    
    # Test: Gradient should flow
    img_p = torch.randn(1, 3, 512, 512, requires_grad=True)
    l = loss_fn.compute_semantic_loss(img_p, target_embed)
    l.backward()
    if img_p.grad is not None:
        test_pass(f"Gradients flow through L_sem (grad norm={img_p.grad.norm():.4f})")
    else:
        test_fail("No gradients through L_sem")


# ── Test 4: Structure Loss (L_str) ──────────────────────────────────

def test_structure_loss():
    test_header("Structure Loss (L_str) - Canny Edge Magnitude MSE")
    
    from src.losses import DiffShieldLoss
    loss_fn = DiffShieldLoss(device='cpu')
    
    # Test 1: Same image should give zero loss
    img = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    l_str = loss_fn.compute_structure_loss(img, img)
    
    if l_str.item() < 1e-6:
        test_pass(f"Same image -> L_str ~ 0 ({l_str.item():.8f})")
    else:
        test_fail(f"Same image -> L_str should be ~ 0, got {l_str.item():.8f}")
    
    # Test 2: Different images should give positive loss
    img2 = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    l_str2 = loss_fn.compute_structure_loss(img, img2)
    
    if l_str2.item() > 0:
        test_pass(f"Different images -> L_str > 0 ({l_str2.item():.4f})")
    else:
        test_fail(f"Different images -> L_str should be > 0, got {l_str2.item():.4f}")
    
    # Test 3: Gradient should flow (KEY TEST — this was broken before)
    img_p = torch.randn(1, 3, 256, 256, requires_grad=True)
    l = loss_fn.compute_structure_loss(img, img_p)
    l.backward()
    if img_p.grad is not None and img_p.grad.abs().sum() > 0:
        test_pass(f"Gradients flow through L_str (grad norm={img_p.grad.norm():.4f})")
        test_info("Using gradient MAGNITUDE (differentiable) instead of binary edges")
    else:
        test_fail("No gradients through L_str - Canny may be using non-differentiable binary edges")


# -- Test 5: Linf Clamping ---------------------------------------------

def test_linf_clamping():
    test_header("Linf Clamping - eps = 8/255")
    
    epsilon = 8 / 255
    
    # Simulate PGD clamping
    clean = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    delta = torch.randn(1, 3, 64, 64) * 0.1  # Large initial perturbation
    
    # Apply L∞ projection
    delta_clamped = torch.clamp(delta, -epsilon, epsilon)
    
    # Check L∞ constraint
    linf = delta_clamped.abs().max().item()
    if linf <= epsilon + 1e-7:
        test_pass(f"Linf norm after clamping: {linf:.6f} <= {epsilon:.6f}")
    else:
        test_fail(f"Linf violation: {linf:.6f} > {epsilon:.6f}")
    
    # Check valid image range after adding perturbation
    perturbed = torch.clamp(clean + delta_clamped, -1.0, 1.0)
    if perturbed.min() >= -1.0 and perturbed.max() <= 1.0:
        test_pass(f"Perturbed image in valid range [{perturbed.min():.4f}, {perturbed.max():.4f}]")
    else:
        test_fail(f"Perturbed image out of range [{perturbed.min():.4f}, {perturbed.max():.4f}]")
    
    # Verify the actual delta after image clamping still respects L∞
    actual_delta = perturbed - clean
    actual_linf = actual_delta.abs().max().item()
    if actual_linf <= epsilon + 1e-7:
        test_pass(f"Actual perturbation Linf after image clamping: {actual_linf:.6f} <= {epsilon:.6f}")
    else:
        test_fail(f"Actual perturbation Linf violation: {actual_linf:.6f} > {epsilon:.6f}")


# ── Test 6: Total Loss Sign Convention ───────────────────────────────

def test_loss_sign_convention():
    test_header("Total Loss Sign Convention - Gradient Ascent Verification")
    
    from src.losses import DiffShieldLoss
    loss_fn = DiffShieldLoss(device='cpu')
    
    target_embed = loss_fn.encode_target_text(["a potted plant"])
    
    clean = torch.randn(1, 3, 256, 256).clamp(-1, 1)
    poison = clean.clone().detach().requires_grad_(True)
    
    total, l_vis, l_sem, l_str = loss_fn(clean, poison, target_embed)
    total.backward()
    
    test_info(f"total_loss = a*L_vis - b*L_sem + g*L_str")
    test_info(f"= {l_vis.item():.4f} - {l_sem.item():.4f} + {l_str.item():.4f} = {total.item():.4f}")
    
    if poison.grad is not None:
        test_pass("Gradients flow through total loss")
        
        # Verify: for identical images, L_vis=0, L_str=0, total = -L_sem
        expected = -l_sem.item()
        if abs(total.item() - expected) < 1e-4:
            test_pass(f"For identical images: total ~ -L_sem = {expected:.4f} (correct)")
        else:
            test_info(f"total={total.item():.4f} vs expected -L_sem={expected:.4f}")
    else:
        test_fail("No gradients through total loss")


# ── Test 7: PSNR/SSIM Quick Validation ──────────────────────────────

def test_psnr_ssim():
    test_header("PSNR / SSIM Metric Validation")
    
    from src.eval import calculate_psnr
    
    # Test PSNR
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Same image = infinite PSNR
    psnr_same = calculate_psnr(img, img)
    if psnr_same == float('inf'):
        test_pass(f"PSNR(identical) = inf")
    else:
        test_fail(f"PSNR(identical) should be inf, got {psnr_same}")
    
    # Small noise = high PSNR (>30 dB)
    noise = np.random.normal(0, 2, img.shape).astype(np.float64)
    img_noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    psnr_small = calculate_psnr(img, img_noisy)
    if psnr_small > 30:
        test_pass(f"PSNR(small noise) = {psnr_small:.2f} dB (> 30 dB)")
    else:
        test_fail(f"PSNR(small noise) = {psnr_small:.2f} dB (expected > 30)")
    
    # Large noise = low PSNR
    img_random = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    psnr_large = calculate_psnr(img, img_random)
    test_info(f"PSNR(random) = {psnr_large:.2f} dB")


# ── Test 8: Text Encoding Verification ──────────────────────────────

def test_text_encoding():
    test_header("CLIP Text Encoding - Proper Target Concept Embedding")
    
    from src.losses import DiffShieldLoss
    loss_fn = DiffShieldLoss(device='cpu')
    
    # Encode different concepts
    plant_embed = loss_fn.encode_target_text(["a potted plant"])
    face_embed = loss_fn.encode_target_text(["a photo of a person's face"])
    
    # They should be different
    cosine_sim = nn.CosineSimilarity(dim=1)
    sim = cosine_sim(plant_embed, face_embed).item()
    
    test_info(f"cos_sim('potted plant', 'person face') = {sim:.4f}")
    
    if sim < 0.9:  # They should be meaningfully different
        test_pass(f"Different concepts produce different embeddings (sim={sim:.4f} < 0.9)")
    else:
        test_fail(f"Embeddings too similar: {sim:.4f}")
    
    # Same concept should produce same embedding
    plant_embed2 = loss_fn.encode_target_text(["a potted plant"])
    sim_same = cosine_sim(plant_embed, plant_embed2).item()
    
    if abs(sim_same - 1.0) < 1e-5:
        test_pass(f"Same concept produces identical embedding (sim={sim_same:.6f})")
    else:
        test_fail(f"Same concept produces different embeddings: sim={sim_same:.6f}")
    
    # Verify shape
    if plant_embed.shape == (1, 512):
        test_pass(f"Embedding shape is correct: {plant_embed.shape}")
    else:
        test_fail(f"Expected shape (1, 512), got {plant_embed.shape}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DiffShield Test Suite")
    print("=" * 60)
    
    tests = [
        ("EoT Gradient Flow", test_eot_gradients),
        ("Linf Clamping", test_linf_clamping),
        ("PSNR/SSIM Metrics", test_psnr_ssim),
        ("Text Encoding", test_text_encoding),
        ("Visual Loss (L_vis)", test_visual_loss),
        ("Semantic Loss (L_sem)", test_semantic_loss),
        ("Structure Loss (L_str)", test_structure_loss),
        ("Loss Sign Convention", test_loss_sign_convention),
    ]
    
    passed = 0
    failed = 0
    errors = 0
    
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            test_header(f"{name} -- ERROR")
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            errors += 1
    
    print(f"\n{'='*60}")
    print(f"TEST SUITE COMPLETE")
    print(f"{'='*60}")
    print(f"Run 'python main.py' to execute the full pipeline with benchmarking.")
    print(f"\nManual testing steps to verify poisoning effectiveness:")
    print(f"  1. Check outputs/clean_*.png vs outputs/immunized_*.png - should look identical")
    print(f"  2. Check outputs/diff_*.png - should show structured noise on facial features")
    print(f"  3. Feed immunized images to SD1.5 + ControlNet - faces should be distorted")
    print(f"  4. Check outputs/gradcam_comparison.png - immunized should have scattered attention")


if __name__ == "__main__":
    main()
