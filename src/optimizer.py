import torch
from tqdm import tqdm
from .eot import EoTLayer
from .losses import DiffShieldLoss

class PGDOptimizer:
    def __init__(self, epsilon=8/255, alpha=2/255, iters=40, device='cpu'):
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.device = device
        
        self.eot = EoTLayer().to(self.device)
        self.loss_fn = DiffShieldLoss(device=self.device)

    def optimize(self, clean_image, target_concept_embedding, 
                 w_alpha=1.0, w_beta=1.0, w_gamma=1.0):
        """
        Runs the PGD optimization loop (gradient ascent on Tri-Modal loss).
        
        Args:
            clean_image: Tensor of shape (1, C, H, W) in range [-1, 1]
            target_concept_embedding: Tensor of shape (1, 512) from CLIP text encoder
            w_alpha, w_beta, w_gamma: Scalar weights for visual, semantic, structural losses
            
        Returns:
            Immunized (poisoned) image tensor in [-1, 1]
        """
        clean_image = clean_image.to(self.device)
        
        # Initialize perturbation with small random noise within epsilon ball
        delta = torch.zeros_like(clean_image).to(self.device)
        delta.uniform_(-self.epsilon, self.epsilon)
        # Clamp so initial perturbed image is in valid range
        delta.data = torch.clamp(clean_image + delta.data, -1.0, 1.0) - clean_image
        delta.requires_grad = True

        for i in range(self.iters):
            # 1. Apply perturbation and clamp to valid image range
            poisoned_image = torch.clamp(clean_image + delta, -1.0, 1.0)
            
            # 2. Apply EoT layer with PAIRED transforms (same random params for both)
            #    This ensures visual/structural losses measure perturbation effect,
            #    not augmentation misalignment.
            clean_image_aug, poisoned_image_aug = self.eot.forward_paired(
                clean_image, poisoned_image
            )
            
            # 3. Calculate joint Tri-Modal loss
            total_loss, l_vis, l_sem, l_str = self.loss_fn(
                clean_image_aug, 
                poisoned_image_aug, 
                target_concept_embedding,
                alpha=w_alpha, beta=w_beta, gamma=w_gamma
            )
            
            # 4. Backpropagate — compute gradients of total_loss w.r.t. delta
            #    PGD ascent: we want to MAXIMIZE total_loss
            total_loss.backward()
            
            # 5. PGD Update (gradient ascent via sign of gradient)
            with torch.no_grad():
                # Get the sign of gradients
                grad_sign = delta.grad.sign()
                
                # Update delta: δ_{t+1} = δ_t + η · sign(∇_δ L)
                delta.data = delta.data + self.alpha * grad_sign
                
                # 6. L∞ Projection: Clamp delta to [-epsilon, epsilon]
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                
                # 7. Clamp the final image to valid [-1, 1] range
                poisoned_image = torch.clamp(clean_image + delta.data, -1.0, 1.0)
                delta.data = poisoned_image - clean_image
                
            # Clear gradients for next iteration
            delta.grad.zero_()
            
            if i % 10 == 0:
                cos_sim_val = 1.0 - l_sem.item()  # L_sem = 1 - cos_sim
                print(f"Iter {i:3d}: Total {total_loss.item():+.4f} | "
                      f"L_vis {l_vis.item():.4f} | L_sem {l_sem.item():.4f} (cos_sim={cos_sim_val:.4f}) | "
                      f"L_str {l_str.item():.4f}")

        final_poisoned_image = torch.clamp(clean_image + delta.data, -1.0, 1.0)
        
        # Verify L∞ constraint
        linf = (final_poisoned_image - clean_image).abs().max().item()
        print(f"\nFinal L∞ norm of perturbation: {linf:.6f} (limit: {self.epsilon:.6f})")
        assert linf <= self.epsilon + 1e-6, f"L∞ violation: {linf} > {self.epsilon}"
        
        return final_poisoned_image.detach()
