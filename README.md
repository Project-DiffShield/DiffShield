# DiffShield: A Tri-Modal Adversarial Framework for Robust Immunization Against Deepfake Generation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Diffusion Defense](https://img.shields.io/badge/Domain-Adversarial%20ML%20%2F%20GenAI-green.svg)]()

DiffShield is a proactive facial immunization framework designed to protect digital facial portraits against unauthorized manipulation and synthetic forgery by Latent Diffusion Models (LDMs, e.g., Stable Diffusion). Rather than relying on passive post-hoc forensic detection, DiffShield injects an imperceptible, bounded adversarial perturbation (&ell;&infin; &le; 8/255) into raw facial images prior to public sharing. When an adversary attempts image-to-image manipulation or concept inpainting on a protected asset, DiffShield triggers a cross-attention breakdown, causing generation collapse and redirecting output semantics toward an orthogonal target class.

---

## Key Features

* **Tri-Modal Loss Objective:** Simultaneously balances pixel-level visual fidelity (L_vis), anatomical structural preservation (L_str), and orthogonal latent redirection (L_sem) via a multi-objective loss manifold.
* **Semantic Latent Hijacking:** Steers deep image representations in CLIP ViT-B/32 joint embedding space toward unrelated concept representations (e.g., `"a potted plant"`), inducing generation collapse in diffusion cross-attention modules.
* **Expectation Over Transformation (EOT):** Incorporates differentiable spatial transformations within the optimization loop (`src/eot.py`) to prevent gradient obfuscation and defend against downstream image processing pipelines (e.g., resizing, Gaussian filtering, compression).
* **Automated Hyperparameter Calibration:** Eliminates manual weighting using Bayesian optimization (Optuna TPE) across mathematically selected calibration coreset archetypes derived from dataset attribute spaces.
* **Rigorous Evaluation Suite:** Quantifies human imperceptibility in Phase 1 (PSNR, SSIM, LPIPS) and generative breakdown in Phase 2 (CLIP alignment failure, ArcFace/InsightFace cosine identity distance, generation MSE).
* **Interpretability via Grad-CAM:** Inspects adversarial gradient spatial distribution across intermediate network feature representations (`src/gradcam.py`).

---

## Theoretical Framework & Mathematical Formulation

DiffShield computes an imperceptible perturbation **&delta;** bounded by an &ell;&infin; norm ball **S = { &delta; | ||&delta;||_&infin; &le; &epsilon; }** added to an input clean face **x &isin; [-1, 1]^(3 &times; H &times; W)**, yielding the protected face **x_adv = x + &delta;**.

### 1. Tri-Modal Optimization Objective

```text
L_total(x, x_adv, c*) = α · L_vis(x, x_adv) + β · L_sem(x_adv, c*) + γ · L_str(x, x_adv)
```

* **Visual Invisibility Loss (L_vis):** Normalized pixel-level &ell;2 distance enforcing strict preservation of original color profiles:
  * `L_vis = (1 / (C · H · W)) · Σ (x - x_adv)²`
* **Semantic Redirection Loss (L_sem):** Maximizes cosine alignment between the normalized visual embedding `f_θ(x_adv)` and an orthogonal textual concept vector `g_φ(c*)` within CLIP joint representation space:
  * `L_sem = 1 - [ ⟨f_θ(x_adv), g_φ(c*)⟩ / (||f_θ(x_adv)||_2 · ||g_φ(c*)||_2) ]`
* **Structural Preservation Loss (L_str):** Measures structural similarity degradation to retain edge contours, facial landmarks, and anatomical fidelity:
  * `L_str = 1 - SSIM(x, x_adv)`

### 2. Projected Gradient Descent (PGD) with EOT

To guarantee stability under geometric transformations, the perturbation is updated iteratively across **T = 40** steps over an Expectation Over Transformation distribution **T**:

```text
x_(t+1) = Π_(x+S) ( x_t + α_step · sign( E_(t~T) [ ∇_(x_t) L_total(t(x_t), c*) ] ) )
```

where **&Pi;_(x+S)(&middot;)** denotes projection back into the &epsilon;-ball around **x**, clipped to the valid normalized image range **[-1.0, 1.0]**. The default parameters are **&epsilon; = 8/255** and step size **&alpha;_step = 1/255**.

---

## Repository Architecture

```text
DiffShield/
├── src/
│   ├── __init__.py          # Module exports
│   ├── data.py              # Dataloaders, normalization, and tensor batching
│   ├── losses.py            # Tri-Modal Loss Module (Lvis, Lsem via CLIP, Lstr via SSIM)
│   ├── optimizer.py         # PGD immunization engine with projection clamping
│   ├── eot.py               # Expectation Over Transformation differentiable pipeline
│   ├── eval.py              # Quantitative metric engines (PSNR, SSIM, LPIPS, ArcFace)
│   └── gradcam.py           # Grad-CAM spatial activation mapping
├── test_diffshield.py       # Unit and integration test suite
├── verify_immunization.py   # Baseline verification on isolated test samples
├── main.py                  # End-to-end execution script
├── requirements.txt         # Hardware and library dependencies
├── .gitignore               # Ignored artifacts, datasets, and checkpoints
└── README.md                # Framework documentation
```

---

## Installation & Setup

### Prerequisites
* Python 3.10+
* NVIDIA GPU with CUDA 11.8+ or 12.0+ (Tested on NVIDIA T4, RTX 3090, A100)
* Recommended: Minimum 12 GB VRAM for Stable Diffusion v1.5 img2img evaluation

### Environment Installation

```bash
# Clone repository
git clone https://github.com/Project-DiffShield/DiffShield.git
cd DiffShield

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies
The primary requirements include:
* `torch`, `torchvision`, `torchaudio`
* `diffusers`, `transformers`, `accelerate`
* `kornia`, `lpips`, `optuna`, `scikit-learn`, `kneed`

---

## Dataset Preparation & Mathematical Stratification

DiffShield uses high-resolution facial datasets (CelebA-HQ, resized to 512 &times; 512). To guarantee demographic variance without evaluation bias or data leakage, the sampling protocol follows a two-stage unsupervised stratification:

* **Macro-Centroid Detection via the Elbow Method:**  
  A 40-dimensional attribute matrix (mapping age, lighting, facial hair, accessories) is processed across candidate cluster counts K &isin; [1, 20]. The KneeLocator algorithm identifies the inflection point in the Within-Cluster Sum of Squares (WCSS), isolating K_calib archetypes.
* **Strict Data Isolation:**  
  The K_calib macro-centroids are reserved exclusively for hyperparameter tuning. They are permanently removed from the general pool.
* **Disjoint Evaluation Centroids:**  
  The remaining images are clustered via K-Means to extract K = 70 evaluation centroids. This guarantees zero leakage between hyperparameter search and final defense validation.

---

## Usage Guide

### 1. Verification of Immunization Pipeline
Run the verification script to validate gradient computation, metric trackers, and base PGD execution on a single tensor:

```bash
python verify_immunization.py
```

### 2. Hyperparameter Search via Optuna (Calibration Coreset)
To calibrate the scalar multipliers (&alpha;, &beta;, &gamma;) across the calibration coreset using Bayesian optimization:

```python
from src.losses import DiffShieldLoss
from src.optimizer import PGDOptimizer
import optuna

loss_fn = DiffShieldLoss(device='cuda')
target_embedding = loss_fn.encode_target_text(["a potted plant"])

def objective(trial):
    w_a = trial.suggest_float('alpha', 0.1, 5.0)
    w_b = trial.suggest_float('beta',  0.1, 5.0)
    w_g = trial.suggest_float('gamma', 50.0, 500.0)
    
    optimizer = PGDOptimizer(epsilon=8/255, alpha=1/255, iters=40, device='cuda')
    # Loop over calibration batch; enforce PSNR >= 38.0 and SSIM >= 0.95
    # ...
```

### 3. Batch Immunization Execution
Run `main.py` to batch-process input directories and export immunized image sets:

```bash
python main.py \
    --input_dir ./data/diverse_70_images \
    --output_dir ./outputs/stage1_protected \
    --epsilon 0.03137 \
    --steps 40 \
    --target_prompt "a potted plant" \
    --alpha 1.9445 \
    --beta 0.1629 \
    --gamma 93.7920
```

### 4. Running Unit Tests
Execute the repository test suite to verify module interfaces:

```bash
python -m unittest test_diffshield.py
```

---

## Experimental Benchmarks & Evaluation

### Stage 1: Invisibility & Image Fidelity (70 Disjoint Centroids)
Phase 1 ensures that the injected perturbation cannot be perceived by human viewers:

| Metric | Target Constraint | Achieved Mean &plusmn; Std | Violations |
| :--- | :---: | :---: | :---: |
| **PSNR (dB)** | &ge; 38.00 dB | **38.64 &plusmn; 0.42** | **0 / 70** |
| **SSIM** | &ge; 0.9500 | **0.9989 &plusmn; 0.0003** | **0 / 70** |
| **LPIPS (VGG)** | Feature Disruption | **0.3481 &plusmn; 0.0210** | Deep representation altered |
| **&ell;&infin; Bound** | &le; 8/255 (&asymp; 0.0314) | **0.0314** | Strictly clamped |

> **Perceptual Metric Analysis:** High PSNR and near-unity SSIM verify that spatial geometry, illumination, and color balance are preserved for human eyes. Conversely, the elevated LPIPS (0.3481) indicates that deep convolutional activations in VGG are disrupted, which destabilizes downstream generative models.

### Stage 2: Generative Immunization Efficacy (Stable Diffusion v1.5)
Evaluating clean versus protected images passed through `StableDiffusionImg2ImgPipeline` (guidance scale 7.5, strength 0.6):

* **Identity Degradation (ArcFace Cosine Similarity):** Substantial reduction in biometric identity matching scores between clean-edited and protected-edited images.
* **CLIP Text-Image Alignment Failure:** Disruption of attacker prompt execution in cross-attention feature maps.
* **Generation Divergence (MSE / LPIPS):** High divergence from expected deepfake generation, forcing semantic degradation or target concept hallucination.

---

## Citation & Academic Attribution

If you use DiffShield or parts of this codebase in your research, please cite:

```bibtex
@inproceedings{diffshield2026,
  title={DiffShield: A Tri-Modal Adversarial Framework for Robust Immunization Against Deepfake Generation},
  author={Kishan, Pranav and Venkatesh, K. and Narain, B. K. and Karthikeyan, P. G. and Raj V, Varun and Shanmuga Priya, S.},
  booktitle={Department of Computer Science and Engineering, Amrita Vishwa Vidyapeetham},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
