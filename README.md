# DiffShield v2

Defensive research project: **adversarial perturbations that protect face photos from diffusion-model editing.** A person posts a photo;
anyone can feed it to an editor (Stable Diffusion img2img, inpainting, ...) to make a fake. DiffShield adds an invisible, L-infinity-bounded
perturbation found by PGD so that edits made from the protected photo no longer look like the same person, while the protected photo looks
identical to the original. v2 makes the protection survive "purification" (an attacker cleaning the noise off first), tests it fairly
(random-noise control, identity metric, several editors and purifiers, confidence intervals, paired comparison with v1) and compares it with
existing tools. It protects people's own photos; it is not for generating fakes of anyone.

**Status.** Code and tests are complete on CPU; the GPU experiments run on Kaggle (see `kaggle/README.md`). **No v2 result is claimed in this
README.** Every number appears only in generated tables (`results/tables/`, `reports/PHASE_*.md`) with its sample size, 95% bootstrap CI over
images and number of seeds. Anything not yet measured shows `PENDING` or `not run`.

Threat model in one paragraph (details: `docs/HANDOFF_v2.txt`, `docs/V2_PLAN_48H.md`): epsilon = 4/255 on the standard 0-255 scale (v1's "8/255" was on
[-1, 1] tensors); white-box access of the *defender* to the SD VAE/UNet; attacker either non-adaptive (picks editor and purifier, does not know the
method; the headline) or adaptive (knows the method, trains an autoencoder purifier; reported separately); editors: SD1.5 img2img, SDXL img2img,
InstructPix2Pix, SD inpainting (keep-face mask), IP-Adapter FaceID; metric: ArcFace similarity and protection success rate (PSR). Known blind spot:
if an attacker masks the face itself, the pixels are discarded before the model sees them. We do **not** claim survival of any scenario.

## Repository layout

```text
config/config.yaml, config/parts/*.yaml   all hyper-parameters, seeds, paths (parts are deep-merged; one owner per part)
src/utils/        config loading, run logging (config + git hash + versions + seed per run), seeding, budget conversions, io
src/protect/      Half A: protectors (v1 adapter, v2 PGD with purifiers in the loop, L_den, anti-purification, ensemble, identity loss)
src/eval/         Half B: editors, purifiers (eval), faces, metrics, edit runner, masks, stats, analysis.py (CIs, PSR, gates)
src/baselines/    noise controls, PhotoGuard- and Mist-style baselines (our reimplementations), adaptive autoencoder
src/{eot,optimizer,losses,data,gradcam}.py   v1 code (frozen), src/legacy_eval.py (old evaluator)
scripts/          run_protect, evaluate, tune_optuna, ingest_results, compute_edit_lpips, make_tables, make_figures,
                  make_human_study, analyze_human_study, demo, budget_check, compare_runs, gpu_ledger, ...
kaggle/           job_runner.ipynb, jobs.yaml, build_bundle.py, README.md (runbook)
data/splits/      fixed identity lists (train / val / test / messy, v1 lists); data/v1_ref/ v1's original stored metrics
reports/          PHASE_0.md ... PHASE_8.md (templates fill from generated tables), reports/rendered/ (generated)
docs/             HANDOFF_v2.txt (brief), V2_PLAN.md, V2_PLAN_48H.md
tests/            CPU tests with fake/tiny models and synthetic CSVs (no weights downloaded)
results/          run folders, results/combined (ingested CSVs), results/tables, results/figures   (never commit images)
```

## Setup

Local CPU development (Windows, Python 3.13): `py -3.13 -m venv .venv`, then
`.venv\Scripts\python -m pip install -r requirements-dev.txt matplotlib scipy`.
GPU jobs run on Kaggle (T4 x2) from one code zip: see **`kaggle/README.md`** (runbook: upload the bundle, set `JOB_IDS`, Run All, download
`<JOB>_outputs.zip`). Never enter credentials anywhere; Hugging Face downloads are anonymous.

```bash
python -m pytest -q tests                              # everything on CPU (a few minutes)
python -m pytest -q tests/test_analysis_*.py           # only the analysis layer (synthetic data)
```

## Reproduce every table (one command per step)

Run these after the Kaggle jobs finished and you unzipped each `<JOB>_outputs.zip` into its own folder under `results/`
(or let step 1 do it with `--unzip`).

| Step | Command | Produces |
|---|---|---|
| 0. Phase 0 table | `python scripts/make_tables.py --run_id <J01 run id> --table invisibility` | `results/<run>/tables/invisibility.{md,csv}`; budget check `scripts/budget_check.py`, run comparison `scripts/compare_runs.py` |
| 1. Ingest | `python scripts/ingest_results.py --results results [--unzip]` | `results/combined/{edits,protect}.csv`, `coverage.md` (which method x purifier x editor cells exist, images, seeds), `coverage_vs_plan.csv`, `psr_thresholds.json`, `ingest_report.json`. Validates the schema, de-duplicates by row key, never deletes |
| 2. Edit LPIPS | `python scripts/compute_edit_lpips.py --combined results/combined --workers 4` | fills `edit_lpips` (LPIPS-VGG of clean edit vs protected edit, both JPEG q90) on CPU; resumable; then re-run step 1 |
| 3. Tables | `python scripts/make_tables.py --combined results/combined --out results/tables` | E0-E10, headline, per-editor, per-prompt, messy, invisibility, paired vs v1 / vs noise, claim numbers, gates (Markdown, CSV, `_long.csv`, LaTeX booktabs) and `numbers.json` |
| 4. Figures | `python scripts/make_figures.py --combined results/combined --out results/figures` | PSR vs PSNR, bars by purifier / editor / ablation, ArcFace distributions, before/after panels, failure gallery, Optuna history (image figures need `--source_dir`, `--edits_root`) |
| 5. Human study | `python scripts/make_human_study.py --source_dir ... --protected_dir ... --edits_root ... --clean_root ...` then `python scripts/analyze_human_study.py` | offline HTML forms + a separate answer key; analysis with binomial CI and a pilot flag |
| 6. Reports | `python scripts/make_tables.py --render reports/PHASE_3.md --tables_dir results/tables --out reports/rendered` | templates with tables and numbers filled in; missing data stays `PENDING` |
| 7. Demo | `python scripts/demo.py --image photo.jpg --out demo_out` (`--cpu-dry-run` checks the wiring without GPU) | protected image + before/after edit panel with ArcFace similarities |

A single table: `python scripts/make_tables.py --combined results/combined --out results/tables --table e4`. Gate cut-offs can be changed with
`--gate_param key=value` (see `GateParams` in `src/eval/analysis.py`); the E3 decision cut-offs are analyst proposals, the doubling and half-rule
values are the handoff's. To exercise the whole pipeline without results: `python scripts/make_tables.py --synthetic_demo <dir>` writes a fake
tree and tables marked **SYNTHETIC DATA** (software check only, never a result).

Statistics used everywhere: image is the unit; edit seeds and prompts are averaged within an image; mean and 95% percentile bootstrap CI over
images (10,000 resamples, seed 2026); paired differences are aligned by `image_id` with a Wilcoxon signed-rank p (uncorrected). PSR threshold =
95th percentile of the ArcFace similarity between clean-image edits and other people's photos (per editor; pooled also reported); a face-less edit
counts as protected and the no-face rate is reported separately.

## Data and privacy

CelebA-HQ images are for research use only. Protected, purified and edited images, human-study forms (they embed photos) and the answer key stay
local and are git-ignored; only CSV/JSON/txt results and file-name lists are committed. No credentials or tokens anywhere.

---

# v1 (legacy)

> **Stale numbers warning.** This section is the original v1 README, kept unchanged below for history. Its numbers do **not** match the
> reproducible v1 results: v1's own stored metrics (`data/v1_ref/stage1_summary_metrics.csv`, 70 images) are PSNR 41.911 +- 0.440 dB, SSIM 0.97496
> +- 0.00480, LPIPS(VGG) 0.23156 +- 0.06304; the table below (PSNR 38.64, SSIM 0.9989, LPIPS 0.3481) comes from older Canny-era trial notebooks
> and is unverified (docs/V2_PLAN.md F11). The "epsilon <= 8/255" is on [-1, 1] tensors, i.e. **4/255 on the standard scale**; the step "1/255" is
> 0.5/255 standard. The weights in the `main.py` example are not v1's final weights (exact v1 weights: alpha 0.2699, beta 0.9024, gamma 60.53 in
> `data/v1_ref/optimal_hyperparameters.json`). v1's Phase 2 test used strength 0.55 (not 0.6), one prompt, one seed, 15 pairs. The v1 code is
> frozen at its import paths (`src/losses.py`, `src/optimizer.py`, `src/eot.py`); run it through `scripts/run_protect.py --method v1`.

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
