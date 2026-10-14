# DiffShield v2: Implementation Plan (docs/V2_PLAN.md)

Status: plan written 2026-10-14 for branch `v2/phase-0`. Authoritative brief: `docs/HANDOFF_v2.txt` (shown as "the handoff" below; section numbers refer to it). Where this plan and the handoff disagree, the handoff wins, except for factual corrections about v1 marked **[FINDING]**. Every number below that does not come from the handoff or from the v1 code is either a placeholder (`<...>`) or marked "estimate" or "unverified".

Implementing agents should do Phase 0 (Section 3) in full, then **STOP and report**. Sections 4–5 are plans for later phases and must be refined at the start of each phase.

---

## 1. Current repo state vs target layout

### 1.1 What exists (as of commit f8e8329, branch v2/phase-0)

| Path | What it is | Notes |
|---|---|---|
| `src/eot.py` | v1 `EoTLayer` (Kornia RandomJPEG + RandomResizedCrop, `forward_paired`) | **Unpatched**: no `same_on_batch=True` |
| `src/optimizer.py` | v1 `PGDOptimizer` | **Unpatched**: no `loss_fn` injection; builds its own `DiffShieldLoss` |
| `src/losses.py` | v1 `DiffShieldLoss` (VAE L_vis, CLIP B/32 L_sem, **Canny** L_str with `+gamma`) | **Unpatched**: the buggy Canny term is still there |
| `src/data.py` | `CelebAHQDataset`, `get_dataloader(shuffle=True default)` | Normalizes to [-1,1] |
| `src/eval.py` | ControlNet-canny evaluator + `calculate_psnr` | Not used by v1 Phase 1/2 notebooks; **name collides with target `src/eval/` package** |
| `src/gradcam.py` | Grad-CAM on UNet | Not used by v2 |
| `main.py` | Legacy CLI (defaults eps=16/255, global non-windowed SSIM) | Not what produced the v1 numbers |
| `test_diffshield.py`, `verify_immunization.py` | Legacy "tests" that download SD/CLIP weights | Not CPU unit tests; pytest would collect them |
| `requirements.txt` | Unpinned (`torch>=2.0.0`, …) | Must be pinned |
| `notebooks/*.ipynb` | 16 historical trials | Leave untouched |
| `notebooks/v1_reference/phase1_v1.ipynb`, `phase2_v1.ipynb` | The v1 notebooks (copies) | Untracked; Kaggle T4, Python 3.12.13 |
| `docs/HANDOFF_v2.txt` | Brief | Untracked; has UTF-8/cp1252 mojibake (ε shown as "Îµ"). Do not edit it |
| `main.pdf`, `main (3).pdf` | Report drafts | Leave |

Local machine: Windows 11, no GPU (`nvidia-smi` not found), default `python` = 3.13.1 with **no packages installed** (no torch, numpy, yaml…). Python 3.11 is also present. `jq` is not installed (use Python to read notebooks). Remote: `origin = github.com/Project-DiffShield/DiffShield` (public). "Fork the repo" in the handoff is done as commits on `v2/*` branches in this clone. Pushing is the user's decision.

### 1.2 Findings from reading the v1 code (these change how Phase 0 is done)

- **F1 (budget) [FINDING].** `phase1_v1` builds `PGDOptimizer(epsilon=8/255, alpha=1/255, ...)` on [-1,1] tensors. So eps = 4/255 and **step = 0.5/255 on the standard scale**. The handoff's "step 1/255, same as v1" is the [-1,1] value read as standard. The v1 preset uses `step_255: 0.5`. Report this in PHASE_0.md.
- **F2 (clean inputs were JPEG) [FINDING].** v1 resized CelebA-HQ 1024→512 with PIL bicubic and saved under the original file name, `<n>.jpg`. That means PIL JPEG at the default quality 75 with 4:2:0 subsampling. The "clean" images v1 protected and scored were JPEG-decoded pixels. Faithful reproduction must do the same. v2 uses lossless PNG clean inputs (`data.clean_format: png`).
- **F3 (metric conversion) [FINDING].** v1 `compute_image_quality_metrics` converts tensors with `((x+1)*127.5).astype(np.uint8)`, which truncates. The saved PNG (`torchvision.save_image`) rounds. So the handoff's 41.9/0.975 are "v1 convention" numbers. Phase 0 reports both conventions: v1-tensor-truncated and published-PNG.
- **F4 (SSIM definition).** v1 SSIM = `skimage.metrics.structural_similarity(channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`. LPIPS = `lpips.LPIPS(net='vgg')` on [-1,1] tensors.
- **F5 (randomness).** v1 used global `SEED=42`, a single EoT draw per PGD iteration, a full-eps uniform random start, and `get_dataloader(..., shuffle=True)` for the 70 eval images. Output names `protected_face_###.png` follow shuffled order, and the mapping is only in `stage1_per_image_metrics.csv`. Bit-exact reproduction is impossible: RNG state depended on all earlier Optuna and sweep calls, and GPU kernels are nondeterministic. Reproduction is statistical.
- **F6 (selection).** Calibration set = KMeans centroids (k chosen by `kneed` elbow on k=1..20, `random_state=42`, `n_init=10`, fallback 4) on the 40 binary CelebAMask-HQ attributes. Eval set = 70 KMeans centroids on the remaining images. This is deterministic given the attribute file and sklearn version. `phase1_v1.ipynb` has **no stored outputs**, so the 70 file names and the exact Optuna weights are not in the repo. The handoff gives α=0.27, β=0.90, γ=60.5 (probably rounded).
- **F7 (L_str placement).** In v1, L_vis and L_str are both computed on the paired-augmented clean and poisoned images (`forward_paired`), not on the published x+δ. v2 moves L_str to the published image, as the handoff specifies.
- **F8 (L_vis scale).** v1 L_vis = MSE of `vae.encode(·).latent_dist.mean` **without** `scaling_factor` (0.18215). α=0.27 was tuned at that scale. Anything feeding the UNet must use scaled latents.
- **F9 (Phase 2 v1).** Prompt "a highly detailed portrait, cyberpunk style, neon lighting, digital art masterpiece", strength 0.55, guidance 7.5, seed 2026, `runwayml/stable-diffusion-v1-5` fp16, default scheduler and steps, 15 pairs. Per-pair LPIPS is in the stored output (mean 0.4601).
- **F10 (model id).** `runwayml/stable-diffusion-v1-5` still resolved on Kaggle in the stored phase2 log (snapshot `451f4fe…`). The repo was taken down in 2024 and this is likely an HF redirect (unverified). Plan: the v1 reproduction keeps the v1 id. v2 uses `stable-diffusion-v1-5/stable-diffusion-v1-5`, and the env probe compares VAE weight sha256 across the two.
- **F11 (README).** The README's v1 numbers (PSNR 38.64, SSIM 0.9989) do not match the handoff (41.9/0.975). They match older Canny-era trial notebooks (for example the 10th try outputs ~38.8 dB). Treat the README numbers as superseded and unverified. Do not quote them.
- **F12 (patch markers).** The phase1_v1 patch cell is idempotent: it skips each patch if `same_on_batch=True`, `loss_fn=None` or `_gaussian_window` is present in the file. Committing the patches with those exact strings keeps the old notebook runnable against the new code.

### 1.3 Migration decision

**Decision: restructure `src/` in place (Option B). Do not create a separate top-level `diffshield/` package.**

- The handoff layout's `diffshield/` is the repo root (`config/`, `src/`, `scripts/` …), so `src/protect`, `src/eval` and `src/baselines` go directly under the existing `src/`.
- **v1 stays reachable at its current import paths.** `src/eot.py`, `src/optimizer.py`, `src/losses.py`, `src/data.py` and `src/gradcam.py` stay where they are, receive the three patches as real commits, and are then **frozen** (bug fixes only, each in its own commit). Old notebooks, `main.py` and `verify_immunization.py` keep importing `src.losses`, `src.optimizer` and `src.eot` unchanged.
- **The only move:** `git mv src/eval.py src/legacy_eval.py` (a move, not a deletion; history is kept). The new `src/eval/__init__.py` lazily re-exports `Evaluator` and `calculate_psnr` from `src.legacy_eval` via module `__getattr__`, so `from src.eval import Evaluator, calculate_psnr` (in `main.py`, `test_diffshield.py`) still works and does not import diffusers when the package loads.
- v2 code lives in new subpackages: `src/protect/`, `src/eval/`, `src/baselines/`, `src/utils/`. A thin adapter `src/protect/v1.py` exposes v1 through the same `Protector` interface that v2 uses, so every script can run `--method v1`.
- Tags (local; the user decides whether to push): `v1-unpatched` on f8e8329, and `v1.0` on the commit after the three patches. That is the code that produced the handoff numbers.
- Why not a new `diffshield/` package: it would leave two code roots, two `losses.py`, and old imports pointing at stale code. Restructuring `src/` costs one file move.

---

## 2. Target tree after Phase 0

```
config/config.yaml
requirements.txt              # exact pins for Kaggle (from env probe)
requirements-dev.txt          # local CPU dev (py3.13), lower bounds only
pytest.ini                    # testpaths = tests
.gitignore                    # + results/**/images, cache, data/images, kaggle/dist, _to_delete/
src/
  __init__.py                 (exists)
  eot.py optimizer.py losses.py data.py gradcam.py      # v1, patched, frozen
  legacy_eval.py              # moved from src/eval.py
  utils/__init__.py config.py runlog.py seeding.py budget.py io.py tiny_models.py
  protect/__init__.py v1.py   # (losses.py purifiers.py pgd.py added in Phase 3)
  eval/__init__.py metrics.py stats.py schema.py        # (editors.py purify_attacks.py in Phase 1/2)
  baselines/__init__.py       # (noise_control.py photoguard.py mist.py in Phase 2)
scripts/
  run_protect.py budget_check.py prepare_v1_data.py make_tables.py
  merge_shards.py gpu_ledger.py check_env.py compare_runs.py
data/splits/  v1_calib.txt v1_eval70.txt  (train/val/test/messy.txt in Phase 1)
results/      .gitkeep  (+ <run_id>/ dirs; images ignored by git)
reports/      PHASE_0.md
kaggle/
  README.md build_bundle.py kaggle_job.py env_probe.ipynb phase0_reproduce.ipynb
tests/
  conftest.py test_config.py test_budget.py test_seeding.py test_runlog.py
  test_v1_patches.py test_v1_pgd_tiny.py test_metrics.py test_stats.py
  test_budget_check.py test_run_protect_smoke.py test_isolation.py
docs/HANDOFF_v2.txt docs/V2_PLAN.md
notebooks/v1_reference/*.ipynb   # commit as-is (checked: no tokens)
```

---

## 3. Phase 0 in full detail

### 3.1 Commit sequence (small, clear commits; never `--no-verify`)

0. `git tag v1-unpatched f8e8329`.
1. `docs: add v2 handoff, plan and v1 reference notebooks`. Adds `docs/HANDOFF_v2.txt`, `docs/V2_PLAN.md`, `notebooks/v1_reference/`. A grep for `hf_…`, `token=`, `password` found nothing.
2. `v1: EoT same_on_batch=True for paired transforms (was a runtime patch in phase1 notebook)`.
3. `v1: allow loss_fn injection in PGDOptimizer (was a runtime patch)`.
4. `v1: replace Canny structure term with windowed-SSIM preservation loss, subtract it (was a runtime patch)`. Then `git tag v1.0`.
5. `v1: optional model ids/revision and component injection in DiffShieldLoss (defaults unchanged)`. See 3.2.4.
6. `repo: v2 layout; move src/eval.py -> src/legacy_eval.py with lazy compat shim`.
7. `utils: config loading, run logging, seeding, budget conversions, image io`, with tests.
8. `eval: invisibility metrics (v1 and PNG conventions), bootstrap stats`, with tests.
9. `scripts: run_protect, budget_check, prepare_v1_data, make_tables, merge_shards, gpu_ledger, check_env, compare_runs`, with smoke tests.
10. `kaggle: code bundle builder, env probe and phase-0 reproduction notebooks`.
11. After the env probe: `deps: pin requirements.txt to Kaggle image <date>`.
12. After the runs: `reports: PHASE_0` (CSV/JSON results committed; images not committed).

### 3.2 The three runtime patches as real code

Apply the **exact** replacement text from `phase1_v1.ipynb` cell 0, so the old notebook's idempotency checks see the markers and skip.

**3.2.1 `src/eot.py`**
- `self.jpeg = K.RandomJPEG(jpeg_quality=jpeg_quality, p=0.5, same_on_batch=True)`
- `self.crop = K.RandomResizedCrop(size=(image_size, image_size), scale=crop_scale, p=0.5, same_on_batch=True)`
- Nothing else changes. Note in the docstring that `same_on_batch=True` also shares parameters across the batch dimension in `forward()`, which is fine for batch=1.

**3.2.2 `src/optimizer.py`**
- `def __init__(self, epsilon=8/255, alpha=2/255, iters=40, device='cpu', loss_fn=None):`
- `self.loss_fn = loss_fn if loss_fn is not None else DiffShieldLoss(device=self.device)`

**3.2.3 `src/losses.py`**
- Add `import torch.nn.functional as F` after `import kornia.filters as KF`.
- Replace `compute_structure_loss` with `_gaussian_window(self, window_size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32)` and the windowed-SSIM `compute_structure_loss` that returns `1.0 - ssim_map.mean()` (C1=0.01², C2=0.03², zero padding 5, on [0,1] images). Copy the notebook text verbatim.
- `forward`: `total_loss = (alpha * l_vis) - (beta * l_sem) - (gamma * l_str)`, plus the notebook's new docstring.
- Fix the class and module docstrings that still describe the Canny term (comment-only change in the same commit).

**3.2.4 Behaviour-preserving injection (commit 5).** It enables CPU tests without downloads and lets the v1 run pin a revision.

```python
class DiffShieldLoss(nn.Module):
    def __init__(self, device='cpu', sd_model_id="runwayml/stable-diffusion-v1-5",
                 clip_model_id="openai/clip-vit-base-patch32", revision=None,
                 vae=None, clip_vision=None, clip_text=None, clip_tokenizer=None):
```

Any component passed in is used as-is (and frozen). Otherwise it is loaded exactly as before (`subfolder="vae"`, fp32). `CLIPProcessor` is loaded only if `clip_tokenizer is None`, which keeps the current behaviour. Defaults reproduce v1 byte-for-byte.

Do **not** add features to v1 code: no logging, no fp16, no generator arguments.

### 3.3 `config/config.yaml` (full contents)

```yaml
# DiffShield v2: single source of truth for hyperparameters, seeds and paths.
# ALL perturbation budgets are on the STANDARD 0-255 scale. Tensors in code are in [-1, 1],
# so code converts with eps_model = eps_255 * 2 / 255 (src/utils/budget.py). Never hard-code 8/255.
project: diffshield
config_version: 1
seed: 2026                     # base seed; per-item seeds = derive_seed(seed, purpose, item_id) (sha256)

paths:
  results_root: results
  data_root: data
  splits_dir: data/splits
  cache_dir: results/cache
  celebahq_root: null          # dir containing CelebA-HQ-img/ and CelebAMask-HQ-attribute-anno.txt
                               # null on Kaggle => auto-discover under /kaggle/input (same logic as v1)

models:                        # ids UNVERIFIED until env probe / first load; then pin `revision` to the commit sha
  sd15:          {id: "stable-diffusion-v1-5/stable-diffusion-v1-5", revision: null}
  sd15_v1:       {id: "runwayml/stable-diffusion-v1-5", revision: null}      # id used by v1; reproduction only
  sd15_inpaint:  {id: "stable-diffusion-v1-5/stable-diffusion-inpainting", revision: null}
  sdxl:          {id: "stabilityai/stable-diffusion-xl-base-1.0", revision: null}
  sdxl_vae_fp16: {id: "madebyollin/sdxl-vae-fp16-fix", revision: null}
  ip2p:          {id: "timbrooks/instruct-pix2pix", revision: null}
  ipadapter_faceid: {id: "h94/IP-Adapter-FaceID", weight: "ip-adapter-faceid_sd15.bin", revision: null}
  clip_sem:      {id: "openai/clip-vit-base-patch32", revision: null}     # L_sem (v1)
  clip_eval:     {id: "openai/clip-vit-large-patch14", revision: null}    # CLIP-I metric; deliberately != clip_sem
  lpips_net: vgg
  arcface_eval:  {pack: buffalo_l, det_size: [640, 640]}                  # insightface; evaluation only

budget:
  scale: standard_0_255
  eps_255: 4                   # PROVISIONAL. v1 code: 8/255 on [-1,1] == 4/255 standard. Phase 0 budget check confirms.
  eps_confirmed: false         # set true (with run_id) after the Phase 0 budget check
  step_255: 0.5                # v1 actual (alpha=1/255 on [-1,1]). Handoff table said 1; see PHASE_0.md finding F1
  iters: 40
  random_start: uniform_full   # uniform_full = U(-eps,eps) (v1) | uniform_half (handoff ref code) | zero
  eps_sweep_255: [2, 4, 6, 8]  # E9

loss:
  alpha: 0.27                  # v1 Optuna (rounded in handoff); exact values in v1 optimal_hyperparameters.json if the user has it
  beta: 0.90
  gamma: 60.5
  eta: 0.27                    # L_den weight; starts = alpha; retuned in Phase 6
  lambda_id: 0.0               # L_id weight; enabled in Phase 6
  sem_target_text: "a potted plant"
  l_vis_latent: unscaled_mean  # MSE on vae.encode(x).latent_dist.mean, no scaling_factor (v1 scale; alpha tuned on it)
  l_str_on: published          # v2: 1-SSIM(x, x+delta). v1 preset overrides to augmented_pair
  ssim: {window: 11, sigma: 1.5}
  normalize_terms: false       # Phase 6
  anti_purify: {enabled: false, w_hf: 0.5, w_mono: 0.5}   # Phase 5 (AntiPure idea)

eot:
  k_draws: 4                   # 8 if memory allows
  spatial: {enabled: false, crop_scale: [0.8, 1.0], p: 0.5}   # if enabled: same crop on clean, z_clean recomputed per draw
  v1: {jpeg_quality: [50, 100], crop_scale: [0.8, 1.0], p: 0.5, same_on_batch: true}

purifiers_train: [jpeg, downup, vae_roundtrip, sd_onestep]
purifier_train_probs: {jpeg: 0.25, downup: 0.25, vae_roundtrip: 0.2, sd_onestep: 0.3}   # Phase 3 renormalizes without sd_onestep
purifier_train_params:
  jpeg:          {impl: kornia_diff, quality: [50, 95]}
  downup:        {scale: [0.5, 0.9], modes: [bilinear, bicubic], antialias: true}
  vae_roundtrip: {model: sd15, latent: mean}
  sd_onestep:    {model: sd15, t_min: 50, t_max: 300, prompt: ""}
purifiers_heldout: [jpeg_q_other, downup_other, diffpure_t0.1, diffpure_t0.2, diffpure_t0.3, gridpure]
purifier_eval:                 # Half B only; real codecs (PIL), never the differentiable approximations
  none: {}
  jpeg_q90: {impl: pil, quality: 90}
  jpeg_q75: {impl: pil, quality: 75}
  jpeg_q50: {impl: pil, quality: 50}
  jpeg_q_other: {impl: pil, quality: 65, subsampling: "4:4:4"}
  downup_0.5:   {impl: pil, scale: 0.5, resample: bicubic}
  downup_other: {impl: pil, scale: 0.375, resample: lanczos}
  vae_rt_sd15:  {model: sd15}
  diffpure_t0.05: {impl: sd_img2img_empty_prompt, strength: 0.05, steps: 50, guidance: 1.0}
  diffpure_t0.1:  {impl: sd_img2img_empty_prompt, strength: 0.1,  steps: 50, guidance: 1.0}
  diffpure_t0.2:  {impl: sd_img2img_empty_prompt, strength: 0.2,  steps: 50, guidance: 1.0}
  diffpure_t0.3:  {impl: sd_img2img_empty_prompt, strength: 0.3,  steps: 50, guidance: 1.0}
  gridpure: {impl: official, feasible: unverified}
  restore_swin2sr: {impl: downup_then_sr, scale: 0.5, model: "caidas/swin2SR-classical-sr-x2-64"}

editors: [sd15_img2img, sdxl_img2img, ip2p, sd_inpaint, ipadapter_faceid]   # identity editor: IP-Adapter FaceID (see Section 6)
edit:
  strengths: [0.3, 0.55, 0.75]          # sd15_img2img sweeps these; others use default_strength
  default_strength: 0.55
  seeds: [1, 2, 3]
  guidance_scale: 7.5
  num_inference_steps: 50
  prompts:
    img2img: ["a person in a suit", "a person at a beach", "an oil painting of a person"]
    ip2p:    ["put the person in a suit", "put the person on a beach", "turn it into an oil painting"]
  ip2p_image_guidance: 1.5
  sdxl_resolution: 1024
  v1_phase2: {prompt: "a highly detailed portrait, cyberpunk style, neon lighting, digital art masterpiece",
              strength: 0.55, guidance: 7.5, seed: 2026}

data:
  image_size: 512
  clean_format: png             # v2: lossless. v1 preset: v1_jpeg (PIL default q75)
  n_train: 20
  n_val: 20
  n_test: 100
  n_messy: 20
  split_seed: 2026
  v1_selection: {kmeans_random_state: 42, n_init: 10, k_range: [1, 20], k_eval: 70, fallback_k_calib: 4}

metrics:
  identity: arcface
  psr_threshold: "from clean-edit distribution"   # 95th pct of ArcFace sim(source_i, clean_edit_j), i != j
  psr_impostor_percentile: 95
  psr_threshold_scope: per_editor
  no_face_counts_as_success: true                  # and report no-face rate separately
  invisibility_min: {psnr: 38.0, ssim: 0.95}

stats: {n_boot: 10000, ci: 0.95, unit: image, bootstrap_seed: 2026}

runtime:
  device: auto                  # cuda if available else cpu
  unet_dtype: fp16
  vae_dtype_protect: fp32
  grad_checkpointing: true
  loss_scale: 1024.0            # multiply loss before autograd.grad (fp16 underflow guard; sign() is scale-invariant)
  checkpoint_every: 5           # images
  cudnn_benchmark: false

methods:                        # presets deep-merged over the root by load_config(method=...)
  v1:
    budget: {eps_255: 4, step_255: 0.5, iters: 40, random_start: uniform_full}
    eot: {k_draws: 1}
    loss: {alpha: 0.27, beta: 0.90, gamma: 60.5, eta: 0.0, lambda_id: 0.0, l_str_on: augmented_pair}
    purifiers_train: []
    model_key: sd15_v1
    data: {clean_format: v1_jpeg}
  # v2_cheap (Phase 3), v2_onestep (Phase 4), v2_full (Phase 6) are added later, never edited after their gate.
```

### 3.4 `src/utils/` API

**`src/utils/budget.py`**
```python
def to_model_scale(v_255: float) -> float              # v * 2 / 255
def to_standard_scale(v_model: float) -> float         # v * 255 / 2
def tensor_to_u8_round(x: Tensor) -> np.ndarray        # ((x+1)*127.5).round().clamp(0,255) HWC uint8 (published)
def tensor_to_u8_trunc_v1(x: Tensor) -> np.ndarray     # exact v1: ((x+1)*127.5).astype(np.uint8) via numpy
```

**`src/utils/seeding.py`**
```python
def derive_seed(base: int, *parts: str | int) -> int    # sha256("|".join(...)) -> int % 2**31; NOT Python hash()
def seed_everything(seed: int) -> None                   # random, numpy, torch, torch.cuda
def torch_generator(seed: int, device) -> torch.Generator
```
Per-image seeding (`derive_seed(cfg.seed, "protect", image_id)`) makes results independent of image order, which makes resume and sharding deterministic.

**`src/utils/config.py`**
```python
def load_config(path: str | Path, method: str | None = None, overrides: Sequence[str] = ()) -> dict
def deep_merge(base: dict, over: dict) -> dict
def apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict   # "a.b.c=val", val parsed with yaml.safe_load
def cfg_get(cfg: dict, dotted: str, default=...) -> Any            # KeyError if missing and no default
def validate_config(cfg: dict) -> None   # required keys; budget.scale == standard_0_255; 0<eps_255<=16; probs sum to 1
def config_hash(cfg: dict) -> str        # sha256 of canonical json, first 12 chars
```
PyYAML only (no OmegaConf). The resolved config records `"_method"` and `"_overrides"`.

**`src/utils/runlog.py`**
```python
@dataclass
class RunContext:
    run_id: str; dir: Path; images_dir: Path; cfg: dict; seed: int; shard: tuple[int, int] | None
    def log(self, msg: str) -> None                     # stdout + dir/logs/run[.shardK].log
    def append_row(self, csv_name: str, row: dict) -> None   # header on first write, flush + os.fsync
    def done_ids(self, csv_name: str, key: str = "image_id") -> set[str]
    def write_progress(self, **kw) -> None              # dir/progress[.shardK].json (atomic write via tmp + os.replace)
    def gpu_timer(self) -> GpuTimer
def make_run_id(phase: str, method: str, tag: str, git_hash: str) -> str   # "p0_v1_eval70_<hash7>_<YYYYmmdd-HHMMSS>"
def get_git_info(repo_root: Path) -> dict   # {"hash","dirty","branch","source": "git"|"bundle"|"unknown"};
                                            # falls back to BUNDLE_INFO.json (Kaggle has no .git)
def collect_versions(pkgs=("torch","torchvision","diffusers","transformers","accelerate","kornia","numpy",
                           "pillow","scikit-image","scikit-learn","lpips","insightface","onnxruntime-gpu",
                           "optuna","kneed","pyyaml")) -> dict   # importlib.metadata; "not-installed" if missing
def collect_env() -> dict   # python, platform, torch.version.cuda, cudnn, GPU names, driver (nvidia-smi query), KAGGLE_* env
def start_run(cfg, run_id, output_root, script, argv, seed, shard=None) -> RunContext
    # writes: config.json (resolved), config.yaml (copy), git_hash.txt, versions.json, env.json,
    #         seed.txt, command.txt (argv); refuses to overwrite a config.json with a different config_hash
def finish_run(ctx, summary: dict) -> None   # summary.json + gpu_time.json
class GpuTimer:  # context manager; torch.cuda.synchronize() at enter/exit when CUDA; .seconds; also peak memory
```

**`src/utils/io.py`**
```python
def list_images(input_dir, list_file=None, exts=(".png",".jpg",".jpeg")) -> list[Path]   # sorted by stem
def shard_items(items: list, shard: tuple[int, int] | None) -> list                       # items[k::n]
def load_tensor_v1(path, size=512) -> Tensor   # exactly src.data transform: Resize((512,512)) -> ToTensor -> Normalize(.5,.5); [1,3,H,W]
def save_png_v1(x: Tensor, path) -> None       # torchvision.utils.save_image((x+1)/2) (rounding), atomic tmp + rename
def load_u8(path) -> np.ndarray                # RGB uint8 HWC
```

**`src/utils/tiny_models.py`** (offline, random-init, for tests and `--backend tiny` only)
```python
def tiny_vae() -> AutoencoderKL   # block_out_channels=(8,16), layers_per_block=1, latent_channels=4, norm_num_groups=8
def tiny_clip_vision(projection_dim=16) -> CLIPVisionModelWithProjection   # image_size=224, patch_size=32, hidden 32, 2 layers
def tiny_target_embedding(dim=16, seed=0) -> Tensor
def tiny_v1_loss(device="cpu") -> DiffShieldLoss   # via the injection kwargs, no tokenizer download
```

### 3.5 `src/protect/v1.py` (adapter, no algorithm changes)

```python
class Protector(Protocol):
    name: str
    def protect(self, x: Tensor, seed: int) -> tuple[Tensor, dict]: ...   # x [1,3,H,W] in [-1,1]; returns (x_adv, info)

class V1Protector:
    def __init__(self, cfg: dict, device: str, backend: str = "real"):
        # real: DiffShieldLoss(device, sd_model_id=cfg.models[cfg.model_key].id, revision=...)
        #       target = loss_fn.encode_target_text([cfg.loss.sem_target_text])
        # tiny: tiny_v1_loss(); tiny_target_embedding()
        # PGDOptimizer(epsilon=to_model_scale(eps_255), alpha=to_model_scale(step_255), iters, device, loss_fn=loss_fn)
        # strict check for method v1: abs(epsilon - 8/255) < 1e-12 and abs(alpha - 1/255) < 1e-12
    def protect(self, x, seed):
        seed_everything(seed)                 # v1 uses the global RNG (delta.uniform_, kornia params)
        x_adv = self.opt.optimize(x, self.target, w_alpha=..., w_beta=..., w_gamma=...)
        return x_adv, {"linf_model": float((x_adv - x).abs().max())}
def build_protector(cfg, method: str, device: str, backend: str) -> Protector   # "v1" only in Phase 0
```

### 3.6 `src/eval/` (Phase 0 subset)

**`src/eval/__init__.py`**: lazy `__getattr__` for `Evaluator` and `calculate_psnr` from `src.legacy_eval`. Nothing else imports at package load.

**`src/eval/metrics.py`**
```python
def psnr_u8(a: np.ndarray, b: np.ndarray) -> float           # float64 MSE; inf if 0 (as v1)
def ssim_u8(a, b) -> float                                    # skimage settings exactly as F4
def max_abs_diff_u8(a, b) -> int
def diff_histogram_u8(a, b, max_level=16) -> dict[int, int]
class LPIPSScorer:                                            # lazy lpips.LPIPS(net=cfg.models.lpips_net); __call__(t1, t2)
def v1_quality_metrics(clean_t, prot_t, lpips: LPIPSScorer | None) -> dict
    # exact replica of phase1_v1 compute_image_quality_metrics -> {MSE, PSNR, SSIM, LPIPS, Linf}
def png_quality_metrics(clean_path, prot_path, lpips) -> dict
    # published convention: decode both files as uint8 (the clean file is what the protector saw)
    # -> {psnr, ssim, lpips, maxabs_u8, frac_at_bound}
```
v2 Half-B metrics (ArcFace, CLIP-I, edit LPIPS) are added here in Phase 1. This module must never import `src.protect`.

**`src/eval/stats.py`**
```python
def bootstrap_ci(x, n_boot=10000, ci=0.95, seed=2026) -> tuple[float, float, float]   # (mean, lo, hi), percentile method
def summarize(x, **kw) -> dict    # n, mean, std (ddof=0, as v1 np.std), min, max, ci_lo, ci_hi
def paired_bootstrap_diff(a, b, **kw) -> dict   # mean diff, CI, share of images improved; a, b aligned by image_id
def per_image_aggregate(df, value_col, by=("image_id",)) -> pd.Series   # average seeds/prompts before bootstrapping over images
```

**`src/eval/schema.py`**: the column constants (Section 4.1). Phase 0 uses `PROTECT_COLUMNS`.

### 3.7 Scripts

Every result-writing script: `--config`, `--set k=v ...`, `--run_id`, `--output_root`. Each calls `start_run` and `finish_run`. Each does `sys.path.insert(0, repo_root)` (no packaging needed).

**`scripts/run_protect.py`** (folder in, protected folder out)
```
python scripts/run_protect.py --config config/config.yaml --method v1
    --input_dir <dir> [--list data/splits/v1_eval70.txt] [--limit 5]
    --run_id <id> [--output_root results] [--shard K/N] [--resume] [--resume_from <prev_run_dir>]
    [--checkpoint_every 5] [--backend real|tiny] [--device auto|cuda|cpu] [--no_lpips]
    [--set budget.iters=2 ...]
```
- `main(argv=None) -> int`, `parse_args(argv)`, `process_one(item, protector, ctx, lpips) -> dict`.
- Loop over `shard_items(list_images(...))`. Skip any image whose id is in `ctx.done_ids("metrics.shardK.csv")` **and** whose PNG exists.
- For each image:
  - `seed = derive_seed(cfg.seed, "protect", image_id)`
  - `with ctx.gpu_timer() as t: x_adv, info = protector.protect(x, seed)`
  - `save_png_v1` writes `images/<stem>.png` (atomic)
  - compute `v1_quality_metrics` and `png_quality_metrics`
  - `append_row`
- Every `checkpoint_every` images: `write_progress(done, total, gpu_seconds_total, last_image, eta_s)`.
- `--resume_from` copies an earlier partial run (for example `/kaggle/input/<prev-output>/results/<run_id>`) into the new run dir, then resumes. The config hash must match.
- Row = `PROTECT_COLUMNS`:
  - keys: `run_id, method, method_cfg_hash, image_id, split, seed, editor="none", prompt_id="none", strength="", purifier="none"`
  - provenance: `src_file, out_file, eps_255, step_255, iters`
  - v1 convention: `v1_psnr, v1_ssim, v1_lpips, v1_linf_model`
  - published convention: `png_psnr, png_ssim, png_lpips, png_maxabs_u8, png_frac_at_bound`
  - cost: `gpu_seconds, peak_mem_mb, created_utc`
- At the end: `summary.json` (`summarize` per metric, both conventions), `gpu_time.json` (`{total_gpu_seconds, n_images, per_image_mean, gpu_name}`).

**`scripts/budget_check.py`** (CPU)
```
python scripts/budget_check.py --clean_dir <dir> --protected_dir <dir> [--pairs_csv v1_stage1_per_image_metrics.csv]
       [--run_id ...]  -> results/<run_id>/budget_check.json + .md
```
- Pairs by stem by default.
- `--pairs_csv` handles the user's original v1 outputs (`Original_Filename` to `Protected_Filename`, F5).
- Reports per-image and global max |diff|, the histogram of |diff| (0…16), the share of pixels at the max, and the inferred `eps_255`.
- Conclusion line: "max gray-level change = N ⇒ effective ε = N/255 (standard)". Warns if the clean files are JPEG (expected for v1).
- Handoff reference: `np.abs(c - p).max()` on `.convert('RGB')` int arrays. The tensor variant uses `((x+1)*127.5).round()`.

**`scripts/prepare_v1_data.py`** (CPU; runs locally or on Kaggle)
```
python scripts/prepare_v1_data.py --celebahq_root <dir> --out_dir data/v1 [--from_lists] [--write_splits]
```
- Reproduces v1 selection exactly (F6): attribute parse `1 if int(x)==1 else 0`, KMeans k=1..20, KneeLocator(convex, decreasing), calib centroids via `pairwise_distances_argmin_min`, eval KMeans(70) on the remaining rows.
- Writes `data/splits/v1_calib.txt` and `v1_eval70.txt` (file names), plus `selection_meta.json` (k_calib, sklearn and kneed versions, WCSS).
- Writes 512 px images with `Image.open().convert('RGB').resize((512,512), Image.Resampling.BICUBIC).save(out/<name>.jpg)`, keeping the original `.jpg` name, which gives PIL default q75 (F2).
- `--from_lists` skips KMeans and uses committed lists. If the user supplies v1's `evaluation_70_manifest.csv`, compare and report overlap. If it differs, use the user's manifest and note it.

**`scripts/make_tables.py`**: `--run_id X --table invisibility`. Writes `results/X/tables/invisibility.{csv,md}`: metric | handoff v1 value | reproduced mean ± std | 95% CI | n | violations (<38 dB / <0.95), for both conventions. Later phases add tables, always generated from CSVs.

**`scripts/merge_shards.py`**: `--run_dir`. Concatenates `metrics.shard*.csv` into `metrics.csv`, checks for no duplicate and no missing ids against the list, and recomputes `summary.json`.

**`scripts/gpu_ledger.py`**: scans `results/*/gpu_time.json` and `kaggle_session.json` and writes `results/gpu_ledger.csv` (run_id, iso_week, session_seconds, gpu_seconds, gpu_name, n_gpus) plus a weekly total.

**`scripts/check_env.py`**: `--requirements requirements.txt [--allow_mismatch]`. Compares installed versions with pins. Exits non-zero on a mismatch of torch, torchvision, diffusers, transformers or kornia. Prints `torch.cuda.get_arch_list()` and the GPU name.

**`scripts/compare_runs.py`**: `--a runA --b runB`. Per-image paired difference of v1_psnr, v1_ssim and v1_lpips, with CI. This is the "another person reproduces" check.

### 3.8 Requirements

**Procedure (no invented pins):**
1. Run `kaggle/env_probe.ipynb` in a **CPU** Kaggle session (no GPU quota) with the same Kaggle image as the GPU jobs. It writes `kaggle_env_freeze.txt` (`pip freeze`), `python --version`, `torch.__version__`, `torch.version.cuda`, and the sha256 metadata of `vae/diffusion_pytorch_model.safetensors` for both SD1.5 ids via `huggingface_hub.HfApi().model_info(..., files_metadata=True)`. No token is needed for public repos.
2. The user downloads the freeze. The agent commits it as `kaggle/env/kaggle_env_freeze_<date>.txt`.
3. Write `requirements.txt` with `==` pins **taken from that freeze** for everything Kaggle preinstalls, plus `==` pins for what we add.
4. Ask the user to set the Kaggle notebook option "Environment: Pin to original environment" after the first successful run, so the image does not drift.

**`requirements.txt` structure** (versions filled in from the probe; floors are the known requirements):
```
# provided by the Kaggle image: verified by scripts/check_env.py, NOT reinstalled by the notebooks
torch==<probe>            # CUDA build; never pip-install torch on Kaggle
torchvision==<probe>
numpy==<probe>
pillow==<probe>
scikit-learn==<probe>
scikit-image==<probe>
pandas==<probe>
pyyaml==<probe>
# installed by the job notebook (exact pins; floors are hard requirements)
diffusers==<probe or chosen>      # >=0.28 (IP-Adapter FaceID support), checked in Phase 1
transformers==<probe>
accelerate==<probe>
huggingface_hub==<probe>
kornia==<chosen>                  # >=0.7.0 (RandomJPEG, jpeg_codec_differentiable)
lpips==0.1.4
kneed==<chosen>                   # 0.8.x
optuna==<chosen>                  # Phase 6
# Phase 1+ (added when first used)
# insightface==0.7.3 ; onnxruntime-gpu==<matches Kaggle CUDA/cuDNN>
```
The Kaggle notebook installs `requirements.txt` minus the "provided by the image" block. `check_env.py` verifies that block.

**`requirements-dev.txt`** (local Windows, Python 3.13, CPU; lower bounds only):
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.6
torchvision>=0.21
kornia>=0.7.0
diffusers>=0.28
transformers>=4.40
numpy>=1.26
pillow>=10
pyyaml>=6
scikit-image>=0.22
scikit-learn>=1.4
kneed>=0.8
pandas>=2.1
lpips==0.1.4
pytest>=8
```
Local setup: `py -3.13 -m venv .venv && .venv\Scripts\python -m pip install -r requirements-dev.txt`. If a package has no 3.13 wheel, use `py -3.11` (installed) and note it.

### 3.9 Tests (CPU, no SD/CLIP downloads, runtime well under 2 minutes)

`pytest.ini`: `testpaths = tests`, `markers = slow: needs downloaded weights; gpu: needs CUDA`. The legacy root `test_diffshield.py` is not collected; it is documented as legacy/slow in the README.

| Test file | What it checks |
|---|---|
| `test_config.py` | Real `config/config.yaml` loads and validates. `--set` overrides parse types. The `v1` preset merges (k_draws=1, l_str_on=augmented_pair, eps 4, step 0.5). `config_hash` is stable. Bad scale or eps raises |
| `test_budget.py` | `to_model_scale(4) == 8/255`, `to_model_scale(0.5) == 1/255`. The round and trunc converters match hand-computed arrays, including the v1 truncation off-by-one on float values like 126.99999 |
| `test_seeding.py` | `derive_seed` is deterministic across processes (subprocess check) and differs by purpose and image id |
| `test_runlog.py` | `start_run` in a temp dir writes config.json, git_hash.txt, versions.json, env.json, seed.txt, command.txt. Without git: reads `BUNDLE_INFO.json`, else "unknown". Refuses a conflicting config. `append_row` + `done_ids` round-trip |
| `test_v1_patches.py` | (a) Marker strings exist in the src files (F12). (b) With `eot.crop.p = eot.jpeg.p = 1.0`, `forward_paired(x, x.clone())` returns identical halves over 10 seeds (this fails without `same_on_batch`). (c) `compute_structure_loss(x, x) ≈ 0` and grows with noise. On the interior (border 5 px trimmed) it is within 1e-2 of `1 - skimage SSIM`. (d) Sign: with monkeypatched component losses returning constants, `total = a*v - b*s - g*t`. (e) `PGDOptimizer(loss_fn=fake)` never calls the `DiffShieldLoss` constructor (monkeypatch it to raise) |
| `test_v1_pgd_tiny.py` | `tiny_v1_loss` + `PGDOptimizer` on a 1×3×64×64 image, iters=3: output in [-1,1], Linf ≤ 8/255 + 1e-6, the same seed gives the same output, a different seed differs, and grads are not all zero |
| `test_metrics.py` | `psnr_u8` with constant diff 4 = 20·log10(255/4) ≈ 36.12 dB. Identical images: SSIM 1.0, PSNR inf. `v1_quality_metrics` equals an oracle copy of the notebook function pasted into the test (LPIPS replaced by a stub) |
| `test_stats.py` | CI contains the mean. Constant input gives a zero-width CI. Deterministic for a seed. Paired diff of identical arrays is 0 |
| `test_budget_check.py` | Synthetic clean JPEG and protected PNG with known |diff| ≤ 4 gives a reported max of 4 and eps_255 = 4. `--pairs_csv` mapping works |
| `test_run_protect_smoke.py` | `run_protect.main([... --backend tiny --device cpu --set budget.iters=2 --no_lpips ...])` on 3 synthetic 64 px images: PNGs written, 3 CSV rows, config.json present. A rerun with `--resume` adds no rows. `--shard 0/2` + `1/2` + `merge_shards` gives 3 unique rows equal to an unsharded run (deterministic per-image seeds) |
| `test_isolation.py` | AST scan: no module under `src/eval/` imports `src.protect` and vice versa (Half A and Half B never share components). `src/baselines` may import neither eval nor protect internals except `src.utils` |

Optional `@pytest.mark.slow`: real SD VAE + CLIP B/32 on CPU at 512 px, 1 image, 2 iterations. Run only if the weights are already cached.

### 3.10 Kaggle packaging and notebooks

**`kaggle/build_bundle.py`**
```
python kaggle/build_bundle.py [--allow_dirty] -> kaggle/dist/diffshield-code-<hash7>.zip
```
- Contents: `src/`, `scripts/`, `config/`, `data/splits/`, `kaggle/kaggle_job.py`, `requirements*.txt`, `pytest.ini`, `tests/`.
- Adds `BUNDLE_INFO.json` with `{git_hash, dirty, branch, created_utc, sha256 per file}`.
- Refuses a dirty tree unless `--allow_dirty` (then `dirty: true` is recorded and shows in every run's `git_hash.txt`).
- The user uploads the zip as a new version of a private Kaggle Dataset `diffshield-code` through the web UI. The agent never handles Kaggle credentials.
- Alternative once the user has pushed the branch: notebook `CODE_SOURCE="git"` with `GIT_REF=<sha>` does `git clone` + `git checkout <sha>`.

**`kaggle/kaggle_job.py`** (helpers imported by notebooks)
```python
def session_start(path="/kaggle/working/kaggle_session.json") -> None   # first cell; records wall-clock start
def session_end(path=...) -> None                                       # last cell; session_seconds
def install_code(source: str, git_ref: str | None, dest="/kaggle/working/DiffShield") -> Path
    # dataset: unzip/copy from /kaggle/input/diffshield-code
def pip_install_pins(req: Path) -> None          # skips the "provided by image" block
def find_celebahq(root="/kaggle/input") -> tuple[Path, Path]   # dir with CelebA-HQ-img + attribute file (v1 logic)
def run_sharded(cmd: list[str], n_gpus: int, log_dir: Path) -> list[int]
    # subprocess.Popen per GPU with CUDA_VISIBLE_DEVICES=k and "--shard k/n"; waits and tails logs
def zip_outputs(run_dirs: list[Path], out_zip: Path) -> None
```

**`kaggle/env_probe.ipynb`** (CPU session, internet on). Cells:
1. `pip freeze > /kaggle/working/kaggle_env_freeze.txt`; Python, torch and CUDA versions; `torch.cuda.get_arch_list()` (informative only on CPU).
2. HF metadata (no downloads): sha256 of the SD1.5 VAE file for `runwayml/...` and `stable-diffusion-v1-5/...`; record whether each id resolves.
3. Check that `pip download --no-deps lpips==0.1.4 kneed kornia` resolves.

**`kaggle/phase0_reproduce.ipynb`** (Accelerator "GPU T4 x2", internet on). Inputs: the same CelebAMask-HQ dataset v1 used (Kaggle `sourceId 3973359`; the user attaches it) and `diffshield-code`.
1. `session_start()`. Parameters cell: `CODE_SOURCE`, `GIT_REF`, `N_IMAGES` (5 = smoke, 70 = full), `ITERS` (40), `RUN_TAG`, `N_GPUS=2`, `RESUME_FROM=None`.
2. `install_code`, `pip_install_pins`, `python scripts/check_env.py`.
3. `python -m pytest -q tests` (about 1 minute; fails fast if the bundle is broken).
4. `prepare_v1_data.py --celebahq_root <found> --out_dir /kaggle/working/data/v1 --from_lists` (the committed lists from local CPU selection), or without `--from_lists` to recompute and compare.
5. `run_sharded(["python","scripts/run_protect.py","--method","v1","--input_dir",".../data/v1","--list","data/splits/v1_eval70.txt","--limit",N_IMAGES,"--run_id",RUN_ID,"--output_root","/kaggle/working/results","--checkpoint_every","5", ...], N_GPUS)`, then `merge_shards.py`.
6. `budget_check.py --clean_dir .../data/v1 --protected_dir results/<RUN_ID>/images --run_id <RUN_ID>`, then `make_tables.py --run_id <RUN_ID> --table invisibility`.
7. `session_end()`, then `zip_outputs` to `/kaggle/working/phase0_<RUN_ID>.zip`.
   - The zip includes protected images for the team's private use. These are CelebA-HQ derivatives: research use only, never committed or published.

Runbook (`kaggle/README.md`):
- User: upload the bundle, attach the datasets, choose T4 x2 (not P100: newer torch CUDA builds may not support sm_60, unverified; check `get_arch_list` output), enable internet, Run All or Save Version.
- User: download the zip and unzip it into `results/`.
- Agent: run `gpu_ledger.py`, `make_tables.py`, write the report.
- If a session dies: rerun with `RESUME_FROM` pointing at the earlier output, added as notebook input.
- Whether outputs of timed-out commit sessions are kept is unverified, so jobs are sized to finish well inside the session limit.

### 3.11 Execution order and gate

1. Commits 0–10 locally. `pytest` green on the local CPU venv.
2. **CPU-only budget check on the original v1 outputs if the user has them** (`stage1_protected_images.zip`, `stage1_original_images.zip`, `diffshield_phase1_complete_results.zip`). Expect max diff 4 if F1 is right. This costs no GPU time.
3. Local: `prepare_v1_data.py` selection on the attribute file if the user can provide it (small text file); commit `v1_calib.txt` and `v1_eval70.txt`. Otherwise run it in the Kaggle job.
4. Kaggle env probe, then pin requirements (commit 11), then build the bundle.
5. Kaggle smoke: `N_IMAGES=5`, `ITERS=40`, 2 GPUs. Check per-image `gpu_seconds`, peak memory, metric sanity, max diff = 4. Fix issues at this scale.
6. Kaggle full: `N_IMAGES=70`. GPU cost = measured smoke per-image seconds × 70 / 2 GPUs + setup overhead (record it; do not guess).
7. A second team member runs the same notebook version with the same bundle. `compare_runs.py` between the two runs.
8. Write `reports/PHASE_0.md`, set `budget.eps_confirmed: true` with the run id, commit, **STOP and report**.

Proposed reproduction tolerance (the gate values are suggestions; ask the user to accept it):
- v1-convention mean PSNR within ±0.5 dB of 41.9
- mean SSIM within ±0.005 of 0.975
- mean LPIPS within ±0.03 of 0.232
- Linf exactly 8/255 on [-1,1]
- PNG max |diff| = 4

If any criterion is missed, report the diagnostics (JPEG vs PNG clean, truncation vs rounding, seed). Do **not** tune anything to make the numbers match.

### 3.12 `reports/PHASE_0.md` template

```markdown
# Phase 0 report: reproducible base
Date: <YYYY-MM-DD> · Code: <git hash / tag v1.0 + phase-0 commits> · Runs: <run_id_A>, <run_id_B (second person)>

## Result first (plain language)
<2–4 sentences: did v1's numbers reproduce? What is the true budget? e.g. "The largest pixel change is N gray levels, so the effective budget is N/255.">

## Budget finding
| Source | n images | max abs gray-level change | share of pixels at max | inferred eps (standard) |
|---|---|---|---|---|
| Original v1 outputs (user's zip) | <n or "not available"> | <> | <> | <> |
| Reproduced v1 run | <> | <> | <> | <> |
Conclusion: budget.eps_255 = <N> (eps_confirmed: true). Step size finding: v1 step = 1/255 on [-1,1] = 0.5/255 standard (handoff table listed 1/255).

## Reproduction of v1 Phase 1 (invisibility), <n> images, 1 protection seed per image
| Metric | v1 reported (handoff) | Reproduced, v1 convention: mean ± std [95% CI] | Reproduced, published PNG: mean ± std [95% CI] | Violations |
|---|---|---|---|---|
| PSNR (dB) | 41.9 ± 0.4 | <> | <> | <k>/<n> below 38 |
| SSIM (windowed) | 0.975 ± 0.005 | <> | <> | <k>/<n> below 0.95 |
| LPIPS VGG (clean vs protected) | 0.232 ± 0.063 | <> | <> | n/a |
| Linf ([-1,1] scale) | 8/255 | <> | n/a | <> |
Second-person run agreement (compare_runs): mean PSNR diff <> [CI], SSIM diff <> [CI].
Gate (proposed tolerance ±0.5 dB / ±0.005 / ±0.03): PASS / FAIL.

## Differences from the original v1 run (and their expected effect)
- Clean inputs: JPEG q75 at 512 px, replicated (F2). Seeds: per-image sha256-derived seeds instead of the global seed 42 (F5). Image order: sorted, not shuffled.
- Weights: α, β, γ = <exact or rounded handoff values>. Model id/revision: <>. Library versions: see versions.json.
- Anything else found: <>.

## What was done
<bullets: the three patches as commits (hashes), config, logging, scripts, tests (count, all passing on CPU), Kaggle bundle>

## GPU time used
| Job | GPUs | session time | GPU-seconds | per image |
|---|---|---|---|---|
| env probe (CPU) | 0 | <> | 0 | – |
| smoke (5 images) | 2×T4 | <> | <> | <> |
| full (70 images) | 2×T4 | <> | <> | <> |

## Decisions taken
<e.g. v2 step_255 default, metric conventions, layout choice>

## Open questions for the user (max 2)
1. <>
2. <>

## Artifacts
results/<run_id>/{config.json, git_hash.txt, versions.json, env.json, seed.txt, metrics.csv, summary.json, gpu_time.json, budget_check.json, tables/}. Images: kept locally/Kaggle only (not in git).

**STOP: waiting for the user's go-ahead before Phase 1.**
```

---

## 4. Shared infrastructure for Phases 1–8

### 4.1 Results CSV schema (`results/<run_id>/metrics.csv`, one row per image × seed × editor × purifier × method)

The editor key is (`editor`, `prompt_id`, `strength`). For protect-only rows, `editor=none`.

| Group | Columns |
|---|---|
| Keys | `run_id, method, method_cfg_hash, image_id, split, seed, editor, prompt_id, strength, purifier` |
| Provenance | `protect_run_id, eps_255, purifier_params (json), editor_params (json), source_path, protected_path, purified_path, edit_path, clean_edit_path, code_git_hash, created_utc` |
| Identity | `face_detected (0/1), n_faces, det_score, arcface_sim (NaN if no face), fr2_sim (second face-recognition model, see 6.R16)` |
| Edit | `edit_lpips (vs clean edit with the same editor/prompt/strength/seed, no purifier), clip_i (source vs edit, clip_eval model)` |
| Invisibility | `inv_psnr, inv_ssim, inv_lpips` (protected PNG vs clean PNG; repeated per row, from the protect run) |
| Cost | `gpu_seconds_edit, gpu_seconds_purify` |

Not stored (computed in `stats.py`/`make_tables.py` from frozen thresholds): `psr_hit = (face_detected == 0) or (arcface_sim < thr[editor])`. Thresholds live in `results/thresholds/psr_thresholds.json`, computed once from E0 as the 95th percentile of `sim(source_i, clean_edit_j)` for i≠j (different identities), per editor. Also report a pooled threshold and the no-face rate.

Statistics: average each image over seeds and prompts first, then bootstrap over images (n_boot 10000). Paired vs v1 on identical `(image_id, seed, editor, prompt_id, strength, purifier)`.

### 4.2 Two-stage evaluation and caching (saves GPU quota)

- **Stage `edit` (GPU):** writes PNG edits plus `edits_index.csv`.
- **Stage `score` (CPU):** ArcFace, LPIPS and CLIP-I. Run it in a **Kaggle CPU session** (no GPU quota) that adds the edit job's output as input. Local scoring is a fallback; insightface may not build on Windows py3.13, so use py3.11.
- Caches (keys and paths):
  - `cache/clean_edits/<editor>/<prompt_id>_s<strength>_seed<k>/<image_id>.png` + `index.csv` (E0, made once)
  - `cache/arcface/<split>_sources.npz` (embedding, bbox, 5 landmarks, det_score)
  - `cache/purified/<protect_run_id>/<purifier>/<image_id>.png` (seed `derive_seed(seed,"purify",image_id,purifier)`)
  - protected images in `results/<protect_run_id>/images/`
- Later Kaggle notebooks take earlier notebook outputs as inputs instead of re-uploading them.
- Model weights download from HF each session, unauthenticated, no tokens.

### 4.3 Job sizing for a limited, unknown weekly quota

- Every GPU job:
  - smoke run first (5 images, 1 seed, 1 prompt)
  - target ≤ 3 h session time
  - 2×T4 sharding
  - `checkpoint_every`
  - `--resume_from`
- `scripts/estimate_budget.py --profile <p> --methods ... --purifiers ...` multiplies grid sizes by **measured** per-unit seconds from the smoke runs (`results/*/gpu_time.json`) and prints GPU-hours per job **before** launch. A plan is only accepted if it fits the user's weekly hours.
- Evaluation profiles (in config, `eval_profiles:`):
  - `smoke`: 5 images, 1 seed, 1 prompt, sd15 s=0.55
  - `dev`: val split (20), 3 seeds, 1 prompt, sd15 s=0.55 + ipadapter_faceid
  - `core`: test (100), 3 seeds, 3 prompts, sd15 s=0.55 + ip2p + ipadapter_faceid
  - `full`: test + messy (120), all editors, the sd15 strength sweep, 3 prompts, 3 seeds
  - Grid sizes per method (no purifier): `core` sd15 = 100×3×3 = 900 edits; `full` sd15 = 120×3×3×3 = 3240 edits; each other editor in `full` = 120×3×3 = 1080 edits.
- **Policy:** tuning and retry decisions in Phases 3–6 use `dev` on val only. Each phase gate is measured once on test with `core`. Phase 7 uses `full` for E0, E3 (no purifier), v1 and v2-final; everything else uses `core`.

### 4.4 Splits (Phase 1, frozen once)

`scripts/make_splits.py` → `data/splits/{train,val,test,messy}.txt` + `splits_meta.json`.
- Identity labels: CelebAMask-HQ's `CelebA-HQ-to-CelebA-mapping.txt` joined with CelebA's `identity_CelebA.txt`. The latter is **not** in CelebAMask-HQ; ask the user (Section 7).
- Fallback if identity labels are unavailable: dedupe by ArcFace (pairs with sim above 0.5 treated as the same person; threshold unverified, report it).
- Test = v1_eval70 (deduped by identity) + 30 more images from the pool, excluding the v1 calibration images. One image per identity. Train/val are disjoint identities.
- Messy (20): ask the user for a source. Fallback: CelebA-HQ images chosen for extreme pose (insightface pose) or low light, plus pre-compression (JPEG q50–70, down-up), each recorded.
- v2 clean images: 1024 → 512 PIL bicubic, saved as PNG. v1 is re-run on these PNGs for all paired comparisons (Phase 1 job J1.1).

---

## 5. Phases 1–8

### Phase 1: Evaluation harness (Half B)

- **Modules:**
  - `src/eval/editors.py`: `class Editor(Protocol): name; load(); edit(img: PIL, prompt_id, prompt, strength, seed) -> PIL`. `build_editor(name, cfg, device)`.
    - `SD15Img2Img` (fp16, default scheduler and steps = v1 settings)
    - `SDXLImg2Img` (fp16-fix VAE, 512 → 1024 bicubic upsample recorded, `enable_model_cpu_offload` if needed)
    - `InstructPix2Pix` (instruction prompts, image_guidance 1.5)
    - `SDInpaint` (keep-face mask from the CelebAMask-HQ parsing labels: background ∪ cloth; insightface-bbox ellipse fallback for messy photos)
    - `IPAdapterFaceID` (insightface embedding of the input image → `ip_adapter_image_embeds`; also the FaceID LoRA if required)
    - One generator per seed; batch seeds where memory allows.
  - `src/eval/metrics.py` additions: `FaceIDScorer` (insightface buffalo_l, largest face, `ctx_id` GPU else CPU; logs which onnxruntime provider was active), `arcface_sim`, `clip_i` (clip_eval ViT-L/14), `edit_lpips`.
  - `src/eval/stats.py`: `psr`, `psr_threshold_from_impostors`, per-image PSR bootstrap, paired tests.
- **Scripts:**
  - `scripts/make_splits.py`
  - `scripts/prepare_data.py` (PNG 512)
  - `scripts/evaluate.py --protected_dir X --method_name v1 --purifier none --profile core --stage edit|score|all --run_id ...` (the handoff's required call is `evaluate.py --protected_dir X --purifier none`)
  - `scripts/make_tables.py --table identity`
- **Kaggle jobs:**
  - J1.0 smoke of all editors (5 images; also records per-edit seconds)
  - J1.1 protect v1 on test + messy (GPU, Phase 0 per-image cost × 120)
  - J1.2 E0 clean edits (`full` profile, cached)
  - J1.3 v1 edits (`full`)
  - J1.4 scoring (CPU session) + thresholds
- **Gate:** CSV + summary table with CIs on v1 outputs. Report v1 identity retention (ArcFace sim, PSR per editor, no-face rate). **STOP.**

### Phase 2: Baselines and controls

- **Modules:**
  - `src/baselines/noise_control.py`:
    - `uniform_linf(x, eps_255, seed)` (U(−ε, ε))
    - `sign_linf(x, eps_255, seed)` (±ε, highest-energy variant, extra control)
    - `gaussian_matched_psnr(x, target_psnr, seed)`: bisection on σ per image, after clipping and uint8 rounding, targeting that image's v1 PNG PSNR
    - CPU only.
  - `src/baselines/photoguard.py`: wraps or follows the official MadryLab encoder attack at **our** budget (4/255 standard) and our step/iteration count. The official code's default eps differs; record it.
  - `src/baselines/mist.py`: official psyker-team code. If it will not run on the Kaggle Python/diffusers versions, record why (exact error) and compare against what runs.
  - `src/eval/purify_attacks.py`: `class EvalPurifier: name; apply(img: PIL, seed) -> PIL` for every `purifier_eval` entry. PIL JPEG, PIL resize and a separately loaded VAE/SD instance. **Never** imported from `src/protect`.
- **Jobs:** noise controls (CPU) → PhotoGuard protect (GPU) → Mist protect (GPU) → purify v1 under JPEG q90/75/50, down-up 0.5, VAE round trip, DiffPure-SD t*=0.05/0.1/0.2 (+ GrIDPure if feasible) → edits (`core` for methods; purifier sweep rows use `core` with 1 prompt if `estimate_budget` says the 3-prompt grid does not fit; record that choice) → score (CPU). Also run E0 under each purifier (clean → purifier → edit) as a control.
- **Gate:** one table with PSR for noise controls, PhotoGuard, Mist, v1, and v1 × each purifier. **STOP.** Decision rule from the handoff: if v1's PSR barely drops under purification, recommend skipping Phase 5.

### Phase 3: Cheap purifiers in the loop (Half A)

- **Modules:**
  - `src/protect/losses.py`: `l_vis(sd, xp, z_clean)`, `l_sem(clip, xp, target)`, `l_str(x, x_pub)` (11×11 SSIM, fp32), `l_den(...)`, `l_id(...)` stubs.
  - `src/protect/models.py`: `SDModels` with `encode_mean(x, scaled: bool)`, `scaling_factor` taken from `vae.config`, `unet`, `scheduler.alphas_cumprod` on device, `null_cond`.
  - `src/protect/purifiers.py`: `TrainPurifier(nn.Module)` subclasses `DiffJPEG(q_range)` (kornia `jpeg_codec_differentiable`, fp32, H and W multiples of 16), `DownUp(scale_range, modes)`, `VAERoundTrip(sd)`, `Identity`; `PurifierMixture(purifiers, probs).sample(gen)`.
  - `src/protect/pgd.py`: `protect(x, cfg, models, mixture, gen) -> (x_adv, log)`. This is the handoff loop with these fixes:
    - `l_str` on the published image computed once per iteration
    - per-draw `autograd.grad` with `runtime.loss_scale`
    - log the zero-gradient pixel fraction and per-term values
    - random start per `budget.random_start`
    - projection then clamp
    - if `eot.spatial.enabled`, recompute `z_clean` per draw
  - Method preset `methods.v2_cheap`.
- **Jobs:** smoke (5 val images, iters 10) for memory and time per draw (K=4, then 8 if it fits) → `dev` on val → one `core` test run under JPEG, down-up and VAE round trip, plus the held-out down-up variant.
- **Gate:** PSR under these purifiers clearly above Phase 2's P0, PSNR ≥ 38. (The handoff's E4 gate says "at least doubles"; see risk R20.) **STOP.**

### Phase 4: One-step SD purifier and L_den

- `SDOneStepPurifier` with all the fixes in R2. Mixture probability 0.3.
- `L_den` (AdvDM-style noise-prediction error):
  - when the drawn purifier is the one-step purifier, reuse its UNet pass (noise target `n`, prediction `eps`)
  - otherwise run one UNet pass on `encode(P(x+δ))·sf`
- Memory: UNet fp16 + `unet.enable_gradient_checkpointing()`, VAE gradient checkpointing, K sequential draws with per-draw `autograd.grad`. Fallbacks: K=2, then 512 px (already 512).
- Measure peak memory in the smoke run.
- **Eval:** held-out DiffPure-SD t*=0.1/0.2/0.3, plus one truly different-model purifier (R17).
- **Gate:** held-out DiffPure PSR clearly above Phase 3 at equal PSNR. Otherwise raise the share once and retry. **STOP.**

### Phase 5 (optional): Anti-purification terms

- Only if the Phase 2 rule says so.
- `l_hf` (patch-wise DCT high-frequency energy of the one-step reconstruction) and `l_mono` (reconstruction error increasing with t, computed at two sampled timesteps). Weights 0.5 / 0.5.
- Optional 3–5 step DDIM purification with checkpointing in the last 5–10 PGD iterations.
- Ablate each term alone on `dev`, confirm on `core`.
- Go/no-go: drop if gains are within the CI. **STOP.**

### Phase 6: Encoder ensemble, L_id, protection-aware tuning

- **Ensemble:** SD1.5 VAE + SDXL VAE (fp16-fix, its own scaling factor). Add a CLIP ensemble only if the Phase 7 ablation (it can be run early on `dev`) shows that L_sem helps.
- **L_id:** differentiable face embedder **different from the evaluation model** (R16).
  - Recommended: insightface `w600k_r50.onnx` converted with `onnx2torch` for the loss.
  - Evaluation: buffalo_l ArcFace, plus a second face-recognition model (facenet-pytorch).
  - Alignment: landmarks detected once on the clean image (no gradient), fixed affine `kornia.geometry.warp_affine` crop on the VAE reconstruction of x+δ.
- **Normalization:** per-term scales measured at δ=random start on the calibration images, stored in config `loss_norm`.
- **Optuna:** the full objective (30–50 trials × 20 val images × protect + edit) is likely far beyond a small weekly quota (estimate it with `estimate_budget.py` first). Proposed cheap proxy: 10 val images, iters 20, 1 prompt, 1 seed, sd15 s=0.55 + FaceID editor, 2 purifiers, then confirm the top 3 trials on the full val set. Constraints PSNR ≥ 38, SSIM ≥ 0.95 handled as in v1 (graded infeasibility penalty).
- **Randomization:** random start + random transform seed per image (already per-image seeds).
- **Gate:** tuned config beats the hand-set one on val PSR. **STOP.**

### Phase 7: Final experiments

- Freeze: `git tag v2.0-final`; config `methods.v2_final`.
- Run E0–E10 on 100 test identities, 3 seeds (`full` for headline rows, `core` otherwise).
- E9 ε sweep (2, 4, 6, 8) with a PSR vs PSNR plot.
- E8 adaptive attacker: `src/eval/adaptive_ae.py`, a small UNet autoencoder trained on (protected, clean) pairs from **disjoint** identities (train pool beyond train/val/test; protect around 200 images; size from the budget estimate), 128 px patch training, applied as a purifier.
- E7 ablations, including the phase ablation (L_vis only, L_den only, both, both without L_sem), PSR per editor.
- E10 human study materials:
  - `scripts/make_human_study.py` builds 20 pairs, 2AFC and same-person forms (static HTML/CSV for the students' form tool)
  - `scripts/analyze_human_study.py`: binomial test vs 50%, CI
- All tables and figures from `make_tables.py` / `make_figures.py`. **STOP** with full results and failure cases.

### Phase 8: Write-up and demo

- Report sections: threat model first; results with CIs; failure cases; limitations (inpainting with a face mask removes the protected pixels; surrogate gap; adaptive attacker result).
- `scripts/demo.py --image in.png --method v2_final --editor sd15_img2img`: protected PNG + before/after panel (GPU; CPU only with very few iterations, documented).
- README rewritten: reproduce the main table from the saved config, list the splits and requirements, credentials policy.

---

## 6. Technical risks in the handoff reference code and plan

- **R1. Latent scaling.** `l_den` calls `m.encode(xp)` and feeds it to the UNet. UNet inputs must be `latent · vae.config.scaling_factor` (0.18215 for SD1.5, **0.13025 for SDXL**). L_vis stays unscaled to keep α's scale (F8). Use `SDModels.encode_mean(x, scaled=...)` explicitly, and never hard-code 0.18215 (`SDOneStepPurifier` default `s=0.18215` must come from the config).
- **R2. SDOneStepPurifier bugs and fragility.**
  - `self.ab[t]`: `alphas_cumprod` is a CPU tensor while `t` is on CUDA. Move it to the device and compute `sqrt` and division in fp32.
  - `torch.randint` and `randn_like` use the global RNG: pass a generator.
  - `torch.randint(t_min, t_max)` excludes `t_max`.
  - `null_cond` must be text-encoder output for `""`, shape (1,77,768), cast to the UNet dtype.
  - The final `clamp(-1,1)` zeroes gradients at saturated pixels (minor).
  - The decoder backward is memory-heavy: use checkpointing.
- **R3. `z_clean` computed once** is only valid for geometry-preserving purifiers. With spatial EoT (v1's crop), apply the same crop to x and recompute `z_clean` under `no_grad`. The plan keeps spatial EoT off in v2 by default.
- **R4. Gradient flow through `latent_dist.mean`** is fine (differentiable). Attackers' img2img uses `latent_dist.sample(generator)` and IP2P uses `.mode()`. The SD VAE posterior std is small, so mean ≈ sample, but evaluation uses the real pipelines, so any gap shows up in the measurements.
- **R5. fp16 + autocast on T4.** T4 has no bf16, so use fp16 autocast for the UNet. Gradients w.r.t. pixels can underflow to 0, and `sign(0)=0` means no step. Use `loss_scale` and log the zero-gradient fraction. Keep VAE encode in fp32 for v1 reproduction. The SDXL VAE NaNs in fp16, so use the fp16-fix VAE.
- **R6. Gradient checkpointing in diffusers.** Some versions apply checkpointing only when `self.training` is True. Check the pinned version's `forward`; if needed call `.train()`. SD1.5 UNet and VAE have dropout 0, so outputs are unchanged; verify numerically on one input.
- **R7. Kornia differentiable JPEG.** Needs inputs in [0,1], H and W multiples of 16, quality as a (B,) tensor, fp32. It approximates libjpeg rounding, so evaluation must use real PIL JPEG. `K.RandomJPEG` with `same_on_batch` is fine (v1).
- **R8. Memory on a 16 GB T4.** UNet fp16 (~1.7 GB of weights) + VAE + activations with checkpointing should fit at 512 px with sequential draws (estimate, verify by measuring `max_memory_allocated` in the Phase 4 smoke run). K=8 is unlikely to fit without checkpointing.
- **R9. insightface on Kaggle.** `insightface==0.7.3` compiles Cython at install time. `onnxruntime-gpu` must match Kaggle's CUDA/cuDNN or it silently falls back to CPU (log the active provider; CPU is acceptable for scoring). The buffalo_l pack downloads from insightface's GitHub releases on first use (availability unverified). Model license is non-commercial research, which is acceptable for an FYP; state it. Local Windows py3.13 builds are likely painful, so score in Kaggle CPU sessions or a py3.11 venv.
- **R10. IP-Adapter FaceID vs InstantID.** FaceID (SD1.5) needs insightface embeddings, diffusers ≥0.28 `load_ip_adapter(..., image_encoder_folder=None)` with precomputed `ip_adapter_image_embeds`, and possibly its LoRA (verify the current recipe). InstantID needs SDXL, a ControlNet, antelopev2 models (no official stable download) and a community pipeline: fragile. Decision: IP-Adapter FaceID. InstantID only if time allows; otherwise document it as a limitation. The FaceID editor reads identity through ArcFace, not the VAE, so v1 and Phase 3–5 methods are expected to give little protection there until L_id (Phase 6). This is a measured result to report, not a bug.
- **R11. L_sem / CLIP question.** SD1.5 img2img never runs a CLIP image encoder, so L_sem (CLIP B/32 toward "a potted plant") probably has no direct path to the edit (unverified; the E7 phase ablation answers it). It **can game the CLIP-I metric** if the same encoder is used, so CLIP-I uses ViT-L/14 (`clip_eval`) by design. IP-Adapter (non-FaceID) uses a CLIP ViT-H image encoder, not B/32.
- **R12. Quantization.** Published images are uint8 PNG. All of Half B reads only saved PNGs. With an integer `eps_255`, rounding keeps |diff| ≤ eps_255.
- **R13. Model availability.** The `runwayml/*` repos were taken down (2024). The v1 id resolved on Kaggle in the stored log (likely a redirect, unverified). Use the `stable-diffusion-v1-5/*` mirrors for v2 and pin revisions. The env probe compares VAE sha256.
- **R14. Kaggle hardware and limits.** Pick T4 (P100 sm_60 may be unsupported by the image's torch build: check `get_arch_list`). Session length and output limits are unverified: design for ≤ 3 h jobs, checkpoints and resume. 2×T4 roughly doubles throughput per quota hour if the quota counts session time (verify with the user).
- **R15. Determinism.** GPU kernels (bicubic backward, atomics) are nondeterministic. Use per-image sha256 seeds. Reproduction is judged statistically.
- **R16. Metric gaming by L_id.** Using the same ArcFace for L_id and evaluation inflates PSR (white-box on the judge). Use a different model for the loss, evaluate with buffalo_l **and** a second face-recognition model, and report both.
- **R17. Held-out DiffPure shares the SD1.5 UNet** with the one-step surrogate (gray-box, not truly held-out). Add one purifier built on a different diffusion model (pixel-space DiffPure with a public ADM checkpoint, or the GrIDPure official code) and label which purifiers share weights.
- **R18. Optuna cost** (see Phase 6). Estimate before launching and use the proxy objective.
- **R19. PSR threshold dataset.** The impostor distribution needs true identity labels (R4.4). Without them, "different people" pairs may include the same person and bias the threshold downward.
- **R20. Gate arithmetic.** "PSR at least doubles vs v1" is trivial if v1's purified PSR ≈ 0 and impossible if it is above 50%. Report the absolute numbers with CIs and ask the user if the gate is degenerate. Do not reinterpret it silently.
- **R21. SDXL resolution.** Upsampling 512 → 1024 is itself a mild purification. Document it, and report SDXL separately.
- **R22. Inpainting masks.** Face masked by the attacker means those pixels are discarded (handoff); we use keep-face masks. CelebAMask-HQ parsing labels exist only for CelebA-HQ images; messy photos use the bbox fallback.

---

## 7. "Ask the user" decisions (handoff Section 9 plus blockers found here)

Ask one or two at a time, only when they block work.

| # | Question | Blocks | When to ask |
|---|---|---|---|
| 1 | Deadline and dates of the final report and demo | Scheduling and cut order (Phases 5, SDXL, Mist, human study) | Now, with #2 |
| 2 | Kaggle GPU hours per week; which team members have accounts; whether the quota counts session time (2×T4 sharding benefit) | Sizing every GPU job; Phase 0 full run can proceed once the smoke run is measured | Now |
| 3 | Where the dataset subset lives: which Kaggle CelebAMask-HQ dataset (v1 used sourceId 3973359). **Do you still have v1's Phase 1 output zips** (protected images, `evaluation_70_manifest.csv`, `optimal_hyperparameters.json`, `stage1_per_image_metrics.csv`)? | Phase 0 (zero-GPU budget check, exact 70-image list and weights) | Phase 0, first |
| 4 | Is the human study required, and are 15 raters realistic? | Phase 7 (E10 materials) | Before Phase 7 |
| 5 | Are the PhotoGuard and Mist baselines required or optional? | Phase 2 scope | Before Phase 2 |
| 6* | Do you have CelebA `identity_CelebA.txt` (identity labels)? Where should the 20 "messy" photos come from (must be from the project dataset)? | Phase 1 splits and PSR threshold | Start of Phase 1 |
| 7* | Accept the proposed Phase 0 reproduction tolerance (±0.5 dB PSNR, ±0.005 SSIM, ±0.03 LPIPS)? | Phase 0 gate verdict | In the PHASE_0 report |
| 8* | May branches be pushed to `origin` so Kaggle can `git clone` a commit? Otherwise use the dataset-bundle route | Packaging convenience only (not blocking) | Phase 0, only if needed |
| 9* | L_id model vs evaluation model choice (R16) | Phase 6 | Start of Phase 6 |

(*) Not in handoff Section 9. They come from blockers found in this review and are raised only when they block.

---

## 8. Operating-rules checklist (applies to every phase)

- **No deletions.** The only file move is `git mv src/eval.py src/legacy_eval.py`. Anything else to be removed goes to `_to_delete/` (git-ignored), and the user is told.
- **Work on copies.** Never edit `notebooks/*.ipynb`, `notebooks/v1_reference/*`, `docs/HANDOFF_v2.txt`, the user's datasets or v1 outputs. Data preparation writes to new folders (`data/v1/`, `data/images/`, `/kaggle/working/...`).
- **No credentials.** HF downloads unauthenticated. The user signs in to Kaggle and uploads or downloads. No tokens in code, config, logs or notebooks. Notebooks print no environment secrets.
- **Never invent numbers.** Templates hold placeholders. Estimates are labelled "estimate". Unverified ids and claims are labelled "unverified". Never mention "RSAS".
- **Smoke first.** 5 images, 1 seed, few iterations, for every new script and job, before any full run.
- **STOP at every gate** with `reports/PHASE_<n>.md`. Report negative results with numbers. Never quietly change a gate.
- **Statistics.** Every reported number states n, mean ± 95% bootstrap CI over images and 3 seeds, paired against v1 on the same images. Never tune on test.
- **Privacy.** Identifiable CelebA-HQ derivatives (protected, purified and edited images) are never committed to the public repo (`.gitignore`: `results/**/images/`, `results/**/edits/`, `results/cache/`, `data/images/`, `data/v1/`, `kaggle/dist/`, `_to_delete/`, `*.zip`). Only CSV and JSON results and file-name lists are committed.

---

### Critical files for implementation (all paths absolute)

- C:\Users\karth\Documents\DiffShield\docs\HANDOFF_v2.txt (brief; the authority)
- C:\Users\karth\Documents\DiffShield\notebooks\v1_reference\phase1_v1.ipynb (exact patch text in cell 0, metric function in cell 1, selection logic in cells 3–4, eval loop in cell 8)
- C:\Users\karth\Documents\DiffShield\src\losses.py (SSIM patch + injection kwargs)
- C:\Users\karth\Documents\DiffShield\src\optimizer.py (loss_fn injection; v1 PGD semantics)
- C:\Users\karth\Documents\DiffShield\src\eot.py (same_on_batch patch)
- C:\Users\karth\Documents\DiffShield\src\eval.py (to be moved to src\legacy_eval.py with a compat shim)
- C:\Users\karth\Documents\DiffShield\notebooks\v1_reference\phase2_v1.ipynb (v1 edit settings, Phase 1 reference)
- New, central: C:\Users\karth\Documents\DiffShield\config\config.yaml, C:\Users\karth\Documents\DiffShield\scripts\run_protect.py, C:\Users\karth\Documents\DiffShield\src\utils\runlog.py
