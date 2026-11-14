# Exploration set X0..X5: three ideas for a big PSR gain over `v2_tgt_zero_mid`

**Status: exploration on the 28 `tune_search` photos. These results are NOT a claim for the test set.** Adopting any variant would need a fresh 100-photo test run (with the paired-bootstrap rule of `scripts/paired_compare.py`). `tune_confirm` is not used (T03 uses it as a gate), nor are the test photos.

## Why
The current final method is `v2_tgt_zero_mid` (preset `v2_next` with defaults reproduces it: alpha 0.55, beta 0.9024, gamma 60.53, eta 0, p_onestep 0, one target-zero latent term, cleanups JPEG / down-up / VAE round trip at 0.35 / 0.35 / 0.30, 40 iterations, K = 4, eps 4/255, step 1/255). PSR (protection success rate) = share of edits whose ArcFace similarity to the source falls below 0.1005 (or no face is found); higher is better. T02 showed the IP-Adapter FaceID editor at PSR about 0 for the current method: it reads identity from a face embedding, not through the SD1.5 VAE that the current loss attacks. The variants test three ideas (bigger budget, longer optimisation, an identity term).

## Variants (each = `v2_next` defaults + ONE change)
Job id = variant id (one job per variant; the brief said "X1..X6", we number X0..X5 so that job and variant match one to one).

| Job / variant | Preset (`config/parts/protect.yaml`) | Change | What it tests |
|---|---|---|---|
| X0 base | `v2_next` | none (fresh baseline at full settings) | the reference every other variant is compared to, produced in the same conditions (same photos, editors, seeds, code) |
| X1 eps8 | `v2_x_eps8` | `budget.eps_255 = 8` (step_255 stays 1.0) | does a larger budget buy PSR, and how much visibility does it cost |
| X2 it100 | `v2_x_it100` | `budget.iters = 100` | is the 40-iteration optimisation under-converged |
| X3 id_direct_a | `v2_x_id_direct_a` | `loss.lambda_id = 2.0` (**UNVERIFIED guess**), `face.id_mode: direct` | pull the protected image's face embedding away from the source's, straight on x+delta (no VAE), through FaceNet (not the evaluation ArcFace) |
| X4 id_direct_b | `v2_x_id_direct_b` | `loss.lambda_id = 10.0` (**UNVERIFIED guess**), direct | same, 5x stronger |
| X5 id_recon | `v2_x_id_recon` | `loss.lambda_id = 2.0` (**UNVERIFIED guess**), `face.id_mode: recon` | the existing identity loss (embedding of the VAE reconstruction of x+delta); separates "an identity term at all" from "direct vs through the VAE" |
| X6 eps8_tuned | `v2_x_eps8_tuned` | `v2_next_tuned` (T03 trial 14: cleaner + denoiser) with `budget.eps_255 = 8` (added 2026-11-14 after X0-X5) | does the tuned cleaner+denoiser version gain more from a larger budget than the old weights (X1)? Same visibility caveat as X1 |

X3-X5 target the FaceID editor, whose identity path bypasses the VAE. `id_mode: direct` minimises cos(FaceNet(source face crop), FaceNet(crop of x+delta)) by PGD; the source embedding is computed without gradient, so the gradient flows only through the protected image. Default `id_mode` is `recon` (the old behaviour; existing presets have `lambda_id` 0, so nothing is loaded and nothing changes).

### Surrogate vs judge
The identity loss must not use the evaluation recogniser (insightface `buffalo_l`, ArcFace w600k_r50), or PSR would be inflated (white-box on the judge). Backend stays `facenet` (facenet-pytorch `InceptionResnetV1`, vggface2). A cheap guard rejects `backend: onnx2torch` when `face.onnx_path` contains `w600k`, `buffalo` or `arcface` unless `face.allow_eval_model: true` is set. This is a file-name check only, not a proof.

### Face crop and its limitation
No detector runs in the loop. `face.bbox: null` means the central 60 % of the image (x and y from 20 % to 80 %), resized to 160 px, applied to the clean source and to the protected image. For aligned, roughly centred CelebA-HQ 512 px faces this is a reasonable but crude crop: it will include hair/background on some photos and cut the chin or forehead on others, and FaceNet sees a differently framed face than the ArcFace detector and alignment used for scoring. UNVERIFIED how large the effect is. State it as a limitation in any write-up. `run_protect.py` does not pass a per-photo bbox.

## What decides success
Per variant, versus the X0 run of the same exploration (paired by photo):
1. PSR over all cells (5 cleanups x 2 editors: none, jpeg_q75, downup_0.5, vae_rt_sd15, diffpure_t0.2 for sd15_img2img strength 0.55 and ipadapter_faceid; edit seeds 1, 2, 3), and separately the sd15 and ipadapter means. Difference to X0 with a paired bootstrap 95 % interval (resample photos, 2000 draws, seed 2026).
2. Invisibility: per-photo PSNR >= 38 and SSIM >= 0.95 (the T02 constraint); mean and minimum, and the number of violating photos.
3. A variant is interesting only if the all-cell PSR gain has an interval above 0 **and** (for X2-X5) it keeps zero or near-zero violations. **X1 (eps8) changes the threat / visibility trade-off and cannot be compared with the others at equal invisibility;** it is a "what does a bigger budget buy" reference and is expected to violate PSNR/SSIM more.
Everything here is a screening signal on 28 photos (wide intervals are expected).

## Running
`kaggle/jobs.yaml` jobs X0..X5, details in `docs/KAGGLE_LAUNCH.md`. Each is independent, `gpu_t4x2`, inputs = code bundle + celebamaskhq (same as T04), **Internet ON** (X3-X5 pip-install `facenet-pytorch --no-deps` and download the vggface2 weights from its GitHub release on first use; the `facenet_check` step fails the job before any GPU work if that is unavailable). Smoke: `PARAMS = {"smoke": True}` (2 photos, 3 iterations). Afterwards:

    py -3.11 scripts/explore_compare.py <unzipped X0 output> <unzipped X1 output> ... --out_dir results/explore

writes `explore_cells.csv`, `explore_summary.csv`, `explore_compare.md` (any subset of the variants; the diff column needs X0).

## Runtime estimates
Measured: protection about 116 s per photo on one T4 at 40 iterations, K = 4; about 2.63 s per edit. Everything below is **UNVERIFIED** arithmetic:
- protect, 28 photos on 2 GPUs: X0, X1 about 28 x 116 / 2 = 27 min; X2 (100 iterations) about 2.5x, about 68 min; X3-X5 add a FaceNet forward/backward per iteration (X5 also a VAE pass), expected a modest overhead, plus model loading.
- evaluate: 280 tier rows x 3 seeds = 840 edits, about 840 x 2.63 s = 37 min of edit time, about 18 min on 2 GPUs, plus purifier cost (diffpure 0.2 is the expensive one) and model loading: roughly 30-60 min.
- so about 1 to 1.5 h per job (X2 about 2 h). Per-row resume applies to protection and evaluation.

## Files
`src/protect/face_loss.py` (`IdentityLoss(mode=...)`, guard), `src/protect/v2_registry.py` (presets, `budget.iters_cap`), `config/parts/protect.yaml` (`face.id_mode`, presets), `config/parts/eval.yaml` (tier `EXPL`), `kaggle/jobs.yaml`, `scripts/explore_compare.py`, tests `tests/test_protect_idmode.py`, `tests/test_explore_compare.py`.
`budget.iters_cap` is a smoke-only knob (`--set budget.iters_cap=N` lowers `budget.iters` to N if larger). It is needed because a preset's own `budget.iters` (X2: 100) wins over `--set budget.iters`, so the smoke flag could not otherwise shorten X2. It is 0 (off) in real runs.
