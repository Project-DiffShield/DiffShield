# DiffPure-style pixel-space face DDPM cleanup (evaluation only)

## What it is
A held-out cleanup for Half B (evaluation harness) that does not share weights with Stable Diffusion 1.5. Our
protection is optimised against SD1.5, so the existing `diffpure_0.x` cleanups (SD1.5 img2img) are only gray-box;
this one is independent of SD1.5.

Pipeline (`PixelDDPMPurifier` in `src/eval/purify_attacks.py`): resize 512 -> 256 (PIL bicubic) -> [-1,1] ->
forward-noise to timestep t* of the model's 1000-step linear schedule, `x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps`,
with a per-image CPU generator seeded by the edit seed -> reverse ancestral sampling with `DDPMScheduler.step` down to 0
-> clamp -> resize 256 -> 512 (same bicubic). Model: `google/ddpm-celebahq-256` (diffusers `UNet2DModel` +
`DDPMScheduler`; Apache-2.0, anonymous download, `diffusion_pytorch_model.bin` 454,853,117 bytes, as verified by the
coordinator). fp16 on GPU, fp32 on CPU. Weights are lazy-loaded; if they cannot be loaded the purifier raises
`NotAvailable` (the runner records it as a documented skip).

## What it is NOT
* Not the official DiffPure checkpoint (that URL returns 403). Always call it "DiffPure-style" in tables and text.
* A 256x256 CelebA-HQ model: the 512 photo is downsampled first. The resize alone already removes high frequencies,
  so a resize confound exists. Always report the gap to `downup_256` (512->256->512, same resampling, no diffusion).
  Only the difference between `diffpure_ddpm256_*` and `downup_256` is attributable to the denoiser.
* Not a certified or adaptive attack; it is a held-out, non-adaptive cleanup.
* CelebA-HQ face prior: outputs may drift toward CelebA-HQ faces (identity shift) at larger t*. Not measured.

## Names and settings
| name | what | t* | UNet calls/photo |
|---|---|---|---|
| `downup_256` | control, resize only | - | 0 |
| `diffpure_ddpm256_t100` | pixel DDPM | 100 / 1000 | 100 |
| `diffpure_ddpm256_t200` | pixel DDPM | 200 / 1000 | 200 |

Parameters live in `config/parts/eval.yaml` (`eval.purifiers.*`: `model_id`, `t_star`, `steps`, `size`, `resample`) and
are mirrored in `config/config.yaml` (`purifier_eval`, `purifiers_heldout`). `steps: null` means the full chain
(`steps = t*`). Setting `steps: N` (< t*) uses an evenly strided timestep list with `DDPMScheduler.step` (a cheaper
strided ancestral variant, covered by a unit test for call count only). All values are UNVERIFIED on real weights.

## How to run
New tier `HELD` (test split; methods clean, v1, v2_main, v2_final; SD1.5 img2img strength 0.55, rotated prompt; the three
names above). Purifier filter works on names inside a tier:

    python scripts/evaluate.py --tier HELD --method v2_final --purifier downup_256,diffpure_ddpm256_t100,diffpure_ddpm256_t200 \
        --protected_dir <dir> --split test ...
    python scripts/evaluate.py --tier HELD --method clean --purifier all ... --count_only   # row count check

In a jobs file (`kaggle/jobs.yaml` was deliberately not edited here): `tier: HELD`, `purifier: "downup_256,diffpure_ddpm256_t100,diffpure_ddpm256_t200"`.
To use them in another tier, add the names to that rule's `purifiers` list.

## Expected cost (UNVERIFIED planner estimate, unmeasured)
About 5-10 s per photo at t*=100 on a T4 (fp16, batch 1 per photo; 100 UNet calls at 256x256), roughly double at
t*=200. Rows are `images x methods x purifiers x seeds`, and the purified image is cached across editors of one run
(`cache_purified`), so this runs once per (method, photo, purifier). Replace with a measurement after the smoke.

## Checklist for the first GPU smoke
1. `python -c "from diffusers import UNet2DModel, DDPMScheduler; UNet2DModel.from_pretrained('google/ddpm-celebahq-256'); DDPMScheduler.from_pretrained('google/ddpm-celebahq-256')"` with internet ON (Kaggle) succeeds; note `diffusers` version; then pin `revision` in `eval.purifiers.*`.
2. Config `scheduler_config.json` is `num_train_timesteps=1000`, linear betas, `clip_sample` true: confirm `sched.alphas_cumprod[99]` and `[199]` printed.
3. Run `diffpure_ddpm256_t100` on 5 test photos: output size 512x512 RGB, no NaN, visually a slightly smoothed face (not a different person); record s/photo and peak GPU memory (fill the cost line above).
4. Check determinism: same photo and seed twice gives identical PNG bytes on the same GPU.
5. Check fp16 vs fp32 sanity on one photo (mean abs pixel difference small); if fp16 produces NaN/black images, set fp32.
6. Run `downup_256` alongside and confirm it differs from the DDPM outputs.
7. Confirm the runner logs `NotAvailable` cleanly (not a crash) when the model is unreachable.
8. Add both names to the report tables labelled "DiffPure-style (pixel DDPM, google/ddpm-celebahq-256)", with the `downup_256` gap.
