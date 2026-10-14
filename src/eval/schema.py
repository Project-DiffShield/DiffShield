"""Column constants for results CSVs."""

PROTECT_COLUMNS = [
    "run_id", "method", "method_cfg_hash", "image_id", "split", "seed", "editor", "prompt_id", "strength",
    "purifier",
    "src_file", "out_file", "eps_255", "step_255", "iters",
    "v1_psnr", "v1_ssim", "v1_lpips", "v1_linf_model",
    "png_psnr", "png_ssim", "png_lpips", "png_maxabs_u8", "png_frac_at_bound",
    "gpu_seconds", "peak_mem_mb", "created_utc",
]

# Half-B columns (Phase 1+), see docs/V2_PLAN.md section 4.1
EDIT_COLUMNS = [
    "run_id", "method", "method_cfg_hash", "image_id", "split", "seed", "editor", "prompt_id", "strength",
    "purifier", "protect_run_id", "eps_255", "purifier_params", "editor_params", "source_path",
    "protected_path", "purified_path", "edit_path", "clean_edit_path", "code_git_hash", "created_utc",
    "face_detected", "n_faces", "det_score", "arcface_sim", "fr2_sim", "edit_lpips", "clip_i",
    "inv_psnr", "inv_ssim", "inv_lpips", "gpu_seconds_edit", "gpu_seconds_purify",
]
