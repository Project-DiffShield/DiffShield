// DiffShield Phase II external review deck, following 1_presentation.tex (Amrita beamer template, Madrid theme).
const pptxgen = require("pptxgenjs");
const OUT = process.argv[2] || "DiffShield_PhaseII_External_Review.pptx";

const MAROON = "A4123F", TINT = "F6E8EC", INK = "1F1F1F", GREY = "5F5F5F", WHITE = "FFFFFF", PH = "8A8A8A";
const FONT = "Arial";
const W = 13.333, H = 7.5, MX = 0.45;
const M = "−"; // minus sign

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "DiffShield: Project Phase II External Review";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

// ---------- rough text-fit check (Arial: average glyph width about 0.5 em) ----------
const warn = [];
function lines(text, pt, wIn) {
  const perLine = Math.max(1, Math.floor((wIn * 72) / (0.5 * pt)));
  return String(text).split("\n").reduce((n, p) => n + Math.max(1, Math.ceil(p.length / perLine)), 0);
}
function check(id, paras, pt, wIn, hIn, spacing = 1.2, gap = 0) {
  const n = paras.reduce((s, p) => s + lines(p, pt, wIn), 0);
  const need = (n * pt * spacing) / 72 + gap * paras.length / 72;
  if (need > hIn) warn.push(`${id}: needs ${need.toFixed(2)}in, box ${hIn.toFixed(2)}in`);
}

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_SLIDE",
  background: { color: WHITE },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: H, fill: { color: WHITE } } },
  ],
});
pres.defineSlideMaster({
  title: "FRAME",
  background: { color: WHITE },
  margin: [0.3, MX, 0.5, MX],
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 1.0, fill: { color: MAROON } } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.1, w: W - 2 * MX, h: 0.52, fontFace: FONT, fontSize: 24,
        bold: true, color: WHITE, valign: "middle", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "sub", type: "body", x: MX, y: 0.6, w: W - 2 * MX, h: 0.34, fontFace: FONT, fontSize: 13,
        color: "F3D3DD", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: W - 0.9, y: H - 0.42, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 11, color: GREY, align: "right" },
});
pres.defineSlideMaster({
  title: "PLAIN",
  background: { color: WHITE },
  slideNumber: { x: W - 0.9, y: H - 0.42, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 11, color: GREY, align: "right" },
});

// ---------- helpers ----------
function frame(section, title, sub) {
  const s = pres.addSlide({ masterName: "FRAME", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (sub) s.addText(sub, { placeholder: "sub" });
  return s;
}
// items: [ [boldLabel, text] | text ]
function runs(items, pt) {
  const out = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    const base = { fontFace: FONT, fontSize: pt, color: INK, bullet: { code: "25B8", indent: 14 }, paraSpaceAfter: 3 };
    if (Array.isArray(it)) {
      out.push({ text: it[0] + ": ", options: { ...base, bold: true } });
      out.push({ text: it[1], options: { fontFace: FONT, fontSize: pt, color: INK, breakLine: !last } });
    } else {
      out.push({ text: it, options: { ...base, breakLine: !last } });
    }
  });
  return out;
}
function block(s, id, x, y, w, h, title, items, pt = 12) {
  const th = 0.36;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: TINT }, line: { color: TINT }, objectName: id + " body" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: th, rectRadius: 0.06, fill: { color: MAROON }, line: { color: MAROON }, objectName: id + " head" });
  s.addShape(pres.shapes.RECTANGLE, { x, y: y + th - 0.08, w, h: 0.08, fill: { color: MAROON }, line: { color: MAROON }, objectName: id + " head base" });
  s.addText(title, { x: x + 0.15, y, w: w - 0.3, h: th, fontFace: FONT, fontSize: 14, bold: true, color: WHITE, valign: "middle", margin: 0, isTextBox: true, objectName: id + " title" });
  const tx = typeof items === "string" ? [{ text: items, options: { fontFace: FONT, fontSize: pt, color: INK } }] : runs(items, pt);
  s.addText(tx, { x: x + 0.15, y: y + th + 0.06, w: w - 0.3, h: h - th - 0.12, valign: "top", margin: 0, isTextBox: true, objectName: id + " text" });
  const paras = typeof items === "string" ? [items] : items.map((it) => (Array.isArray(it) ? it[0] + ": " + it[1] : it));
  check(id, paras, pt, w - 0.3 - (typeof items === "string" ? 0 : 0.2), h - th - 0.12, 1.2, 3);
}
function placeholderBox(s, id, x, y, w, h, head, detail) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: id + " frame" });
  s.addText([
    { text: head, options: { fontFace: FONT, fontSize: 14, bold: true, color: INK, breakLine: true } },
    { text: " ", options: { fontFace: FONT, fontSize: 6, breakLine: true } },
    { text: detail, options: { fontFace: FONT, fontSize: 11, color: PH } },
  ], { x: x + 0.3, y, w: w - 0.6, h, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: id + " text" });
}
// booktabs-style table
function booktab(s, id, x, y, colW, header, rows, opt = {}) {
  const pt = opt.pt || 12, rowH = opt.rowH || 0.42, boldLast = opt.boldLast;
  const thick = { type: "solid", pt: 1.5, color: INK }, thin = { type: "solid", pt: 0.75, color: INK }, none = { type: "none" };
  const cell = (t, i, r, isHead, bold) => ({
    text: t,
    options: { fontFace: FONT, fontSize: pt, bold: isHead || bold, color: INK, align: i === 0 ? "left" : (opt.alignCols || [])[i] || "center",
      valign: "middle", margin: [2, 6, 2, 6],
      border: [isHead ? thick : none, none, isHead ? thin : r === rows.length - 1 ? thick : none, none] },
  });
  const data = [header.map((t, i) => cell(t, i, -1, true))].concat(
    rows.map((row, r) => row.map((t, i) => cell(t, i, r, false, boldLast && r === rows.length - 1))));
  s.addTable(data, { x, y, w: colW.reduce((a, b) => a + b, 0), colW, rowH, autoPage: false, objectName: id });
  rows.forEach((row, r) => row.forEach((t, i) => { if (lines(t, pt, colW[i] - 0.17) > Math.floor(rowH * 72 / (pt * 1.2))) warn.push(`${id} r${r}c${i} wraps: ${t}`); }));
}
function box(s, id, x, y, w, h, text, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: o.fill || TINT }, line: { color: o.line || MAROON, width: 1 }, objectName: id });
  s.addText(text, { x: x + 0.06, y, w: w - 0.12, h, fontFace: FONT, fontSize: o.pt || 11, bold: !!o.bold, color: o.color || INK,
    align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: id + " label" });
  check(id, [typeof text === "string" ? text : text.map((r) => r.text).join("")], o.pt || 11, w - 0.12, h);
}
function arrow(s, x1, y1, x2, y2, id) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color: MAROON, width: 1.5, endArrowType: "triangle" }, objectName: id });
}

// =========================================================================================
// TITLE SLIDE
// =========================================================================================
pres.addSection({ title: "Title" });
{
  const s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: "Title" });
  s.addText("Type 1: Research Project", { x: W - 3.05, y: 0.15, w: 2.85, h: 0.36, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.04,
    fill: { color: MAROON }, fontFace: FONT, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "type badge" });
  placeholderBox(s, "logo", (W - 2.3) / 2, 0.35, 2.3, 1.05, "[Insert logo.png]", "[University logo]");
  s.addText("DiffShield: Protecting Face Photos from Diffusion-Model Editing", { x: 0.8, y: 1.55, w: W - 1.6, h: 0.55, fontFace: FONT, fontSize: 26,
    bold: true, color: MAROON, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "project title" });
  s.addText([
    { text: "23CSE498 – Project Phase II (External Review)", options: { bold: true } },
    { text: "     |     " },
    { text: "Team Number: [GXXX]", options: { bold: true } },
  ], { x: 0.8, y: 2.15, w: W - 1.6, h: 0.38, fontFace: FONT, fontSize: 14, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "course line" });
  const b = { type: "solid", pt: 0.75, color: INK };
  const c = (t, o = {}) => ({ text: t, options: { fontFace: FONT, fontSize: 13, color: INK, valign: "middle", border: [b, b, b, b], margin: [2, 8, 2, 8], ...o } });
  const rows = [[c("S.No", { bold: true, align: "center" }), c("Register Number", { bold: true, align: "center" }), c("Student Name", { bold: true })]];
  for (let i = 1; i <= 4; i++) rows.push([c(String(i), { align: "center" }), c("[CB.SC.U4CSE23XXX]", { align: "center" }), c(`[Student Name ${i}]`)]);
  s.addTable(rows, { x: (W - 7.6) / 2, y: 2.75, w: 7.6, colW: [1.0, 3.2, 3.4], rowH: 0.42, autoPage: false, objectName: "student table" });
  s.addText([
    { text: "Guide: " }, { text: "[Guide Name]", options: { bold: true } }, { text: ", [Designation], Department of CSE" },
  ], { x: 0.8, y: 5.05, w: W - 1.6, h: 0.38, fontFace: FONT, fontSize: 14, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "guide line" });
  s.addText("[Date of Review]", { x: 0.8, y: 5.5, w: W - 1.6, h: 0.32, fontFace: FONT, fontSize: 12, color: GREY, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "date" });
  s.addNotes("Title slide. Fill in team number, register numbers, names, guide and date; insert the university logo.");
}

// =========================================================================================
// GUIDE APPROVAL (mandatory)
// =========================================================================================
pres.addSection({ title: "Guide Approval" });
{
  const s = frame("Guide Approval", "Guide Review & Submission Approval", "Confirmation of Project Review and Approval from Guide");
  placeholderBox(s, "approval", MX + 0.2, 1.3, W - 2 * MX - 0.4, 5.5, "[Insert Screenshot of Guide Approval Email Here]",
    "[Mandatory: Attach a clear, readable screenshot of the email received from your project guide approving this presentation and final review submission]");
  s.addNotes("Mandatory slide: paste the guide's approval email screenshot.");
}

// =========================================================================================
// SECTION 1: PROBLEM & SOLUTION DESIGN
// =========================================================================================
pres.addSection({ title: "Problem & Solution Design" });
{
  const s = frame("Problem & Solution Design", "1. Problem Definition & Research Motivation", "Problem Statement, Identified Literature Gap & Objectives");
  const bw = W - 2 * MX;
  block(s, "problem", MX, 1.2, bw, 1.6, "Problem Statement", [
    ["Core Research Problem", "Diffusion editors (Stable Diffusion img2img, InstructPix2Pix, inpainting, IP-Adapter FaceID) can turn any posted face photo into a convincing fake of that person. Can an invisible perturbation (L∞ ≤ 4/255), added by the owner before posting, make such edits stop preserving the person’s identity?"],
    ["Domain Context", "Non-consensual deepfakes of real people are cheap to make. A defence must run offline, stay invisible and survive the cleanup an attacker applies first (JPEG, resizing, diffusion purification)."],
  ]);
  block(s, "gap", MX, 2.95, bw, 1.75, "Existing Limitations & Research Gap", [
    ["Summary of Existing Approaches", "PhotoGuard [1] and Mist [2] attack the Stable Diffusion encoder/UNet; Glaze [3] and Anti-DreamBooth [4] protect against style mimicry and fine-tuning. Most report image-quality metrics at budgets of 8–16/255."],
    ["Identified Research Gap", "Little evidence on (i) identity, measured with a face recogniser, (ii) robustness to purification such as DiffPure [5], and (iii) invisibility on every photo at a small budget. Our own v1 also had a mislabelled budget and no controls."],
  ]);
  block(s, "objectives", MX, 4.85, bw, 1.95, "Research Objectives", [
    ["Objective 1 (Invisibility)", "PSNR ≥ 38 dB and SSIM ≥ 0.95 on every one of the 100 test photos at a 4/255 budget."],
    ["Objective 2 (Protection)", "Protection success rate (PSR) above clean photos, a random-noise control and v1, with paired 95% confidence intervals that exclude 0."],
    ["Objective 3 (Robustness & generality)", "Measure protection under 13 held-out cleanups and 5 editors, with a paired, reproducible protocol (fixed splits, seeds, logged runs)."],
  ]);
  s.addNotes("One problem, one gap, three measurable objectives. PSR = share of edits that a face recogniser no longer matches to the person.");
}
{
  const s = frame("Problem & Solution Design", "2. Proposed Methodology & Overall Workflow", "End-to-End Research Methodology / System Workflow Architecture");
  // diagram frame
  const fx = MX + 0.2, fy = 1.2, fw = W - 2 * MX - 0.4, fh = 3.95;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "workflow frame" });
  const lane = (y, label) => s.addText(label, { x: fx + 0.15, y, w: 2.3, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: MAROON, margin: 0, isTextBox: true, objectName: label });
  lane(fy + 0.12, "HALF A – Protection (offline)");
  lane(fy + 2.07, "HALF B – Attack & score (held-out)");
  const bh = 1.15, by1 = fy + 0.5, by2 = fy + 2.45, gap = 0.32;
  const n = 5, bw = (fw - 0.3 - gap * (n - 1)) / n;
  const xs = [...Array(n)].map((_, i) => fx + 0.15 + i * (bw + gap));
  const A = ["Face photo x\n512×512 (CelebA-HQ)",
    "Sample cleanup P\nJPEG · resize · VAE · one-step DiffPure-style",
    "Loss on P(x+δ)\nlatent → target, denoiser, CLIP, SSIM",
    "Sign step, clip ± 4/255, round to 8-bit\n(40 steps, K = 4)",
    "Protected photo\n(the one posted)"];
  const B = ["Protected photo",
    "Held-out cleanup\n13 types, never trained on",
    "Editor (5)\nSD1.5, SDXL, InstructPix2Pix, inpainting, FaceID",
    "ArcFace similarity\nedit vs source person",
    "PSR + paired 95% CI\nvs clean, noise, v1"];
  A.forEach((t, i) => box(s, "A" + i, xs[i], by1, bw, bh, t, { pt: 11, fill: i === 4 ? MAROON : TINT, color: i === 4 ? WHITE : INK, bold: i === 4 }));
  B.forEach((t, i) => box(s, "B" + i, xs[i], by2, bw, bh, t, { pt: 11, fill: i === 4 ? MAROON : TINT, color: i === 4 ? WHITE : INK, bold: i === 4 }));
  for (let i = 0; i < n - 1; i++) { arrow(s, xs[i] + bw, by1 + bh / 2, xs[i + 1], by1 + bh / 2, "a" + i); arrow(s, xs[i] + bw, by2 + bh / 2, xs[i + 1], by2 + bh / 2, "b" + i); }
  // loop-back label and A4 -> B0 link
  s.addText("repeat ×40", { x: xs[1], y: by1 + bh + 0.02, w: xs[3] + bw - xs[1], h: 0.26, fontFace: FONT, fontSize: 11, italic: true, color: MAROON, align: "center", margin: 0, isTextBox: true, objectName: "loop label" });
  arrow(s, xs[4] + bw / 2, by1 + bh, xs[0] + bw / 2, by2, "publish");
  block(s, "workflow", MX, 5.3, W - 2 * MX, 1.5, "Workflow Summary",
    "Half A finds a 4/255 perturbation with 40 PGD steps, scoring every step on a randomly cleaned copy so the noise must survive cleanup, and penalising visibility on the photo that is actually posted. Half B attacks that photo with cleanups and editors Half A never saw, scores each edit’s identity with ArcFace, and reports PSR with paired bootstrap intervals over photos.");
  s.addNotes("Two halves that never share components: if the test used the training cleanups, it would prove nothing.");
}
{
  const s = frame("Problem & Solution Design", "3. Methodological Components & Formulation", "Sub-Module Breakdown, Governing Mathematical Model & Datasets");
  const c1 = MX, w1 = 6.35, c2 = MX + w1 + 0.25, w2 = W - MX - c2;
  block(s, "breakdown", c1, 1.2, w1, 2.75, "Methodological Breakdown", [
    ["Cleanup mixture (EOT [10])", "every step samples a differentiable cleanup P: JPEG, down-up resize, SD VAE round trip or a one-step DiffPure-style SD purifier."],
    ["Target-steered latent loss", "pulls the SD1.5 VAE latent of P(x+δ) toward a fixed destructive target (zero latent); optional UNet denoiser and CLIP terms."],
    ["Visibility control", "SSIM penalty on the published photo; sign-gradient step, clip to ±ε, round to 8-bit."],
  ]);
  block(s, "formulation", c1, 4.1, w1, 2.7, "Mathematical / Algorithmic Formulation", [
    ["Governing Formulation", `max over ‖δ‖∞ ≤ ε of  E_P [ α·L_vis + η·L_den ${M} β·L_sem ${M} γ·L_str ${M} w·‖E(P(x+δ)) ${M} z_t‖² ]`],
    ["Key Variables", "x photo, δ perturbation, ε = 4/255, E = SD1.5 VAE encoder, z_t = 0 target, P drawn from the cleanup mixture; 40 steps of 1/255, K = 4 draws per step; final weights α 0.55, β 0.90, γ 60.5, w 1."],
  ]);
  block(s, "dataset", c2, 1.2, w2, 3.3, "Dataset & Input Specifications", [
    ["Source & Size", "CelebA-HQ [16] portraits, 512×512 RGB; CelebAMask-HQ [17] masks for inpainting."],
    ["Data Partitioning", "fixed lists: test 100 (incl. v1’s 70), validation 20, train 20, messy 20, adaptive pool 120; re-tuning 28 + 12 photos chosen by MMD-critic [11] from 28,974."],
    ["Preprocessing", "512 px PNG; per-image seeds from a hash; padding before face detection."],
  ]);
  block(s, "metrics", c2, 4.65, w2, 2.15, "Evaluation Metrics", [
    ["PSR", "share of edits with no face or ArcFace [7] similarity to the source below 0.1005 (95th percentile of different-person pairs)."],
    ["PSNR / SSIM, paired CI", "invisibility checked on every photo; paired bootstrap 95% CI over photos."],
  ]);
  s.addNotes("The target term is the main new idea: L_vis alone pushes the latent away with no direction.");
}

// =========================================================================================
// SECTION 2: IMPLEMENTATION DETAILS
// =========================================================================================
pres.addSection({ title: "Implementation Details" });
{
  const s = frame("Implementation Details", "4. Implementation Details – Overview", "Module-Wise Implementation & Pipeline Integration Status");
  booktab(s, "modules", MX + 0.2, 1.25, [3.4, 6.85, 2.0],
    ["Module / Stage", "Implemented Functionality", "Status"],
    [["Module 1: Data / Input Pipeline", "CelebA-HQ ingestion, fixed splits, MMD-critic tuning selection (reproducible), Kaggle code bundles", "100% (Integrated)"],
     ["Module 2: Baselines / Reference", "v1 reproduced within tolerance, random-noise controls, PhotoGuard and Mist re-implemented at 4/255", "100% (Integrated)"],
     ["Module 3: Core Proposed Method", "PGD loop with cleanup mixture, target-steered term, denoiser term, one-step DiffPure-style cleaner; Optuna [12] re-tuning", "100% (Integrated)"],
     ["Module 4: Evaluation & Prototype", "edit runner (5 editors, 13 cleanups, 3 seeds), ArcFace + FaceNet scoring, paired bootstrap analysis, demo script", "100% (Functional)"]],
    { pt: 12, rowH: 0.72, alignCols: [null, "left", "center"] });
  block(s, "env", MX, 4.65, W - 2 * MX, 2.15, "Implementation Environment & Repository", [
    ["Software / Compute Environment", "Python, PyTorch, diffusers, transformers, insightface, facenet-pytorch, Optuna; Kaggle sessions on 2× NVIDIA T4 GPUs; one YAML config, logged seeds, library versions and git hash per run; 377 CPU unit tests."],
    ["Source Repository & Execution Readiness", "[Insert repository URL]. All results were produced from hash-stamped code bundles (final evaluation bundle 78ca757); every run resumes from checkpoints and is reproducible with one command per step."],
  ]);
  s.addNotes("All four modules are complete and integrated; about 40 Kaggle sessions were run.");
}
{
  const s = frame("Implementation Details", "5. Core Module Implementation Details", "Internal Module Mechanics, Parameter Configurations & Interfaces");
  const c1 = MX, w1 = 6.6, c2 = MX + w1 + 0.3, w2 = W - MX - c2;
  block(s, "modA", c1, 1.2, w1, 3.0, "Core Module A: Protection Loop (src/protect)", [
    ["Inputs & Outputs", "512 px PNG + method preset → protected 8-bit PNG, per-photo PSNR/SSIM/LPIPS and a loss trace (JSON)."],
    ["Implementation Details", "PGD with K = 4 cleanup draws per step; fp16 VAE and gradient checkpointing; photos sharded over 2 GPUs; resumable."],
    ["Key Parameters", "ε 4/255, step 1/255, 40 iterations; zero-latent target, w 1; α 0.55, β 0.90, γ 60.5."],
  ]);
  block(s, "modB", c1, 4.35, w1, 2.45, "Core Module B: Evaluation Harness (src/eval)", [
    ["Inputs & Outputs", "protected PNG folders → one CSV row per (photo, seed, editor, prompt, strength, cleanup, method) with ArcFace similarity and a no-face flag."],
    ["Integration Interface", "jobs declared in kaggle/jobs.yaml and run by run_job.py; results analysed by paired_compare.py and make_tables.py."],
  ]);
  // module interaction diagram
  const fx = c2, fy = 1.2, fw = w2, fh = 5.6;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "pipeline frame" });
  const steps = ["config.yaml + fixed photo lists", "run_protect.py\nPGD on 2× T4, sharded, resumable", "protected PNGs + metrics.csv",
    "evaluate.py\ncleanup → editor → ArcFace", "edit-level CSV (one row per edit)", "paired_compare.py / make_tables.py\nPSR, paired 95% CI → report"];
  const bw = fw - 0.6, bh = 0.62, g = (fh - 0.3 - steps.length * bh) / (steps.length - 1);
  steps.forEach((t, i) => {
    const y = fy + 0.15 + i * (bh + g);
    const dark = i === 1 || i === 3;
    box(s, "p" + i, fx + 0.3, y, bw, bh, t, { pt: 11, fill: dark ? MAROON : TINT, color: dark ? WHITE : INK, bold: dark });
    if (i < steps.length - 1) arrow(s, fx + fw / 2, y + bh, fx + fw / 2, y + bh + g, "pa" + i);
  });
  s.addNotes("Protection and evaluation only exchange files, so either side can be rerun alone.");
}

// =========================================================================================
// SECTION 3: VERIFICATION & DEMONSTRATION
// =========================================================================================
pres.addSection({ title: "Verification & Demonstration" });
{
  const s = frame("Verification & Demonstration", "6. Preliminary Verification & Quantitative Results", "Experimental Benchmarking & Comparative Evaluation");
  booktab(s, "results", MX + 0.6, 1.2, [5.6, 2.0, 2.0, 2.4],
    ["Method / Configuration (SD1.5 img2img, 100 test photos)", "PSR, no cleanup", "PSR, JPEG 75", "Photos failing invisibility"],
    [["Clean photo (no protection)", "13.9%", "10.3%", "–"],
     ["Baseline: DiffShield v1", "24.0%", "15.0%", "0 of 100"],
     ["Baseline: PhotoGuard [1] at 4/255", "81.0%", "18.3%", "100 of 100"],
     ["Baseline: Mist [2] at 4/255", "63.1%", "21.0%", "100 of 100"],
     ["Ablation: cleanups only, no target term", "24.3%", "18.3%", "0 of 100"],
     ["Variant: re-tuned with cleaner + denoiser", "51.2%", "22.7%", "0 of 100"],
     ["Proposed: v2 target-steered (mid)", "49.7%", "23.3%", "0 of 100"]],
    { pt: 12, rowH: 0.36, boldLast: true });
  block(s, "verify", MX, 4.35, W - 2 * MX, 2.45, "Experimental Verification Summary", [
    ["Quantitative Findings", `the target-steered method doubles v1 without cleanup (+25.7 points [+21.8, +29.8]) with no visible photo, and beats PhotoGuard under JPEG 75 (+5.0 [+1.3, +9.0]). Re-tuning with the cleaner and denoiser is on par (+1.6 [${M}0.9, +4.0]).`],
    ["Verification of Expected Behavior", `random noise at the same budget does nothing (+1.0 [${M}1.0, +3.1] over clean), and the same cleanups without the target term gain nothing, so the target term causes the gain. PhotoGuard and Mist protect more only by visibly altering every photo.`],
  ]);
  s.addNotes("PSR = share of edits no longer matched to the person. Intervals: paired bootstrap over the 100 photos.");
}
{
  const s = frame("Verification & Demonstration", "7. Experimental Visualizations & Prototype Demonstration", "Result Visualizations, Convergence / Performance Curves & Live Demo");
  const cw = (W - 2 * MX - 0.3) / 2, ch = 4.1, cy = 1.2;
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: cy, w: cw, h: ch, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "chart frame" });
  s.addChart(pres.charts.BAR, [{ name: "PSR (%)", labels: ["Clean", "Random noise", "v1", "Cleanups only", "v2 target (mid)", "Re-tuned", "Mist*", "PhotoGuard*"],
    values: [13.9, 14.9, 24.0, 24.3, 49.7, 51.2, 63.1, 81.0] }], {
    x: MX + 0.1, y: cy + 0.05, w: cw - 0.2, h: ch - 0.45, barDir: "col", chartColors: [MAROON],
    showTitle: true, title: "PSR (%), SD1.5 img2img, no cleanup, n = 100", titleFontFace: FONT, titleFontSize: 12, titleColor: INK,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: FONT, dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: "0.0",
    catAxisLabelFontFace: FONT, catAxisLabelFontSize: 10, catAxisLabelColor: GREY, valAxisLabelFontFace: FONT, valAxisLabelFontSize: 10, valAxisLabelColor: GREY,
    valAxisMaxVal: 100, valAxisMinVal: 0, valGridLine: { color: "E3E3E3", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, objectName: "psr chart" });
  s.addText("* visibly alter all 100 photos (fail PSNR/SSIM rule)", { x: MX + 0.15, y: cy + ch - 0.38, w: cw - 0.3, h: 0.3, fontFace: FONT, fontSize: 10, color: GREY, margin: 0, isTextBox: true, objectName: "chart note" });
  placeholderBox(s, "demo shot", MX + cw + 0.3, cy, cw, ch, "[Insert Working Prototype / Qualitative Output Snapshot]",
    "[Placeholder: panel.png from scripts/demo.py – source | protected | edit of original | edit of protected, with ArcFace similarity. Use only a photo you have consent to show; CelebA-HQ faces must not be redistributed.]");
  block(s, "demo", MX, 5.45, W - 2 * MX, 1.35, "Demonstration Overview",
    "python scripts/demo.py --image <photo> --method v2_tgt_zero_mid --out demo_out protects one photo on a GPU, edits the original and the protected copy with SD1.5 img2img using the same seed, and shows both edits with PSNR/SSIM and each edit’s ArcFace similarity to the source (lower = no longer the person).");
  s.addNotes("Pass --method v2_tgt_zero_mid: the script's default method is v2_final, which breaks the invisibility rule on 29 of 100 photos.");
}

// =========================================================================================
// SECTION 4: TECHNICAL KNOWLEDGE
// =========================================================================================
pres.addSection({ title: "Technical Knowledge" });
{
  const s = frame("Technical Knowledge", "8. Technical Justification, Design Trade-offs & Insights", "Methodological Rationale, Trade-offs & Boundary Conditions");
  const cw = (W - 2 * MX - 0.3) / 2, x2 = MX + cw + 0.3, h1 = 2.75, h2 = 2.75, y1 = 1.2, y2 = 4.05;
  block(s, "just", MX, y1, cw, h1, "1. Methodological Justification", [
    "Image-to-image editors start from the photo’s VAE latent, so steering it toward a fixed destructive target derails the edit; pushing it “away” alone has no direction and barely beat v1.",
    "Optimising over a random cleanup mixture assumes the attacker cleans first; PSR assumes that “a face recogniser sees a different person” approximates loss of identity.",
  ]);
  block(s, "trade", MX, y2, cw, h2, "2. Design & Parameter Trade-offs", [
    "Budget vs visibility (28 tuning photos): 6/255 adds +2.7 points with 1 photo visible; 8/255 gives the same gain with 12 of 28 visible, so we keep 4/255.",
    "40 PGD steps have converged (100 steps: +0.5 points); target weight “mid”, because “high” adds 3 points but makes 4 of 100 photos visible.",
  ]);
  block(s, "bound", x2, y1, cw, h1, "3. Boundary Conditions & Edge Cases", [
    "Invisibility is enforced on every photo, not on averages; JPEG and resizing cut the effect by more than half; it survives SD-based DiffPure-style cleanup and the VAE round trip.",
    "No protection for IP-Adapter FaceID, SDXL or inpainting; a pixel-space DDPM purifier and a small adaptive autoencoder remove it.",
  ]);
  block(s, "limits", x2, y2, cw, h2, "4. Limitations & Future Scope", [
    "A single-surrogate identity loss (FaceNet) did not transfer to ArcFace; no human study; second-recogniser check: [insert F1 result].",
    "Future: train against pixel-space purifiers and adaptive attackers; identity loss on an ensemble of face models incl. ArcFace; SDXL targets.",
  ]);
  s.addNotes("Every number here is measured; the budget curve is on the 28 tuning photos, the rest on the 100 test photos.");
}

// =========================================================================================
// REFERENCES & THANK YOU
// =========================================================================================
pres.addSection({ title: "References" });
const REFS = [
  "H. Salman, A. Khaddaj, G. Leclerc, A. Ilyas and A. Madry, “Raising the cost of malicious AI-powered image editing,” in Proc. ICML, 2023.",
  "C. Liang et al., “Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples,” in Proc. ICML, 2023.",
  "S. Shan, J. Cryan, E. Wenger, H. Zheng, R. Hanocka and B. Y. Zhao, “Glaze: Protecting artists from style mimicry by text-to-image models,” in Proc. USENIX Security, 2023.",
  "T. Van Le et al., “Anti-DreamBooth: Protecting users from personalized text-to-image synthesis,” in Proc. ICCV, 2023.",
  "W. Nie, B. Guo, Y. Huang, C. Xiao, A. Vahdat and A. Anandkumar, “Diffusion models for adversarial purification,” in Proc. ICML, 2022.",
  "R. Rombach, A. Blattmann, D. Lorenz, P. Esser and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proc. CVPR, 2022.",
  "J. Deng, J. Guo, N. Xue and S. Zafeiriou, “ArcFace: Additive angular margin loss for deep face recognition,” in Proc. CVPR, 2019.",
  "F. Schroff, D. Kalenichenko and J. Philbin, “FaceNet: A unified embedding for face recognition and clustering,” in Proc. CVPR, 2015.",
  "A. Madry, A. Makelov, L. Schmidt, D. Tsipras and A. Vladu, “Towards deep learning models resistant to adversarial attacks,” in Proc. ICLR, 2018.",
  "A. Athalye, L. Engstrom, A. Ilyas and K. Kwok, “Synthesizing robust adversarial examples,” in Proc. ICML, 2018.",
  "B. Kim, R. Khanna and O. Koyejo, “Examples are not enough, learn to criticize! Criticism for interpretability,” in Proc. NeurIPS, 2016.",
  "T. Akiba et al., “Optuna: A next-generation hyperparameter optimization framework,” in Proc. ACM KDD, 2019.",
  "T. Brooks, A. Holynski and A. A. Efros, “InstructPix2Pix: Learning to follow image editing instructions,” in Proc. CVPR, 2023.",
  "H. Ye, J. Zhang, S. Liu, X. Han and W. Yang, “IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models,” arXiv:2308.06721, 2023.",
  "D. Podell et al., “SDXL: Improving latent diffusion models for high-resolution image synthesis,” in Proc. ICLR, 2024.",
  "T. Karras, T. Aila, S. Laine and J. Lehtinen, “Progressive growing of GANs for improved quality, stability, and variation,” in Proc. ICLR, 2018.",
  "C.-H. Lee, Z. Liu, L. Wu and P. Luo, “MaskGAN: Towards diverse and interactive facial image manipulation,” in Proc. CVPR, 2020.",
];
function refSlide(title, from, to) {
  const s = frame("References", title, "");
  const items = [];
  for (let i = from; i < to; i++) {
    const last = i === to - 1;
    items.push({ text: `[${i + 1}]  `, options: { fontFace: FONT, fontSize: 12, bold: true, color: MAROON } });
    items.push({ text: REFS[i], options: { fontFace: FONT, fontSize: 12, color: INK, breakLine: !last, paraSpaceAfter: 6 } });
  }
  s.addText(items, { x: MX + 0.1, y: 1.25, w: W - 2 * MX - 0.2, h: 5.6, valign: "top", margin: 0, isTextBox: true, objectName: "reference list" });
  check(title, REFS.slice(from, to).map((r, k) => `[${from + k + 1}]  ` + r), 12, W - 2 * MX - 0.2, 5.6, 1.2, 6);
}
refSlide("References", 0, 9);
refSlide("References (cont.)", 9, REFS.length);

pres.addSection({ title: "Thank You" });
{
  const s = pres.addSlide({ masterName: "PLAIN", sectionTitle: "Thank You" });
  s.addText("Thank You", { x: 0.8, y: 2.55, w: W - 1.6, h: 1.1, fontFace: FONT, fontSize: 54, bold: true, color: MAROON, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "thank you" });
  s.addText("Questions & Viva Voce", { x: 0.8, y: 3.75, w: W - 1.6, h: 0.55, fontFace: FONT, fontSize: 22, bold: true, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "questions" });
}

pres.writeFile({ fileName: OUT }).then(() => {
  console.log("wrote", OUT);
  console.log(warn.length ? "FIT WARNINGS:\n" + warn.join("\n") : "fit check: no warnings");
});
