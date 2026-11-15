// DiffShield Phase II external review deck (final method = trial 14 / v2_next_tuned), Amrita beamer template look.
const pptxgen = require("pptxgenjs");
const OUT = process.argv[2] || "DiffShield_PhaseII_External_Review.pptx";

const MAROON = "A4123F", TINT = "F6E8EC", INK = "1F1F1F", GREY = "5F5F5F", WHITE = "FFFFFF", PH = "8A8A8A";
const PINK = "D9A3B4", GOLD = "C98A1E", SILVER = "9B9B9B";
const FONT = "Arial";
const W = 13.333, H = 7.5, MX = 0.45;
const M = "−";
const REPO = "https://github.com/Project-DiffShield/DiffShield";
const SITE = "https://diffshield-site.vercel.app/";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "DiffShield: Project Phase II External Review";
pres.author = "Team 15";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

// ---------- rough text-fit check (Arial: average glyph width about 0.5 em) ----------
const warn = [];
function lines(text, pt, wIn) {
  const perLine = Math.max(1, Math.floor((wIn * 72) / (0.5 * pt)));
  return String(text).split("\n").reduce((n, p) => n + Math.max(1, Math.ceil(p.length / perLine)), 0);
}
function check(id, paras, pt, wIn, hIn, spacing = 1.2, gap = 0) {
  const n = paras.reduce((s, p) => s + lines(p, pt, wIn), 0);
  const need = (n * pt * spacing) / 72 + (gap * paras.length) / 72;
  if (need > hIn) warn.push(`${id}: needs ${need.toFixed(2)}in, box ${hIn.toFixed(2)}in`);
}

// ---------- layouts ----------
pres.defineSlideMaster({ title: "TITLE_SLIDE", background: { color: WHITE } });
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
  rows.forEach((row, r) => row.forEach((t, i) => {
    if (typeof t === "string" && lines(t, pt, colW[i] - 0.17) > Math.floor((rowH * 72) / (pt * 1.2))) warn.push(`${id} r${r}c${i} wraps: ${t}`);
  }));
}
function box(s, id, x, y, w, h, text, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: o.fill || TINT }, line: { color: o.line || MAROON, width: 1 }, objectName: id });
  s.addText(text, { x: x + 0.06, y, w: w - 0.12, h, fontFace: FONT, fontSize: o.pt || 11, bold: !!o.bold, color: o.color || INK,
    align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: id + " label" });
  check(id, [text], o.pt || 11, w - 0.12, h);
}
function arrow(s, x1, y1, x2, y2, id) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color: MAROON, width: 1.5, endArrowType: "triangle" }, objectName: id });
}
const chartText = { titleFontFace: FONT, titleFontSize: 12, titleColor: INK,
  dataLabelFontFace: FONT, dataLabelFontSize: 10, dataLabelColor: INK, dataLabelFormatCode: "0.0",
  catAxisLabelFontFace: FONT, catAxisLabelFontSize: 10, catAxisLabelColor: GREY,
  valAxisLabelFontFace: FONT, valAxisLabelFontSize: 10, valAxisLabelColor: GREY };

// =========================================================================================
// TITLE SLIDE
// =========================================================================================
pres.addSection({ title: "Title" });
{
  const s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: "Title" });
  s.addText("Type 1: Research Project", { x: W - 3.05, y: 0.15, w: 2.85, h: 0.36, shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.04,
    fill: { color: MAROON }, fontFace: FONT, fontSize: 12, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "type badge" });
  s.addText("AMRITA VISHWA VIDYAPEETHAM", { x: 0.8, y: 0.5, w: W - 1.6, h: 0.45, fontFace: FONT, fontSize: 20, bold: true, color: MAROON,
    align: "center", valign: "middle", margin: 0, charSpacing: 2, isTextBox: true, objectName: "university" });
  s.addText("DiffShield: Protecting Face Photos from Diffusion-Model Editing", { x: 0.8, y: 1.15, w: W - 1.6, h: 0.6, fontFace: FONT, fontSize: 28,
    bold: true, color: MAROON, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "project title" });
  s.addText([
    { text: "23CSE498 – Project Phase II (External Review)", options: { bold: true } },
    { text: "     |     " },
    { text: "Team Number: 15", options: { bold: true } },
  ], { x: 0.8, y: 1.85, w: W - 1.6, h: 0.38, fontFace: FONT, fontSize: 14, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "course line" });
  const b = { type: "solid", pt: 0.75, color: INK };
  const c = (t, o = {}) => ({ text: t, options: { fontFace: FONT, fontSize: 13, color: INK, valign: "middle", border: [b, b, b, b], margin: [2, 8, 2, 8], ...o } });
  const team = [["CB.SC.U4CSE23437", "Pranav Kishan T Y"], ["CB.SC.U4CSE23519", "Venkatesh K"], ["CB.SC.U4CSE23536", "Narain B K"],
    ["CB.SC.U4CSE23539", "Karthikeyan P G"], ["CB.SC.U4CSE23556", "Varun Raj V"]];
  const rows = [[c("S.No", { bold: true, align: "center" }), c("Register Number", { bold: true, align: "center" }), c("Student Name", { bold: true })]];
  team.forEach((t, i) => rows.push([c(String(i + 1), { align: "center" }), c(t[0], { align: "center" }), c(t[1])]));
  s.addTable(rows, { x: (W - 7.6) / 2, y: 2.45, w: 7.6, colW: [1.0, 3.2, 3.4], rowH: 0.4, autoPage: false, objectName: "student table" });
  s.addText([
    { text: "Guide: " }, { text: "Dr. Shanmuga Priya S", options: { bold: true } }, { text: ", Assistant Professor, Department of CSE" },
  ], { x: 0.8, y: 5.05, w: W - 1.6, h: 0.38, fontFace: FONT, fontSize: 14, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "guide line" });
  s.addText("15 November 2026", { x: 0.8, y: 5.5, w: W - 1.6, h: 0.32, fontFace: FONT, fontSize: 12, color: GREY, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "date" });
  s.addNotes("DiffShield: a defensive tool that protects a person's own face photos from being edited into fakes.");
}

// =========================================================================================
// GUIDE APPROVAL (mandatory in the template)
// =========================================================================================
pres.addSection({ title: "Guide Approval" });
{
  const s = frame("Guide Approval", "Guide Review & Submission Approval", "Confirmation of Project Review and Approval from Guide");
  placeholderBox(s, "approval", MX + 0.2, 1.3, W - 2 * MX - 0.4, 5.5, "[Insert Screenshot of Guide Approval Email Here]",
    "[Mandatory: Attach a clear, readable screenshot of the email received from your project guide approving this presentation and final review submission]");
  s.addNotes("Mandatory slide in the department template: paste the guide's approval email screenshot here.");
}

// =========================================================================================
// SECTION 1: PROBLEM & SOLUTION DESIGN
// =========================================================================================
pres.addSection({ title: "Problem & Solution Design" });
{
  const s = frame("Problem & Solution Design", "1. Problem Definition & Research Motivation", "Problem Statement, Identified Literature Gap & Objectives");
  const bw = W - 2 * MX;
  block(s, "problem", MX, 1.2, bw, 1.5, "Problem Statement", [
    ["Core Research Problem", "Diffusion editors can turn any posted face photo into a convincing fake of that person. Can an invisible perturbation (at most 4/255 per pixel), added by the owner before posting, make such edits stop preserving the person’s identity?"],
    ["Domain Context", "Non-consensual deepfakes are cheap to make. A defence must run offline, stay invisible and survive the cleanup an attacker applies first (JPEG, resizing, diffusion denoising)."],
  ]);
  block(s, "gap", MX, 2.82, bw, 1.55, "Existing Limitations & Research Gap", [
    ["Summary of Existing Approaches", "PhotoGuard [1] and Mist [2] attack the Stable Diffusion encoder/UNet; Glaze [3] and Anti-DreamBooth [4] protect against style mimicry and fine-tuning. Most report image-quality metrics at budgets of 8–16/255."],
    ["Identified Research Gap", "Little evidence on whether protection removes a person’s identity (measured with a face recogniser), survives purification such as DiffPure [5], and stays invisible on every photo at a small budget."],
  ]);
  block(s, "objectives", MX, 4.49, bw, 2.31, "Research Objectives", [
    ["Objective 1 (Invisibility)", "the protected photo must look identical to the original: PSNR ≥ 38 dB and SSIM ≥ 0.95 on every one of the 100 test photos, with no pixel changed by more than 4/255."],
    ["Objective 2 (Protection)", "edits must stop matching the person, measured by the protection success rate (PSR): the share of edits the ArcFace face-recognition model no longer matches to the person. PSR must clearly exceed unprotected photos and random noise (paired 95% CIs excluding zero), compared with PhotoGuard and Mist."],
    ["Objective 3 (Robustness & generality)", "measure protection after common cleanups (JPEG, resizing, diffusion-based denoising) and across several editors, including some never used while making the protection, with a reproducible protocol."],
  ]);
  s.addNotes("PSR = share of edits that a face recogniser no longer matches to the person. Higher is better.");
}
{
  const s = frame("Problem & Solution Design", "2. Proposed Methodology & Overall Workflow", "End-to-End Research Methodology / System Workflow Architecture");
  const fx = MX + 0.2, fy = 1.2, fw = W - 2 * MX - 0.4, fh = 3.95;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "workflow frame" });
  s.addText("HALF A – Protection (offline)", { x: fx + 0.15, y: fy + 0.12, w: 4, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: MAROON, margin: 0, isTextBox: true, objectName: "lane A" });
  s.addText("HALF B – Attack & score (held-out)", { x: fx + fw - 4.15, y: fy + 2.07, w: 4, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: MAROON, align: "right", margin: 0, isTextBox: true, objectName: "lane B" });
  const bh = 1.15, by1 = fy + 0.5, by2 = fy + 2.45, gap = 0.32, n = 5;
  const bw = (fw - 0.3 - gap * (n - 1)) / n;
  const xs = [...Array(n)].map((_, i) => fx + 0.15 + i * (bw + gap));
  const A = ["Face photo x\n+ random start within ±4/255",
    "Apply 4 random cleanups\nJPEG · resize · VAE · one-step DiffPure-style",
    "Score cleaned copies\ntarget pull, push-away, denoiser, decoy, visibility",
    "Step 1/255 uphill, clip to ±4/255, round to 8-bit",
    "Protected photo\n(the one posted)"];
  const B = ["Protected photo",
    "Held-out cleanup\n(optional)",
    "Editor (5)\nSD1.5, SDXL, InstructPix2Pix, inpainting, FaceID",
    "ArcFace similarity\nedit vs source person",
    "PSR + paired 95% CI\nvs clean and baselines"];
  A.forEach((t, i) => box(s, "A" + i, xs[i], by1, bw, bh, t, { pt: 11, fill: i === 4 ? MAROON : TINT, color: i === 4 ? WHITE : INK, bold: i === 4 }));
  B.forEach((t, i) => box(s, "B" + i, xs[i], by2, bw, bh, t, { pt: 11, fill: i === 4 ? MAROON : TINT, color: i === 4 ? WHITE : INK, bold: i === 4 }));
  for (let i = 0; i < n - 1; i++) { arrow(s, xs[i] + bw, by1 + bh / 2, xs[i + 1], by1 + bh / 2, "a" + i); arrow(s, xs[i] + bw, by2 + bh / 2, xs[i + 1], by2 + bh / 2, "b" + i); }
  s.addText("repeat ×40", { x: xs[1], y: by1 + bh + 0.02, w: xs[3] + bw - xs[1], h: 0.26, fontFace: FONT, fontSize: 11, italic: true, color: MAROON, align: "center", margin: 0, isTextBox: true, objectName: "loop label" });
  arrow(s, xs[4] + bw / 2, by1 + bh, xs[0] + bw / 2, by2, "publish");
  block(s, "workflow", MX, 5.3, W - 2 * MX, 1.5, "Workflow Summary",
    "Half A refines a random 4/255 pattern for 40 steps; each step scores four randomly cleaned copies of the photo, so the noise must survive cleanup, while a visibility penalty keeps the posted photo unchanged. Half B attacks the posted photo with cleanups and editors, including ones Half A never used, scores each edit’s identity with ArcFace, and reports PSR with paired bootstrap intervals.");
  s.addNotes("Two halves that never share components, so robustness is not only measured on the training conditions.");
}
{
  const s = frame("Problem & Solution Design", "3. Methodological Components & Formulation", "Sub-Module Breakdown, Governing Mathematical Model & Datasets");
  const c1 = MX, w1 = 6.35, c2 = MX + w1 + 0.25, w2 = W - MX - c2;
  block(s, "breakdown", c1, 1.2, w1, 2.75, "Methodological Breakdown", [
    ["Cleanup mixture", "each step applies 4 random cleanups: JPEG q50–95 (24.5%), resize 50–90% (24.5%), SD VAE round trip (21%), one-step DiffPure-style purifier (30%)."],
    ["Target pull + push-away", "pull the SD1.5 VAE latent of the cleaned photo toward an all-zeros target, and away from the original’s latent."],
    ["Denoiser attack + visibility", "raise the UNet’s denoising error (timesteps 50–300); SSIM penalty on the published photo."],
  ]);
  block(s, "formulation", c1, 4.1, w1, 2.7, "Mathematical / Algorithmic Formulation", [
    ["Governing Formulation", "Score = target pull + α·push-away + η·denoiser error + β·decoy − γ·(1 − SSIM), averaged over 4 cleanups and maximised with every pixel within ±4/255."],
    ["Key Variables", "α 0.188, β 0.546, γ 21.6, η 0.012, target weight 1; 40 steps of 1/255 from a random start. Weights chosen by an Optuna search on 28 + 12 MMD-critic tuning photos with a per-photo invisibility rule."],
  ]);
  block(s, "dataset", c2, 1.2, w2, 3.3, "Dataset & Input Specifications", [
    ["Source & Size", "CelebA-HQ [16] portraits, 512×512 RGB; CelebAMask-HQ [17] masks for inpainting."],
    ["Data Partitioning", "100 fixed test photos for all results; 28 search + 12 confirmation tuning photos chosen by MMD-critic [11] from 28,974, never test photos."],
    ["Preprocessing", "512 px PNG; per-photo seeds from a hash; padding before face detection."],
  ]);
  block(s, "metrics", c2, 4.65, w2, 2.15, "Evaluation Metrics", [
    ["PSR", "share of edits with no face or ArcFace [7] similarity to the source below 0.1005 (95th percentile of different-person pairs)."],
    ["PSNR / SSIM, paired CI", "invisibility on every photo; paired bootstrap 95% CI over photos."],
  ]);
  s.addNotes("The target pull is the key ingredient; the denoiser weight is small because large weights hurt protection in the search.");
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
     ["Module 2: Baselines / Reference", "Unprotected and random-noise controls, our first prototype, PhotoGuard and Mist re-implemented at 4/255", "100% (Integrated)"],
     ["Module 3: Core Proposed Method", "Protection loop with cleanup mixture (incl. one-step DiffPure-style purifier), target pull, denoiser attack; Optuna weight search", "100% (Integrated)"],
     ["Module 4: Evaluation & Prototype", "Edit runner (5 editors, cleanups, 3 seeds), ArcFace scoring, paired bootstrap analysis, demo script", "100% (Functional)"]],
    { pt: 12, rowH: 0.72, alignCols: [null, "left", "center"] });
  const th = 0.36, bx = MX, by = 4.65, bw = W - 2 * MX, bh = 2.15;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: by, w: bw, h: bh, rectRadius: 0.06, fill: { color: TINT }, line: { color: TINT }, objectName: "env body" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y: by, w: bw, h: th, rectRadius: 0.06, fill: { color: MAROON }, line: { color: MAROON }, objectName: "env head" });
  s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by + th - 0.08, w: bw, h: 0.08, fill: { color: MAROON }, line: { color: MAROON }, objectName: "env head base" });
  s.addText("Implementation Environment & Repository", { x: bx + 0.15, y: by, w: bw - 0.3, h: th, fontFace: FONT, fontSize: 14, bold: true, color: WHITE, valign: "middle", margin: 0, isTextBox: true, objectName: "env title" });
  const bul = { bullet: { code: "25B8", indent: 14 }, paraSpaceAfter: 3 };
  s.addText([
    { text: "Software / Compute Environment: ", options: { ...bul, bold: true } },
    { text: "Python, PyTorch, diffusers, transformers, insightface (ArcFace), Optuna; Kaggle sessions on 2× NVIDIA T4 GPUs (274 GPU-seconds to protect one photo); one YAML config, logged seeds and versions; 377 unit tests.", options: { breakLine: true } },
    { text: "Source Repository & Execution Readiness: ", options: { ...bul, bold: true } },
    { text: REPO, options: { hyperlink: { url: REPO }, color: MAROON, underline: { style: "sng" } } },
    { text: " – every run is produced from a hash-stamped code bundle, resumes from checkpoints and is reproducible with one command per step." },
  ], { x: bx + 0.15, y: by + th + 0.06, w: bw - 0.3, h: bh - th - 0.12, fontFace: FONT, fontSize: 12, color: INK, valign: "top", margin: 0, isTextBox: true, objectName: "env text" });
  s.addNotes("All four modules are complete and integrated.");
}
{
  const s = frame("Implementation Details", "5. Core Module Implementation Details", "Internal Module Mechanics, Parameter Configurations & Interfaces");
  const c1 = MX, w1 = 6.6, c2 = MX + w1 + 0.3, w2 = W - MX - c2;
  block(s, "modA", c1, 1.2, w1, 3.0, "Core Module A: Protection Loop (src/protect)", [
    ["Inputs & Outputs", "512 px PNG + configuration → protected 8-bit PNG, per-photo PSNR/SSIM and a loss trace (JSON)."],
    ["Implementation Details", "4 cleanup draws per step, gradients averaged; half-precision VAE and gradient checkpointing; photos sharded over 2 GPUs; resumable."],
    ["Key Parameters", "budget 4/255, step 1/255, 40 steps; α 0.188, β 0.546, γ 21.6, η 0.012; one-step purifier in 30% of draws."],
  ]);
  block(s, "modB", c1, 4.35, w1, 2.45, "Core Module B: Evaluation Harness (src/eval)", [
    ["Inputs & Outputs", "protected PNG folders → one CSV row per (photo, seed, editor, prompt, strength, cleanup, method) with ArcFace similarity and a no-face flag."],
    ["Integration Interface", "jobs declared in kaggle/jobs.yaml and run by run_job.py; comparisons by paired_compare.py."],
  ]);
  const fx = c2, fy = 1.2, fw = w2, fh = 5.6;
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: fh, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "pipeline frame" });
  const steps = ["config.yaml + fixed photo lists", "run_protect.py\nPGD on 2× T4, sharded, resumable", "protected PNGs + metrics.csv",
    "evaluate.py\ncleanup → editor → ArcFace", "edit-level CSV (one row per edit)", "paired_compare.py\nPSR, paired 95% CI → report"];
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
  const s = frame("Verification & Demonstration", "6. Preliminary Verification – Methods Compared", "What Each Comparison Method Contains (all at the same 4/255 budget)");
  booktab(s, "compared", MX + 0.1, 1.2, [3.0, 9.1],
    ["Compared method", "What it contains and what it does"],
    [["Clean photo", "No protection; the original photo is edited directly. Reference for all other methods."],
     ["Uniform noise", "Random values within ±4/255 on every pixel, no optimisation: does any noise of this size help?"],
     ["First prototype", "Pushes the SD1.5 latent away from the original, with random JPEG/crop; no cleanup mixture, no target pull, no denoiser."],
     ["No-target ablation", "Same cleanup mixture and a denoiser term as DiffShield, but no target pull: isolates the target pull."],
     ["Hand-set weights", "Same design as DiffShield with hand-chosen weights; no DiffPure-style purifier, no denoiser term."],
     ["PhotoGuard [1] (re-impl.)", "1,000 steps pulling the SD latent to a zero target; no cleanups, no visibility penalty (full budget everywhere)."],
     ["Mist [2] (re-impl.)", "100 steps raising the diffusion training loss and pulling toward a fixed texture image; built for artworks."],
     ["DiffShield (final)", "Target pull + push-away + denoiser attack, trained against JPEG, resize, VAE and one-step DiffPure-style cleanups."]],
    { pt: 12, rowH: 0.6, boldLast: true, alignCols: [null, "left"] });
  s.addNotes("Comparisons are paired: same 100 test photos, prompts and seeds for every method.");
}
{
  const s = frame("Verification & Demonstration", "7. Preliminary Verification & Quantitative Results", "Experimental Benchmarking & Comparative Evaluation");
  booktab(s, "results", MX + 0.6, 1.15, [5.0, 2.2, 2.2, 2.6],
    ["Method (SD1.5 img2img, 100 test photos)", "PSR, no cleanup", "PSR, JPEG 75", "Photos failing invisibility"],
    [["Clean photo (no protection)", "13.9%", "10.3%", "–"],
     ["Uniform noise", "14.9%", "not run", "43 of 100"],
     ["First prototype", "24.0%", "15.0%", "0 of 100"],
     ["No-target ablation", "24.3%", "18.3%", "0 of 100"],
     ["Hand-set weights", "49.7%", "23.3%", "0 of 100"],
     ["PhotoGuard [1] (re-impl.)", "81.0%", "18.3%", "100 of 100"],
     ["Mist [2] (re-impl.)", "63.1%", "21.0%", "100 of 100"],
     ["DiffShield (final)", "51.2%", "22.7%", "0 of 100"]],
    { pt: 12, rowH: 0.34, boldLast: true });
  block(s, "verify", MX, 4.35, W - 2 * MX, 2.45, "Experimental Verification Summary", [
    ["Quantitative Findings", `DiffShield beats unprotected photos by +37.3 points [+32.7, +42.2] and the first prototype by +27.2 [+23.0, +31.7], with no visible photo; under JPEG 75 it is ahead of PhotoGuard (+4.3 [+0.7, +8.0]). It is on par with the hand-set weights (+1.6 [${M}0.9, +4.0]).`],
    ["Verification of Expected Behavior", "random noise does nothing (+1.0 over clean), and the no-target ablation stays at 24.3%, so the target pull causes the gain (+26.9 [+23.0, +30.9]). Only 1.8% of edits lose the face; similarity to the person falls from 0.222 to 0.103."],
  ]);
  s.addNotes("Intervals: paired bootstrap over the 100 photos. PhotoGuard and Mist score higher without cleanup only by visibly altering every photo.");
}
{
  const s = frame("Verification & Demonstration", "8. Experimental Visualizations & Prototype Demonstration", "Result Visualizations, Robustness to Cleanups & Live Demo");
  const cw = (W - 2 * MX - 0.3) / 2, ch = 4.1, cy = 1.2;
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: cy, w: cw, h: ch, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "chart1 frame" });
  s.addChart(pres.charts.BAR, [{ name: "PSR (%)", labels: ["Clean", "Noise", "Prototype", "No target", "Hand-set", "DiffShield", "Mist*", "PhotoGuard*"],
    values: [13.9, 14.9, 24.0, 24.3, 49.7, 51.2, 63.1, 81.0] }], {
    x: MX + 0.1, y: cy + 0.05, w: cw - 0.2, h: ch - 0.45, barDir: "col",
    chartColors: [SILVER, SILVER, SILVER, SILVER, PINK, MAROON, GOLD, GOLD],
    showTitle: true, title: "PSR (%), SD1.5 img2img, no cleanup, n = 100", ...chartText,
    showValue: true, dataLabelPosition: "outEnd", valAxisMaxVal: 100, valAxisMinVal: 0, valGridLine: { color: "E3E3E3", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false, objectName: "psr chart" });
  s.addText("* visibly alter all 100 photos (fail PSNR/SSIM rule)", { x: MX + 0.15, y: cy + ch - 0.38, w: cw - 0.3, h: 0.3, fontFace: FONT, fontSize: 10, color: GREY, margin: 0, isTextBox: true, objectName: "chart1 note" });
  const x2 = MX + cw + 0.3;
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: cy, w: cw, h: ch, fill: { color: WHITE }, line: { color: INK, width: 1 }, objectName: "chart2 frame" });
  s.addChart(pres.charts.BAR, [{ name: "Net protection (points)", labels: ["None", "JPEG 75", "JPEG 50", "Resize", "VAE", "DP 0.1", "DP 0.2", "DP 0.3", "Pixel t100", "Pixel t200"],
    values: [37.3, 12.3, 6.0, 4.0, 40.3, 41.7, 35.0, 37.0, -1.7, 1.0] }], {
    x: x2 + 0.1, y: cy + 0.05, w: cw - 0.2, h: ch - 0.45, barDir: "col", chartColors: [MAROON], invertedColors: [MAROON],
    showTitle: true, title: "DiffShield minus unprotected, PSR points, by cleanup", ...chartText,
    showValue: true, dataLabelPosition: "outEnd", valAxisMaxVal: 50, valAxisMinVal: -10, valGridLine: { color: "E3E3E3", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: false, objectName: "cleanup chart" });
  s.addText("DP = DiffPure-style (SD1.5); Pixel = pixel-space face DDPM purifier", { x: x2 + 0.15, y: cy + ch - 0.38, w: cw - 0.3, h: 0.3, fontFace: FONT, fontSize: 10, color: GREY, margin: 0, isTextBox: true, objectName: "chart2 note" });
  block(s, "demo", MX, 5.45, W - 2 * MX, 1.35, "Demonstration Overview",
    "python scripts/demo.py --image <photo> --method v2_next_tuned --out demo_out protects one photo on a GPU, edits the original and the protected copy with SD1.5 img2img using the same seed, and reports PSNR/SSIM and each edit’s ArcFace similarity to the person (lower = no longer the person).");
  s.addNotes("Protection survives DiffPure-style cleanup and the VAE round trip, shrinks under JPEG and resizing, and is removed by the pixel-space purifier.");
}

// =========================================================================================
// SECTION 4: TECHNICAL KNOWLEDGE
// =========================================================================================
pres.addSection({ title: "Technical Knowledge" });
{
  const s = frame("Technical Knowledge", "9. Technical Justification, Design Trade-offs & Insights", "Methodological Rationale, Trade-offs & Boundary Conditions");
  const cw = (W - 2 * MX - 0.3) / 2, x2 = MX + cw + 0.3, h1 = 2.75, y1 = 1.2, y2 = 4.05;
  block(s, "just", MX, y1, cw, h1, "1. Methodological Justification", [
    "Image-to-image editors start from the photo’s VAE latent; pulling it toward a fixed target gives one consistent direction and derails the edit (no-target ablation: 24.3% vs 51.2%).",
    "Scoring randomly cleaned copies makes the noise survive cleanup; the DiffPure-style purifier in training and the denoiser term target diffusion-based cleanup.",
  ]);
  block(s, "trade", MX, y2, cw, h1, "2. Design & Parameter Trade-offs", [
    "Budget (28 tuning photos): 6/255 lifts PSR from 18.9% to 21.3% with 1 photo visible; 8/255 gives the same with 12 visible, so 4/255 is kept.",
    "Large denoiser weights gave the worst search trials, so η is 0.012; 100 steps instead of 40 added only +0.5 points.",
  ]);
  block(s, "bound", x2, y1, cw, h1, "3. Boundary Conditions & Edge Cases", [
    "Invisibility is enforced on every photo, not on averages; edits with no face count as protected, so the no-face share (1.8%) is always reported.",
    "No protection for SDXL, inpainting or IP-Adapter FaceID; a pixel-space DDPM purifier removes it; JPEG and resizing reduce it.",
  ]);
  block(s, "limits", x2, y2, cw, h1, "4. Limitations & Future Scope", [
    "Adaptive attacker not run on the final method; one face recogniser (ArcFace) scores the results; no human study; CelebA-HQ only.",
    "Future: train against pixel-space purifiers and adaptive attackers; identity loss on several face models incl. ArcFace; SDXL targets.",
  ]);
  s.addNotes("Every number is measured: budget curve on the 28 tuning photos, everything else on the 100 test photos.");
}

// =========================================================================================
// DELIVERABLES
// =========================================================================================
pres.addSection({ title: "Deliverables" });
{
  const s = frame("Deliverables", "10. Project Deliverables & Technical Enrichment", "Repository, Certifications and Group TEA Deliverables");
  const thick = { type: "solid", pt: 1.5, color: INK }, thin = { type: "solid", pt: 0.75, color: INK }, none = { type: "none" };
  const base = (o = {}) => ({ fontFace: FONT, fontSize: 12, color: INK, valign: "middle", margin: [3, 6, 3, 6], ...o });
  const head = (t) => ({ text: t, options: base({ bold: true, border: [thick, none, thin, none] }) });
  const cell = (t, last, o = {}) => ({ text: t, options: base({ border: [none, none, last ? thick : none, none], ...o }) });
  const link = (u) => [{ text: u, options: { hyperlink: { url: u }, color: MAROON, underline: { style: "sng" } } }];
  const certs = [
    { text: "Narain B K: Computer Vision Fundamentals: Image Formation, Filtering and Machine Learning", options: { breakLine: true } },
    { text: "Varun Raj V: Deep Learning with TensorFlow and PyTorch", options: { breakLine: true } },
    { text: "Pranav Kishan T Y: CS207: Fundamentals of Machine Learning", options: { breakLine: true } },
    { text: "Karthikeyan P G: Master Stable Diffusion with Python: AI Images and Video", options: { breakLine: true } },
    { text: "Venkatesh K: Vision Language Models Bootcamp" }];
  const rows = [
    [head("Deliverable Category"), head("Details / Title / Link"), { text: "Status", options: base({ bold: true, align: "center", border: [thick, none, thin, none] }) }],
    [cell("Source Code Repository"), cell(link(REPO)), cell("Completed", false, { align: "center" })],
    [cell("Individual Online Certifications"), cell(certs), cell("Completed", false, { align: "center" })],
    [cell("Group TEA Deliverable 1"), cell([{ text: "Project website: " }, ...link(SITE)]), cell("Completed", false, { align: "center" })],
    [cell("Group TEA Deliverable 2"), cell("Research paper: DiffShield: A Tri-Modal Adversarial Framework for Robust Immunization Against Deepfake Generation"), cell("Completed", false, { align: "center" })],
    [cell("Group TEA Deliverable 3", true), cell([{ text: "Project repository: " }, ...link(REPO)], true), cell("Completed", true, { align: "center" })],
  ];
  s.addTable(rows, { x: MX + 0.2, y: 1.3, w: W - 2 * MX - 0.4, colW: [3.3, 7.25, 1.5], rowH: [0.42, 0.5, 1.7, 0.5, 0.75, 0.5], autoPage: false, objectName: "deliverables" });
  s.addNotes("All deliverables completed.");
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
