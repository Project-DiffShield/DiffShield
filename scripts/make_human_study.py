"""Build the E10 human-study materials: an offline HTML form (no server, no upload) + a SEPARATE answer key.

  python scripts/make_human_study.py --source_dir data/images/test --protected_dir results/<protect_run>/images \
      --edits_root results/<S06_output> --clean_root results/<S07_output>/.../cache/clean_edits --out results/human_study

Task A (2AFC, 20 pairs): "which of the two images has noise added?" One image of each pair is the clean photo, the other the
  protected (method, default v2_final) photo, both shown LOSSLESS (PNG) so nothing leaks through file format; sides randomised
  (balanced). 50% = the noise cannot be seen.
Task B (20 items): reference photo + one edited image; "is this the same person?" (4-point scale). Half of the items show an
  edit of the clean photo, half an edit of the protected photo (SD1.5 img2img, strength 0.55, one prompt/seed). With
  --versions 2 (default) the second form swaps the conditions, so across raters every photo is seen in both conditions.
  Give raters alternately study_v1.html and study_v2.html.

Outputs
  <out>/for_raters/study_v<k>.html   self-contained (images embedded), saves answers as a downloadable CSV; contains no key
  <out>/private/answer_key.csv       item -> image id, protected side, condition. DO NOT give this to raters.
  <out>/answers/                     put the rater CSVs here, then run scripts/analyze_human_study.py
  <out>/FACILITATOR_README.txt

Privacy: the photos are CelebA-HQ research images (not ours to redistribute). The forms are for showing raters locally on
your own machine; do not upload or publish them (results/human_study/for_raters and private are git-ignored).
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from src.eval import analysis as A  # noqa: E402

LABELS_B = [("same", "Same person"), ("probably_same", "Probably the same person"),
            ("probably_different", "Probably a different person"), ("different", "Different person")]


# ------------------------------------------------------------------------------------------ selection
def balanced_flags(n: int, rng: random.Random):
    """n booleans, exactly floor/ceil half True, shuffled."""
    flags = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(flags)
    return flags


def select_ids(ok_a, ok_b, n_a, n_b, seed):
    """Task B ids first (they need the saved edits), then Task A ids disjoint from them when enough candidates exist."""
    rng = random.Random(seed)
    pb = sorted(ok_b)
    rng.shuffle(pb)
    ids_b = pb[:n_b]
    pa = [i for i in sorted(ok_a) if i not in set(ids_b)]
    overlap = len(pa) < n_a
    if overlap:
        pa = sorted(ok_a)
    rng.shuffle(pa)
    return pa[:n_a], ids_b, overlap


# --------------------------------------------------------------------------------------------- images
def _data_uri(path: Path, fmt: str, quality: int = 92) -> str:
    im = Image.open(path)
    im.load()
    im = im.convert("RGB")
    buf = io.BytesIO()
    if fmt == "png":
        im.save(buf, format="PNG")
        mime = "image/png"
    else:
        im.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def locate(source_dir, protected_dir, edits_roots, clean_roots, method, editor, strength, prompt_id, seed, ids):
    """Per image id: paths of source, protected image, clean edit and protected edit (None if missing)."""
    out = {}
    edit_dirs = [Path(r) / "edits" for r in edits_roots]
    for iid in ids:
        out[iid] = {
            "source": A.find_with_ext([Path(source_dir)], iid, (".png", ".jpg", ".jpeg")),
            "protected": A.find_with_ext([Path(protected_dir)], iid, (".png", ".jpg", ".jpeg")),
            "clean_edit": A.find_with_ext([Path(r) for r in clean_roots], A.clean_rel_path(editor, prompt_id, strength, seed, iid)),
            "protected_edit": A.find_with_ext(edit_dirs, A.edit_rel_path(method, "none", editor, prompt_id, strength, seed, iid)),
        }
    return out


# ----------------------------------------------------------------------------------------------- build
def build_versions(paths: dict, n_a: int, n_b: int, versions: int, seed: int, method: str):
    """Returns (list of version dicts, key rows, warnings). A version = {'A': [...], 'B': [...], 'images': {key: uri}}."""
    warnings = []
    ok_a = [i for i, p in paths.items() if p["source"] and p["protected"]]
    ok_b = [i for i, p in paths.items() if p["source"] and p["clean_edit"] and p["protected_edit"]]
    ids_a, ids_b, overlap = select_ids(ok_a, ok_b, n_a, n_b, seed)
    if len(ids_a) < n_a:
        warnings.append(f"only {len(ids_a)} images have source+protected files for Task A (wanted {n_a})")
    if len(ids_b) < n_b:
        warnings.append(f"only {len(ids_b)} images have source + both edits for Task B (wanted {n_b})")
    if overlap and set(ids_a) & set(ids_b):
        warnings.append("Task A and Task B share some photos (not enough distinct candidates)")
    vers, key = [], []
    for v in range(1, versions + 1):
        rng = random.Random(seed * 100 + v)
        # --- Task A
        prot_right = balanced_flags(len(ids_a), rng)
        a_items, imgs = [], {}
        order = list(range(len(ids_a)))
        rng.shuffle(order)
        for n, k in enumerate(order, start=1):
            iid = ids_a[k]
            pr = prot_right[k]
            iid_item = f"A{n:02d}"
            left_path, right_path = (paths[iid]["source"], paths[iid]["protected"]) if pr else (paths[iid]["protected"], paths[iid]["source"])
            imgs[f"{iid_item}_L"], imgs[f"{iid_item}_R"] = _data_uri(left_path, "png"), _data_uri(right_path, "png")
            a_items.append({"id": iid_item, "L": f"{iid_item}_L", "R": f"{iid_item}_R"})
            key.append({"version": v, "task": "A", "item_id": iid_item, "image_id": iid, "protected_side": "R" if pr else "L",
                        "condition": f"protected={method}"})
        # --- Task B (condition flips between versions)
        flags = balanced_flags(len(ids_b), random.Random(seed * 100 + 7))       # same base assignment for all versions ...
        if v % 2 == 0:
            flags = [not f for f in flags]                                       # ... flipped for even versions
        orderb = list(range(len(ids_b)))
        rng.shuffle(orderb)
        b_items = []
        for n, k in enumerate(orderb, start=1):
            iid = ids_b[k]
            prot = flags[k]
            item = f"B{n:02d}"
            imgs[f"{item}_ref"] = _data_uri(paths[iid]["source"], "jpeg", 95)
            imgs[f"{item}_edit"] = _data_uri(paths[iid]["protected_edit" if prot else "clean_edit"], "jpeg", 92)
            b_items.append({"id": item, "ref": f"{item}_ref", "edit": f"{item}_edit"})
            key.append({"version": v, "task": "B", "item_id": item, "image_id": iid, "protected_side": "",
                        "condition": "protected_edit" if prot else "clean_edit"})
        vers.append({"version": v, "A": a_items, "B": b_items, "images": imgs})
    return vers, key, warnings


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo study</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f6f6f4;color:#1b1b1b}
 main{max-width:1180px;margin:0 auto;padding:16px}
 .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:12px 0}
 .imgs{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
 .imgs figure{margin:0;text-align:center}
 .imgs img{width:512px;max-width:100%;height:auto;image-rendering:auto;border:1px solid #bbb}
 button{font-size:16px;padding:10px 18px;border-radius:6px;border:1px solid #0072B2;background:#0072B2;color:#fff;cursor:pointer;margin:6px 6px 6px 0}
 button.sec{background:#fff;color:#0072B2} button:disabled{opacity:.4;cursor:default}
 label.opt{display:block;padding:8px 10px;border:1px solid #ccc;border-radius:6px;margin:6px 0;cursor:pointer}
 label.opt:hover{background:#eef5fb} .small{font-size:13px;color:#555} .warn{background:#fff4e5;border:1px solid #e0b060;padding:8px;border-radius:6px}
 input[type=text]{font-size:16px;padding:6px;width:220px}
 @media (prefers-color-scheme:dark){body{background:#181818;color:#eee}.card{background:#222;border-color:#444}.small{color:#aaa}label.opt{border-color:#555}label.opt:hover{background:#2a3a48}}
</style></head><body><main>
<div id="intro" class="card">
 <h1>Photo study</h1>
 <p>Thank you for helping. This takes about 10 minutes. You will do two short tasks and then save a small file with your answers.</p>
 <p class="warn"><b>Privacy:</b> the photos are from a face research dataset. Please view them only here, do not save, copy, screenshot or share them.
   Your answers contain only the code you type below (do not use your name).</p>
 <p><b>How to look:</b> use the photos at their normal size, as you would browse social media. Do not zoom in or use image editors.
   There are no trick questions; if you cannot tell, just guess.</p>
 <p>Your code (for example the number given to you): <input type="text" id="rater" maxlength="24" placeholder="e.g. R07">
   &nbsp; Screen: <select id="device"><option>laptop</option><option>desktop monitor</option><option>tablet</option><option>phone</option></select></p>
 <button id="start" disabled>Start Task A</button>
</div>
<div id="stage" class="card" style="display:none"></div>
<div id="done" class="card" style="display:none">
 <h2>All done, thank you!</h2>
 <p>Click the button to save your answers as a small CSV file, then hand that file to the person running the study.</p>
 <button id="dl">Save my answers (CSV)</button> <button class="sec" id="cp">Copy answers as text</button>
 <p class="small" id="dlmsg"></p>
</div>
</main>
<script>
"use strict";
const VERSION = __VERSION__;
const ITEMS = __ITEMS__;
const IMG = __IMAGES__;
const LABELS_B = __LABELS_B__;
let rater = "", phase = "intro", idx = 0, order = [], answers = [], t0 = 0;
const $ = id => document.getElementById(id);
function shuffle(a){ for(let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } return a; }
function save(){ try{ localStorage.setItem("study_v"+VERSION+"_"+rater, JSON.stringify(answers)); }catch(e){} }
$("rater").addEventListener("input", e => { $("start").disabled = e.target.value.trim().length < 2; });
$("start").onclick = () => { rater = $("rater").value.trim().replace(/[^A-Za-z0-9_-]/g,"_");
  answers.push({task:"meta", item:"device", response:$("device").value, order:0, ms:0});
  start("A"); };
function start(task){ phase = task; order = shuffle(ITEMS[task].map((_,i)=>i)); idx = 0; $("intro").style.display="none"; $("stage").style.display="block"; show(); }
function show(){
  const task = phase, n = order.length, it = ITEMS[task][order[idx]];
  let h = "<p class='small'>Task " + task + " &middot; item " + (idx+1) + " of " + n + "</p>";
  if(task==="A"){
    h += "<h2>Which of the two photos has noise added to it?</h2><p class='small'>One photo is the original; the other has a faint pattern of noise added. Choose the noisy one (guess if unsure).</p>";
    h += "<div class='imgs'><figure><img src='"+IMG[it.L]+"' alt='left photo'><figcaption>Left</figcaption></figure>";
    h += "<figure><img src='"+IMG[it.R]+"' alt='right photo'><figcaption>Right</figcaption></figure></div>";
    h += "<p><button data-r='L'>The LEFT photo has noise</button> <button data-r='R'>The RIGHT photo has noise</button></p>";
  } else {
    h += "<h2>Is the person in the edited image the same person as in the reference photo?</h2><p class='small'>The edit may change clothes, background or style. Judge the person's identity (face).</p>";
    h += "<div class='imgs'><figure><img src='"+IMG[it.ref]+"' alt='reference'><figcaption>Reference photo</figcaption></figure>";
    h += "<figure><img src='"+IMG[it.edit]+"' alt='edited'><figcaption>Edited image</figcaption></figure></div><div style='max-width:420px;margin:8px auto'>";
    LABELS_B.forEach(l => { h += "<label class='opt'><input type='radio' name='b' value='"+l[0]+"'> "+l[1]+"</label>"; });
    h += "<button id='nextb' disabled>Next</button></div>";
  }
  $("stage").innerHTML = h; t0 = performance.now(); window.scrollTo(0,0);
  if(task==="A") document.querySelectorAll("button[data-r]").forEach(b => b.onclick = () => answer(it, b.dataset.r));
  else { document.querySelectorAll("input[name=b]").forEach(r => r.onchange = () => { $("nextb").disabled = false; });
         $("nextb").onclick = () => answer(it, document.querySelector("input[name=b]:checked").value); }
}
function answer(it, resp){
  answers.push({task:phase, item:it.id, response:resp, order:idx+1, ms:Math.round(performance.now()-t0)}); save(); idx++;
  if(idx < order.length) show();
  else if(phase==="A"){ $("stage").innerHTML = "<h2>Task A finished</h2><p>Now a second, different task.</p><button id='goB'>Start Task B</button>"; $("goB").onclick = () => start("B"); }
  else { $("stage").style.display="none"; $("done").style.display="block"; }
}
function csv(){
  const ts = new Date().toISOString(); const rows = ["study_version,rater_id,task,item_id,response,order_index,response_ms,timestamp_utc"];
  answers.forEach(a => rows.push([VERSION, rater, a.task, a.item, a.response, a.order, a.ms, ts].join(",")));
  return rows.join("\r\n") + "\r\n";
}
$("dl").onclick = () => { const blob = new Blob([csv()], {type:"text/csv"}); const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = "diffshield_study_" + rater + "_v" + VERSION + ".csv"; document.body.appendChild(a); a.click(); a.remove();
  $("dlmsg").textContent = "Saved. If nothing downloaded, use 'Copy answers as text' and paste it into a text file."; };
$("cp").onclick = () => { navigator.clipboard.writeText(csv()).then(()=>{$("dlmsg").textContent="Copied.";}, ()=>{$("dlmsg").textContent="Copy failed: select the text manually.";}); };
</script></body></html>
"""


def render_html(ver: dict) -> str:
    items = {"A": ver["A"], "B": ver["B"]}
    return (HTML.replace("__VERSION__", str(ver["version"])).replace("__ITEMS__", json.dumps(items))
            .replace("__IMAGES__", json.dumps(ver["images"])).replace("__LABELS_B__", json.dumps(LABELS_B)))


FACILITATOR = """DiffShield human study (E10) - facilitator notes
==================================================
Files
  for_raters/study_v1.html, study_v2.html : give these to raters (open in any browser, works offline, nothing is uploaded).
      Hand out v1 and v2 alternately so every photo is seen in both conditions across raters.
  private/answer_key.csv                  : NEVER give to raters.
  answers/                                : collect each rater's downloaded CSV here (diffshield_study_<code>_v<k>.csv).

Procedure
  1. Each rater types a code (not their name), does Task A (20 pairs) and Task B (20 items), clicks 'Save my answers'.
  2. Put the CSVs into answers/ and run:  python scripts/analyze_human_study.py --study_dir results/human_study
  3. Report the actual number of raters. With fewer than 10 raters the result is a PILOT (the analysis flags it).
Instructions to say aloud: normal viewing size, no zoom, guess if unsure, no talking about the photos while others take part.
Privacy: the photos are CelebA-HQ research images. Show them locally only; do not upload, publish or post screenshots.
Task A: chance level is 50% (if the noise cannot be seen). Task B: share of 'same person' answers for edits of clean vs protected photos.
"""


def write_study(out: Path, vers, key, meta: dict):
    (out / "for_raters").mkdir(parents=True, exist_ok=True)
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "answers").mkdir(parents=True, exist_ok=True)
    for ver in vers:
        (out / "for_raters" / f"study_v{ver['version']}.html").write_text(render_html(ver), encoding="utf-8")
    with open(out / "private" / "answer_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["version", "task", "item_id", "image_id", "protected_side", "condition"])
        w.writeheader()
        w.writerows(key)
    (out / "private" / "study_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "FACILITATOR_README.txt").write_text(FACILITATOR, encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_dir", required=True, help="clean 512 px source images (PNG)")
    ap.add_argument("--protected_dir", required=True, help="protected PNGs of --method (the published images)")
    ap.add_argument("--edits_root", action="append", default=[], help="folder containing edits/<method>/none/<editor>/... (job output)")
    ap.add_argument("--clean_root", action="append", default=[], help="clean_edits folder (E0 cache)")
    ap.add_argument("--results", default=None, help="auto-discover edits/ and clean_edits/ under this folder")
    ap.add_argument("--ids", default=None, help="text file of image ids (default: every id found in --protected_dir)")
    ap.add_argument("--method", default="v2_final")
    ap.add_argument("--editor", default=A.HEAD_EDITOR)
    ap.add_argument("--strength", default=A.HEAD_STRENGTH)
    ap.add_argument("--prompt_id", default="p0")
    ap.add_argument("--edit_seed", type=int, default=1)
    ap.add_argument("--n_a", type=int, default=20)
    ap.add_argument("--n_b", type=int, default=20)
    ap.add_argument("--versions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(REPO / "results" / "human_study"))
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    edits_roots, clean_roots = list(a.edits_root), list(a.clean_root)
    if a.results:
        import os
        for dp, dn, _ in os.walk(a.results):
            if "edits" in dn:
                edits_roots.append(dp)
                dn.remove("edits")
            if "clean_edits" in dn:
                clean_roots.append(str(Path(dp) / "clean_edits"))
                dn.remove("clean_edits")
    if a.ids:
        ids = [Path(l.strip()).stem for l in Path(a.ids).read_text().splitlines() if l.strip()]
    else:
        ids = sorted({p.stem for p in Path(a.protected_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")})
    paths = locate(a.source_dir, a.protected_dir, edits_roots, clean_roots, a.method, a.editor, a.strength, a.prompt_id, a.edit_seed, ids)
    vers, key, warns = build_versions(paths, a.n_a, a.n_b, a.versions, a.seed, a.method)
    for w in warns:
        print("WARNING:", w)
    if not vers or not vers[0]["A"]:
        print("no usable Task A items (need source + protected files)", file=sys.stderr)
        return 2
    meta = {"method": a.method, "editor": a.editor, "strength": a.strength, "prompt_id": a.prompt_id, "edit_seed": a.edit_seed,
            "n_a": len(vers[0]["A"]), "n_b": len(vers[0]["B"]), "versions": a.versions, "seed": a.seed, "warnings": warns}
    out = Path(a.out)
    write_study(out, vers, key, meta)
    print(f"wrote {a.versions} forms to {out / 'for_raters'} and the answer key to {out / 'private'} (keep the key away from raters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
