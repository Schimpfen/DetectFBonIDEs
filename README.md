# FB580nm Cyan Pipeline

This README captures example commands for running the full pipeline. Replace
the placeholder paths with locations on your machine.

Assumptions:
- Run commands from the repo root (`DetectFBonIDEs`).
- Set `DATA_ROOT` to the folder that contains your input/output folders.

```powershell
$DATA_ROOT = "C:\\path\\to\\FB580nm\\cyan"
```

## 1) VSI -> OME-TIFF (with bright copies)

**IDES -> IDES_out (no-split, with bright copies)**
```powershell
python ".\vsi_to_ome_tiff.py" --backend bftools --no-split --indir "$DATA_ROOT\\IDES" --recursive --outdir "$DATA_ROOT\\IDES_out" --brighten --bright-factor 2.2 --bright-gamma 0.6
```

**MR -> MR_out (no-split, with bright copies)**
```powershell
python ".\vsi_to_ome_tiff.py" --backend bftools --no-split --indir "$DATA_ROOT\\MR" --recursive --outdir "$DATA_ROOT\\MR_out" --brighten --bright-factor 2.2 --bright-gamma 0.6
```

Notes:
- `--brighten` writes extra files with `_bright` in the name (original OME-TIFFs are preserved).
- bftools will auto-download on first run to `$env:USERPROFILE\.bftools`.

## 2) Crop bright images (with debug outputs)

Commands below are the tuned prompts you requested.

**IDES_out**
```powershell
py -3 ".\IDEScrop.py" `
  --root "$DATA_ROOT\\IDES_out" `
  --pattern "*.ome*bright*.tif*" `
  --energy_pct 75 --min_ratio 1.2 --mean_pct 30 `
  --close_radius 16 --min_cc_area 1500 `
  --tighten_pct 30 `
  --expand_cols_hi 40 --expand_cols_lo 2 `
  --expand_rows_hi 40 --expand_rows_lo 15 `
  --expand_close_cols 110 --expand_close_rows 24 `
  --expand_gap_max 260 --expand_smooth 41 `
  --expand_pad 45 `
  --refine_mean_hi 0 --refine_mean_lo 0 --refine_score_hi 0 --refine_score_lo 0 `
  --refine_gap_max 200 --refine_min_width_frac 0.0 --refine_close_radius 2 --refine_pad 0 `
  --debug
```


**MR_out**
```powershell
py -3 ".\IDEScrop.py" `
  --root "$DATA_ROOT\\MR_out" `
  --pattern "*.ome*bright*.tif*" `
  --energy_pct 75 --min_ratio 1.2 --mean_pct 25 `
  --close_radius 16 --min_cc_area 1500 `
  --tighten_pct 30 `
  --expand_cols_hi 40 --expand_cols_lo 2 `
  --expand_rows_hi 40 --expand_rows_lo 15 `
  --expand_close_cols 110 --expand_close_rows 24 `
  --expand_gap_max 260 --expand_smooth 41 `
  --expand_pad 45 `
  --refine_mean_hi 0 --refine_mean_lo 0 --refine_score_hi 0 --refine_score_lo 0 `
  --refine_gap_max 200 --refine_min_width_frac 0.0 --refine_close_radius 2 --refine_pad 0 `
  --debug
```

Outputs per file:
- `<basename>_crop.png`
- `<basename>_debug.png`
- `<basename>_crop_debug.png`

## 3) Detect FB particles (debug images only)

The detector uses peak-based LoG response with MAD thresholding and per-peak classification.
Default file pattern is `**/*_S0.ome_bright_debug.png`.
Use `--no-mask` for overlay-only output (or `--no-overlays --no-mask` for summary only).

**IDES_out**
```powershell
cd "$DATA_ROOT\\IDES_out"
python "..\\detectFBincrop_v02000.py" `
  --small-particle-mode `
  --mag-prior local `
  --mag-pct 75 `
  --v-min-pct 60 `
  --resp-k 4.0 `
  --snr-min 2.0 `
  --blob-k 0.3 `
  --elong-ratio-max 4.5 `
  --peak-window 7 `
  --nms-ksize 5 `
  --area-k 1.0 `
  --min-area 1 `
  --max-area 4000 `
  --median-ksize 3 `
  --open-ksize 3 `
  --no-mask

```

**MR_out**
```powershell
cd "$DATA_ROOT\\MR_out"
python "..\\detectFBincrop_v02000.py" `
  --small-particle-mode `
  --mag-prior local `
  --mag-pct 75 `
  --v-min-pct 60 `
  --resp-k 2.5 `
  --snr-min 2.5 `
  --blob-k 0.3 `
  --elong-ratio-max 3.5 `
  --peak-window 7 `
  --nms-ksize 5 `
  --area-k 1.0 `
  --min-area 1 `
  --max-area 4000 `
  --median-ksize 3 `
  --open-ksize 3 `
  --no-mask

```

Outputs per file:
- `summary_counts.csv`
