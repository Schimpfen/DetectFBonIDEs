#!/usr/bin/env python3
"""
IDEScrop_v3.py — Robust crop to the electrode stripe field (reject dark background slabs).

What it does
- Computes local |dI/dx| (vertical stripes => high x-gradient).
- Computes anisotropy ratio: mean(|dI/dx|) / (mean(|dI/dy|)+eps). Vertical stripes => ratio >> 1.
- Rejects dark/background regions using a local mean-intensity floor.
- Builds a binary mask from these criteria, cleans it (closing), finds connected components.
- Chooses the best component by "stripe quality" score (not by largest area).
- Optionally biases selection to the right side (useful for your shown failure mode).

Outputs
- <basename>_crop.png
- if --debug:
    <basename>_debug.png       (full image with crop rectangle)
    <basename>_crop_debug.png  (cropped image with red border)

Example (Windows)
  py -3 "C:\\...\\IDEScrop_v3.py" --root "C:\\...\\IDES_out" --pattern "*.ome*bright*.tif*" --prefer_right --debug
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image

try:
    from tifffile import TiffFile  # optional
except Exception:
    TiffFile = None


# ------------------ I/O ------------------
def load_color_from_ome(path: str) -> np.ndarray:
    """Return RGB uint8 image; if not RGB, pick highest-variance plane and replicate; percentile-scale to uint8."""
    ext = os.path.splitext(path)[1].lower()
    if TiffFile is not None and ext in [".tif", ".tiff", ".ome.tif", ".ome.tiff"]:
        with TiffFile(path) as tf:
            arr = tf.asarray()
    else:
        arr = np.asarray(Image.open(path))

    arr = np.asarray(arr)

    # Already RGB
    if arr.ndim >= 3 and arr.shape[-1] == 3:
        img = arr
    else:
        # Pick highest-variance plane
        H, W = arr.shape[-2], arr.shape[-1]
        flat = arr.reshape((-1, H, W))
        variances = flat.reshape(flat.shape[0], -1).var(axis=1)
        g = flat[int(np.argmax(variances))]
        img = np.stack([g, g, g], axis=-1)

    # Scale to uint8 if needed
    if img.dtype != np.uint8:
        img = img.astype(np.float64)
        out = np.zeros_like(img, dtype=np.uint8)
        for c in range(img.shape[-1]):
            lo, hi = np.percentile(img[..., c], [1, 99.8])
            if hi <= lo:
                lo, hi = img[..., c].min(), img[..., c].max()
                if hi <= lo:
                    hi = lo + 1.0
            ch = np.clip((img[..., c] - lo) / (hi - lo), 0.0, 1.0)
            out[..., c] = (ch * 255.0).astype(np.uint8)
        img = out

    return img


def rgb_to_gray_f32(rgb: np.ndarray) -> np.ndarray:
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    return g.astype(np.float32)


def smooth1d(v: np.ndarray, win: int) -> np.ndarray:
    win = int(win)
    if win <= 1:
        return v.astype(np.float32)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(v.astype(np.float32), k, mode="same")


def binary_close_1d(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask
    win = 2 * radius + 1
    m = mask.astype(np.int8)
    dil = np.convolve(m, np.ones(win, dtype=np.int8), mode="same") > 0
    ero = np.convolve(dil.astype(np.int8), np.ones(win, dtype=np.int8), mode="same") == win
    return ero


def longest_true_run(mask: np.ndarray):
    if mask.size == 0 or not mask.any():
        return None
    m = mask.astype(np.int8)
    dm = np.diff(np.r_[0, m, 0])
    starts = np.flatnonzero(dm == 1)
    ends = np.flatnonzero(dm == -1)
    lengths = ends - starts
    i = int(np.argmax(lengths))
    return int(starts[i]), int(ends[i])


# ------------------ drawing / bbox ------------------
def clamp_bbox(x0, y0, x1, y1, W, H):
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(1, min(x1, W))
    y1 = max(1, min(y1, H))
    if x1 <= x0 + 1:
        x0, x1 = 0, W
    if y1 <= y0 + 1:
        y0, y1 = 0, H
    return x0, y0, x1, y1


def draw_rect_rgb(rgb: np.ndarray, x0, y0, x1, y1):
    dbg = rgb.copy()
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    x0 = max(0, min(x0, dbg.shape[1] - 1))
    x1 = max(1, min(x1, dbg.shape[1]))
    y0 = max(0, min(y0, dbg.shape[0] - 1))
    y1 = max(1, min(y1, dbg.shape[0]))
    t = 2
    dbg[y0:y0 + t, x0:x1] = [255, 0, 0]
    dbg[y1 - t:y1, x0:x1] = [255, 0, 0]
    dbg[y0:y1, x0:x0 + t] = [255, 0, 0]
    dbg[y0:y1, x1 - t:x1] = [255, 0, 0]
    return dbg


def apply_margins(x0, y0, x1, y1, W, H, left, right, top, bottom):
    x0 = x0 - int(left)
    x1 = x1 + int(right)
    y0 = y0 - int(top)
    y1 = y1 + int(bottom)
    return clamp_bbox(x0, y0, x1, y1, W, H)


# ------------------ fast box mean (integral image) ------------------
def box_mean_2d(img_f32: np.ndarray, win_y: int, win_x: int) -> np.ndarray:
    """
    Fast 2D box mean using integral image (correct, bounds-safe).
    Window sizes are forced odd. Padding is edge.
    """
    img = img_f32.astype(np.float32, copy=False)
    H, W = img.shape
    win_y = int(win_y); win_x = int(win_x)

    if win_y <= 1 and win_x <= 1:
        return img.astype(np.float32, copy=True)

    if win_y % 2 == 0:
        win_y += 1
    if win_x % 2 == 0:
        win_x += 1

    ry = win_y // 2
    rx = win_x // 2

    pad = np.pad(img, ((ry, ry), (rx, rx)), mode="edge")
    Hp, Wp = pad.shape  # Hp = H + 2*ry, Wp = W + 2*rx

    # Integral image with leading zeros: shape (Hp+1, Wp+1)
    ii = np.zeros((Hp + 1, Wp + 1), dtype=np.float32)
    ii[1:, 1:] = pad.cumsum(axis=0).cumsum(axis=1)

    # For each output pixel (y,x) in original image, the window in padded coords is:
    # y..y+win_y-1, x..x+win_x-1
    y0 = np.arange(0, H, dtype=np.int32)
    x0 = np.arange(0, W, dtype=np.int32)
    y1 = y0 + win_y
    x1 = x0 + win_x

    Y0 = y0[:, None]
    Y1 = y1[:, None]
    X0 = x0[None, :]
    X1 = x1[None, :]

    # Use ii with +1 offset already baked in
    s = ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]
    return (s / float(win_y * win_x)).astype(np.float32)


# ------------------ binary morphology (box-based) ------------------
def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask
    win = 2 * radius + 1
    m = mask.astype(np.float32)
    # window sum > 0 -> any True
    sm = box_mean_2d(m, win, win) * float(win * win)
    return sm > 0.0


def binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask
    win = 2 * radius + 1
    m = mask.astype(np.float32)
    sm = box_mean_2d(m, win, win) * float(win * win)
    return sm >= (float(win * win) - 1e-6)


def binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    return binary_erode(binary_dilate(mask, radius), radius)


# ------------------ connected components (4-neighborhood) ------------------
def connected_components_all_bboxes(mask: np.ndarray, min_area: int):
    """Return list of (x0,y0,x1,y1, area, cx, cy) for all CCs with area>=min_area."""
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out = []

    min_area = int(max(1, min_area))

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True

            area = 0
            xmin = xmax = x
            ymin = ymax = y
            sumx = 0.0
            sumy = 0.0

            while stack:
                yy, xx = stack.pop()
                area += 1
                xmin = min(xmin, xx); xmax = max(xmax, xx)
                ymin = min(ymin, yy); ymax = max(ymax, yy)
                sumx += xx; sumy += yy

                if yy > 0 and mask[yy - 1, xx] and not visited[yy - 1, xx]:
                    visited[yy - 1, xx] = True; stack.append((yy - 1, xx))
                if yy + 1 < H and mask[yy + 1, xx] and not visited[yy + 1, xx]:
                    visited[yy + 1, xx] = True; stack.append((yy + 1, xx))
                if xx > 0 and mask[yy, xx - 1] and not visited[yy, xx - 1]:
                    visited[yy, xx - 1] = True; stack.append((yy, xx - 1))
                if xx + 1 < W and mask[yy, xx + 1] and not visited[yy, xx + 1]:
                    visited[yy, xx + 1] = True; stack.append((yy, xx + 1))

            if area >= min_area:
                cx = sumx / float(area)
                cy = sumy / float(area)
                out.append((xmin, ymin, xmax + 1, ymax + 1, area, cx, cy))

    return out


def expand_bbox_by_projections(x0, y0, x1, y1, gx_m, ratio, mean_m, args):
    """
    Robust expansion using hysteresis on row/col stripe score.

    - score = gx_m * ratio, gated by brightness (mean_m)
    - build HIGH mask (seed) and LOW mask (expand)
    - choose LOW-mask run that intersects the seed range (not just contains center)
    - merge small gaps up to expand_gap_max
    """
    H, W = gx_m.shape
    eps = 1e-6

    # Gate out dark regions
    mean_thr = float(np.percentile(mean_m, args.mean_pct))
    score = (gx_m * (ratio + 0.0)).astype(np.float32)
    score = np.where(mean_m >= mean_thr, score, 0.0).astype(np.float32)

    # 1D projections
    col = score.mean(axis=0)
    row = score.mean(axis=1)

    # Smooth
    col_s = smooth1d(col, int(args.expand_smooth))
    row_s = smooth1d(row, int(args.expand_smooth))

    # Hysteresis thresholds
    col_hi = float(np.percentile(col_s, args.expand_cols_hi))
    col_lo = float(np.percentile(col_s, args.expand_cols_lo))
    row_hi = float(np.percentile(row_s, args.expand_rows_hi))
    row_lo = float(np.percentile(row_s, args.expand_rows_lo))

    col_high = col_s >= col_hi
    col_low = col_s >= col_lo
    row_high = row_s >= row_hi
    row_low = row_s >= row_lo

    # Closing (separate radii strongly recommended)
    col_low = binary_close_1d(col_low, int(getattr(args, "expand_close_cols", 30)))
    row_low = binary_close_1d(row_low, int(getattr(args, "expand_close_rows", 10)))

    # Merge small gaps in LOW masks (bridges tiny holes but not huge dark slabs)
    def merge_small_gaps(mask, max_gap):
        m = mask.astype(np.int8)
        dm = np.diff(np.r_[0, m, 0])
        starts = np.flatnonzero(dm == 1)
        ends = np.flatnonzero(dm == -1)
        if starts.size <= 1:
            return mask
        out = mask.copy()
        for i in range(len(starts) - 1):
            gap = starts[i + 1] - ends[i]
            if 0 < gap <= int(max_gap):
                out[ends[i]:starts[i + 1]] = True
        return out

    col_low = merge_small_gaps(col_low, args.expand_gap_max)
    row_low = merge_small_gaps(row_low, args.expand_gap_max)

    rx = grow_from_seed(col_low, int(x0), int(x1), int(args.expand_gap_max))
    ry = grow_from_seed(row_low, int(y0), int(y1), int(args.expand_gap_max))

    if rx is not None:
        x0, x1 = rx
    if ry is not None:
        y0, y1 = ry

    # Final pad
    pad = int(args.expand_pad)
    x0 = max(0, int(x0) - pad); x1 = min(W, int(x1) + pad)
    y0 = max(0, int(y0) - pad); y1 = min(H, int(y1) + pad)

    return int(x0), int(y0), int(x1), int(y1)


def grow_from_seed(low_mask: np.ndarray, seed_a: int, seed_b: int, gap_max: int):
    """
    Expand [seed_a, seed_b) inside low_mask by growing left/right.
    Allows crossing False gaps up to gap_max (pixels). Stops when a gap exceeds gap_max.
    """
    n = int(low_mask.size)
    if n == 0 or not low_mask.any():
        return None

    seed_a = int(max(0, min(seed_a, n)))
    seed_b = int(max(0, min(seed_b, n)))
    if seed_b <= seed_a:
        return None

    if not low_mask[seed_a:seed_b].any():
        idx = np.flatnonzero(low_mask)
        if idx.size == 0:
            return None
        c = int(0.5 * (seed_a + seed_b))
        j = int(idx[np.argmin(np.abs(idx - c))])
        seed_a = j
        seed_b = j + 1

    a = seed_a
    gap = 0
    i = a - 1
    while i >= 0:
        if low_mask[i]:
            a = i
            gap = 0
        else:
            gap += 1
            if gap > int(gap_max):
                break
        i -= 1

    b = seed_b
    gap = 0
    i = b
    while i < n:
        if low_mask[i]:
            b = i + 1
            gap = 0
        else:
            gap += 1
            if gap > int(gap_max):
                break
        i += 1

    return int(a), int(b)


def refine_bbox_edges(x0, y0, x1, y1, gx_m, ratio, mean_m, args):
    """
    Gentle edge trim inside current bbox (won't collapse to a thin strip).
    Uses hysteresis + grow-from-seed on per-column scores.

    - col_mean: mean intensity per column
    - col_score: mean(gx_m * ratio) per column
    - keep_lo: permissive mask
    - keep_hi: seed mask
    Grow keep_lo from the seed region (center of bbox) allowing small gaps.
    If refined width shrinks too much, keep original bbox.
    """
    H, W = gx_m.shape
    x0 = int(max(0, min(x0, W-1))); x1 = int(max(x0+1, min(x1, W)))
    y0 = int(max(0, min(y0, H-1))); y1 = int(max(y0+1, min(y1, H)))

    sub_g = gx_m[y0:y1, x0:x1].astype(np.float32)
    sub_r = ratio[y0:y1, x0:x1].astype(np.float32)
    sub_m = mean_m[y0:y1, x0:x1].astype(np.float32)

    col_score = (sub_g * sub_r).mean(axis=0)
    col_mean = sub_m.mean(axis=0)

    # Hysteresis thresholds inside bbox (lo/hi)
    m_hi = float(np.percentile(col_mean, args.refine_mean_hi))
    m_lo = float(np.percentile(col_mean, args.refine_mean_lo))
    s_hi = float(np.percentile(col_score, args.refine_score_hi))
    s_lo = float(np.percentile(col_score, args.refine_score_lo))

    keep_lo = (col_mean >= m_lo) & (col_score >= s_lo)
    keep_hi = (col_mean >= m_hi) & (col_score >= s_hi)

    keep_lo = binary_close_1d(keep_lo, int(args.refine_close_radius))

    # Seed region = middle 30% of bbox, but restricted to keep_hi if possible
    n = keep_lo.size
    sa = int(0.35 * n); sb = int(0.65 * n)
    seed = keep_hi.copy()
    if seed[sa:sb].any():
        tmp = np.zeros_like(seed)
        tmp[sa:sb] = seed[sa:sb]
        seed = tmp
    elif not seed.any():
        seed = np.zeros_like(keep_lo)
        seed[sa:sb] = keep_lo[sa:sb]

    seed_idx = np.flatnonzero(seed)
    if seed_idx.size == 0:
        return x0, y0, x1, y1

    seed_a = int(seed_idx.min())
    seed_b = int(seed_idx.max()) + 1

    rx = grow_from_seed(keep_lo, seed_a, seed_b, int(args.refine_gap_max))
    if rx is None:
        return x0, y0, x1, y1

    lx0, lx1 = rx
    new_x0 = x0 + lx0
    new_x1 = x0 + lx1

    orig_w = (x1 - x0)
    new_w = (new_x1 - new_x0)
    if new_w < int(args.refine_min_width_frac * orig_w):
        return x0, y0, x1, y1

    pad = int(args.refine_pad)
    new_x0 = max(0, new_x0 - pad)
    new_x1 = min(W, new_x1 + pad)
    return int(new_x0), int(y0), int(new_x1), int(y1)


# ------------------ detection ------------------
def detect_stripe_field_bbox(gray_f32: np.ndarray, args):
    """
    Detect stripe field bbox using:
      - local x-gradient strength (gx)
      - anisotropy ratio gx/(gy+eps)
      - local mean intensity floor to reject dark background slabs

    Select best CC by quality score:
      score = mean(ratio over mask) * mean(gx_m over mask)
    """
    H, W = gray_f32.shape
    eps = 1e-6

    # Gradients (absolute)
    gx = np.abs(np.diff(gray_f32, axis=1))  # (H, W-1)
    gy = np.abs(np.diff(gray_f32, axis=0))  # (H-1, W)

    # Pad to (H, W)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge").astype(np.float32)
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge").astype(np.float32)

    # Local means of gradients and intensity
    gx_m = box_mean_2d(gx, args.energy_win_y, args.energy_win_x)
    gy_m = box_mean_2d(gy, args.energy_win_y, args.energy_win_x)
    mean_m = box_mean_2d(gray_f32.astype(np.float32), args.mean_win_y, args.mean_win_x)

    ratio = gx_m / (gy_m + eps)

    # Thresholds
    thr_e = float(np.percentile(gx_m, args.energy_pct))
    thr_mean = float(np.percentile(mean_m, args.mean_pct))

    mask = (gx_m >= thr_e) & (ratio >= float(args.min_ratio)) & (mean_m >= thr_mean)

    if args.close_radius > 0:
        mask = binary_close(mask, args.close_radius)

    comps = connected_components_all_bboxes(mask, min_area=args.min_cc_area)
    if not comps:
        # fallback: right-biased reasonable crop (better than full-frame)
        x0 = int(W * 0.35); x1 = int(W * 0.95)
        y0 = int(H * 0.10); y1 = int(H * 0.90)
        return x0, y0, x1, y1

    best = None  # (score, x0,y0,x1,y1)
    for (x0, y0, x1, y1, area, cx, cy) in comps:
        if args.prefer_right and cx < (W * float(args.prefer_right_xmin)):
            continue

        inner = float(args.seed_inner_frac)
        xL = int((1.0 - inner) * 0.5 * W)
        xR = int(W - xL)
        yT = int((1.0 - inner) * 0.5 * H)
        yB = int(H - yT)

        sx0 = max(x0, xL); sx1 = min(x1, xR)
        sy0 = max(y0, yT); sy1 = min(y1, yB)
        if sx1 <= sx0 or sy1 <= sy0:
            continue

        sub_mask = mask[sy0:sy1, sx0:sx1]
        n = int(sub_mask.sum())
        if n < args.min_cc_area:
            continue

        r = ratio[sy0:sy1, sx0:sx1][sub_mask]
        e = gx_m[sy0:sy1, sx0:sx1][sub_mask]

        bw = float(x1 - x0) / float(W)
        bh = float(y1 - y0) / float(H)
        coverage = (bw ** 2.0) * (bh ** 1.0)

        score = float(np.mean(r) * np.mean(e) * coverage)

        if bw < 0.45:
            continue

        # mild center preference as tie-breaker
        dist2 = (cx - (W - 1) * 0.5) ** 2 + (cy - (H - 1) * 0.5) ** 2

        if best is None:
            best = (score, dist2, x0, y0, x1, y1)
        else:
            if score > best[0] * (1.0 + 1e-6):
                best = (score, dist2, x0, y0, x1, y1)
            elif abs(score - best[0]) <= best[0] * 1e-6 and dist2 < best[1]:
                best = (score, dist2, x0, y0, x1, y1)

    # If prefer_right filtered everything, retry without it
    if best is None and args.prefer_right:
        args2 = argparse.Namespace(**vars(args))
        args2.prefer_right = False
        return detect_stripe_field_bbox(gray_f32, args2)

    if best is None:
        x0 = int(W * 0.30); x1 = int(W * 0.90)
        y0 = int(H * 0.10); y1 = int(H * 0.90)
        return x0, y0, x1, y1

    x0, y0, x1, y1 = best[2], best[3], best[4], best[5]

    # Optional tightening: trim weak columns/rows inside bbox
    pad = int(args.tighten_pad)
    x0t = max(0, x0 - pad); x1t = min(W, x1 + pad)
    y0t = max(0, y0 - pad); y1t = min(H, y1 + pad)

    sub = gx_m[y0t:y1t, x0t:x1t]
    col = sub.mean(axis=0)
    row = sub.mean(axis=1)

    col_thr = float(np.percentile(col, args.tighten_pct))
    row_thr = float(np.percentile(row, args.tighten_pct))

    cols = np.flatnonzero(col >= col_thr)
    rows = np.flatnonzero(row >= row_thr)

    if cols.size > 10 and rows.size > 10:
        x0 = x0t + int(cols[0])
        x1 = x0t + int(cols[-1]) + 1
        y0 = y0t + int(rows[0])
        y1 = y0t + int(rows[-1]) + 1

    # Expand to full stripe field using projections (fixes "bbox too small")
    x0, y0, x1, y1 = expand_bbox_by_projections(x0, y0, x1, y1, gx_m, ratio, mean_m, args)
    # Final refinement to trim stray dark/flat columns
    x0, y0, x1, y1 = refine_bbox_edges(x0, y0, x1, y1, gx_m, ratio, mean_m, args)
    return int(x0), int(y0), int(x1), int(y1)


# ------------------ per-file ------------------
def process_file(path: str, args):
    rgb = load_color_from_ome(path)
    gray = rgb_to_gray_f32(rgb)
    H, W = gray.shape

    x0, y0, x1, y1 = detect_stripe_field_bbox(gray, args)
    x0, y0, x1, y1 = apply_margins(x0, y0, x1, y1, W, H, args.left, args.right, args.top, args.bottom)

    crop_rgb = rgb[y0:y1, x0:x1]

    base, _ = os.path.splitext(path)
    out_png = base + "_crop.png"
    Image.fromarray(crop_rgb).save(out_png)

    if args.debug:
        dbg = draw_rect_rgb(rgb, x0, y0, x1, y1)
        Image.fromarray(dbg).save(base + "_debug.png")

        crop_dbg = crop_rgb.copy()
        hh, ww = crop_dbg.shape[:2]
        t = 2
        crop_dbg[0:t, :, :] = [255, 0, 0]
        crop_dbg[hh - t:hh, :, :] = [255, 0, 0]
        crop_dbg[:, 0:t, :] = [255, 0, 0]
        crop_dbg[:, ww - t:ww, :] = [255, 0, 0]
        Image.fromarray(crop_dbg).save(base + "_crop_debug.png")

    return out_png, (x0, y0, x1, y1)


# ------------------ CLI / Main ------------------
def parse_args():
    p = argparse.ArgumentParser(description="Crop to electrode stripe field (reject dark background)")

    p.add_argument("--root", default=".", help="Root folder to search (default: current directory).")
    p.add_argument("--pattern", default="*_bright.tif*", help="Glob pattern for input files.")

    # Mask construction
    p.add_argument("--energy_win_x", type=int, default=31, help="Local window X for gradient mean.")
    p.add_argument("--energy_win_y", type=int, default=31, help="Local window Y for gradient mean.")
    p.add_argument("--energy_pct", type=float, default=92.0, help="Percentile threshold for gx local mean.")
    p.add_argument("--min_ratio", type=float, default=2.8, help="Min anisotropy ratio gx/(gy+eps).")
    p.add_argument("--mean_win_x", type=int, default=51, help="Local window X for intensity mean.")
    p.add_argument("--mean_win_y", type=int, default=51, help="Local window Y for intensity mean.")
    p.add_argument("--mean_pct", type=float, default=40.0,
                   help="Brightness floor percentile (reject dark background).")

    # Morphology + CC
    p.add_argument("--close_radius", type=int, default=6, help="Binary closing radius.")
    p.add_argument("--min_cc_area", type=int, default=2000, help="Min connected-component area.")
    p.add_argument("--seed_inner_frac", type=float, default=0.80,
                   help="Only score CC pixels inside central fraction (0..1).")

    # Selection bias (useful for your shown example)
    p.add_argument("--prefer_right", action="store_true",
                   help="Prefer components whose centroid is on the right side of the image.")
    p.add_argument("--prefer_right_xmin", type=float, default=0.50,
                   help="Centroid must be >= this fraction of width if --prefer_right.")

    # Tightening pass
    p.add_argument("--tighten_pad", type=int, default=10, help="Pad around bbox for tightening pass.")
    p.add_argument("--tighten_pct", type=float, default=55.0, help="Percentile for tightening on gx_m.")

    # Expansion pass (projection-based)
    p.add_argument("--expand_cols_pct", type=float, default=55.0,
                   help="Percentile threshold on column stripe score for expanding bbox.")
    p.add_argument("--expand_rows_pct", type=float, default=55.0,
                   help="Percentile threshold on row stripe score for expanding bbox.")
    p.add_argument("--expand_close_radius", type=int, default=20,
                   help="(Deprecated) Closing radius for expansion masks.")
    p.add_argument("--expand_close_cols", type=int, default=35,
                   help="Closing radius for column expansion mask (fills gaps).")
    p.add_argument("--expand_close_rows", type=int, default=12,
                   help="Closing radius for row expansion mask (fills gaps).")
    p.add_argument("--expand_pad", type=int, default=8,
                   help="Extra padding after expansion (pixels).")
    p.add_argument("--expand_cols_hi", type=float, default=60.0,
                   help="High percentile for column seed (hysteresis).")
    p.add_argument("--expand_cols_lo", type=float, default=30.0,
                   help="Low percentile for column expansion (hysteresis).")
    p.add_argument("--expand_rows_hi", type=float, default=70.0,
                   help="High percentile for row seed (hysteresis).")
    p.add_argument("--expand_rows_lo", type=float, default=40.0,
                   help="Low percentile for row expansion (hysteresis).")
    p.add_argument("--expand_gap_max", type=int, default=40,
                   help="Max gap (pixels) to bridge when merging runs in expansion.")
    p.add_argument("--expand_smooth", type=int, default=41,
                   help="Smoothing window for row/col score in expansion.")

    # Refinement pass
    p.add_argument("--refine_mean_hi", type=float, default=60.0)
    p.add_argument("--refine_mean_lo", type=float, default=30.0)
    p.add_argument("--refine_score_hi", type=float, default=60.0)
    p.add_argument("--refine_score_lo", type=float, default=25.0)
    p.add_argument("--refine_close_radius", type=int, default=2)
    p.add_argument("--refine_gap_max", type=int, default=20)
    p.add_argument("--refine_min_width_frac", type=float, default=0.75)
    p.add_argument("--refine_pad", type=int, default=2)

    # margins
    p.add_argument("--left", type=int, default=0, help="Extra pixels to include on LEFT (negative shrinks).")
    p.add_argument("--right", type=int, default=0, help="Extra pixels to include on RIGHT (negative shrinks).")
    p.add_argument("--top", type=int, default=0, help="Extra pixels to include on TOP (negative shrinks).")
    p.add_argument("--bottom", type=int, default=0, help="Extra pixels to include on BOTTOM (negative shrinks).")

    # legacy aliases
    p.add_argument("--padL", type=int, default=0, help="Alias for --left (added).")
    p.add_argument("--padR", type=int, default=0, help="Alias for --right (added).")
    p.add_argument("--padT", type=int, default=0, help="Alias for --top (added).")
    p.add_argument("--padB", type=int, default=0, help="Alias for --bottom (added).")

    p.add_argument("--debug", action="store_true", help="Save debug images.")
    return p.parse_args()


def main():
    args = parse_args()

    # merge legacy pad* flags into margins
    args.left += args.padL
    args.right += args.padR
    args.top += args.padT
    args.bottom += args.padB

    root = os.path.abspath(args.root)
    pattern = os.path.join(root, "**", args.pattern)
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No files found for pattern: {args.pattern} under {root}")
        return

    for f in files:
        print(f"Cropping {f} ...")
        out_png, bbox = process_file(f, args)
        print(f"  -> {out_png}  bbox={bbox}")


if __name__ == "__main__":
    main()
