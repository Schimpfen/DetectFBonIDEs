#!/usr/bin/env python3
"""
detectFBincrop_v2.py -- Detect fluorescent particles on CROPPED images.
Also supports *_debug.png by auto-detecting the red crop rectangle and restricting detection to its interior.

Outputs per image (unless disabled by flags):
  - *_overlay.png (skip with --no-overlays)
  - *_mask.png (skip with --no-mask)
  - if matching *_crop_debug.png exists: *_crop_debug_overlay.png (skip with --no-overlays)
And summary_counts.csv (the only CSV output)

Main behavior:
  - ROI = inner red rectangle if present (debug images) OR full image (crop images)
  - Do NOT accidentally process *_debug.png unless --include-debug
  - Peak detection on LoG response with MAD-based thresholding
  - Adaptive V/mag thresholds inside ROI using background-suppressed signals
  - Optional "small-particle mode" to preserve 1-5 px components
"""

from __future__ import annotations

import os
import glob
import argparse
import csv
import numpy as np
import cv2


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _suppress_vertical_stripes(gray: np.ndarray, kx: int) -> np.ndarray:
    """
    For vertical stripe patterns (variation along x), estimate background by blurring along x only, then subtract.
    """
    kx = int(kx)
    if kx < 3:
        kx = 3
    if kx % 2 == 0:
        kx += 1

    bg = cv2.blur(gray, (kx, 1))
    hp = cv2.subtract(gray, bg)
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return hp


def _ensure_odd(k: int, *, min_k: int = 3) -> int:
    k = int(k)
    if k < min_k:
        k = min_k
    if k % 2 == 0:
        k += 1
    return k


def _normalize_to_u8(arr: np.ndarray, roi_mask: np.ndarray | None = None) -> np.ndarray:
    a = arr.astype(np.float32)
    if roi_mask is not None:
        vals = a[roi_mask.astype(bool)]
        if vals.size == 0:
            return np.zeros_like(a, dtype=np.uint8)
        mn = float(np.min(vals))
        mx = float(np.max(vals))
    else:
        mn = float(np.min(a))
        mx = float(np.max(a))
    if mx <= mn:
        return np.zeros_like(a, dtype=np.uint8)
    scaled = (a - mn) * (255.0 / (mx - mn))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _background_suppress_u8(
    img_u8: np.ndarray,
    *,
    stripe_kx: int,
    use_tophat: bool,
    tophat_ksize: int,
) -> np.ndarray:
    hp = _suppress_vertical_stripes(img_u8, kx=stripe_kx)
    if use_tophat:
        k = _ensure_odd(tophat_ksize, min_k=3)
        thk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        hp = cv2.morphologyEx(hp, cv2.MORPH_TOPHAT, thk)
    return hp


def _magenta_local_contrast(mag_score: np.ndarray, ksize: int) -> np.ndarray:
    k = _ensure_odd(ksize, min_k=3)
    mag_f = mag_score.astype(np.float32)
    blur = cv2.GaussianBlur(mag_f, (k, k), sigmaX=0, sigmaY=0)
    return mag_f - blur


def _percentile_robust(arr: np.ndarray, roi_idx: np.ndarray, pct: float) -> int:
    vals = arr[roi_idx]
    if vals.size == 0:
        return 0
    nonzero = vals[vals > 0]
    if nonzero.size >= max(100, int(0.01 * vals.size)):
        return int(np.percentile(nonzero, pct))
    return int(np.percentile(vals, pct))


def _mad_sigma(vals: np.ndarray) -> float:
    if vals.size == 0:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return 1.4826 * mad


def _elongation_ratio(weight: np.ndarray) -> float:
    if weight.size == 0:
        return 1.0
    w = weight.astype(np.float32)
    s = float(np.sum(w))
    if s <= 0:
        return 1.0
    h, wdim = w.shape
    ys, xs = np.mgrid[0:h, 0:wdim].astype(np.float32)
    cx = float(np.sum(xs * w) / s)
    cy = float(np.sum(ys * w) / s)
    x0 = xs - cx
    y0 = ys - cy
    cov_xx = float(np.sum(w * x0 * x0) / s)
    cov_yy = float(np.sum(w * y0 * y0) / s)
    cov_xy = float(np.sum(w * x0 * y0) / s)
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    if det < 0:
        det = 0.0
    disc = max(0.0, trace * trace / 4.0 - det)
    l1 = trace / 2.0 + np.sqrt(disc)
    l2 = trace / 2.0 - np.sqrt(disc)
    return (l1 + 1e-6) / (l2 + 1e-6)


def _blob_response(gray_hp: np.ndarray, sigma: float) -> np.ndarray:
    """
    LoG-ish bright blob response (float):
      - Gaussian blur
      - Laplacian
      - invert (bright blobs -> positive)
    """
    sigma = float(sigma)
    g = cv2.GaussianBlur(gray_hp.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    resp = (-lap)
    return resp


def _magenta_score(bgr: np.ndarray) -> np.ndarray:
    """
    Magenta score = (R+B) - 2G in int16.
    Higher => more magenta/pink.
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return (r + b) - 2 * g


def _find_red_rect_roi(bgr: np.ndarray, *, tol: int = 35, min_len: int = 50, pad_in: int = 3):
    """
    Detect the inner ROI from a red rectangle drawn as (255,0,0) in RGB (appears as BGR (0,0,255)).
    Returns (x0,y0,x1,y1) exclusive bounds of ROI interior, or None if not found.
    """
    B = bgr[:, :, 0].astype(np.int16)
    G = bgr[:, :, 1].astype(np.int16)
    R = bgr[:, :, 2].astype(np.int16)

    # "red-ish" mask: R high, G/B low-ish
    red = (R >= 200) & (G <= tol) & (B <= tol)
    red_u8 = (red.astype(np.uint8) * 255)

    # connect line segments
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_u8 = cv2.morphologyEx(red_u8, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(red_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # choose contour with largest area that looks rectangular-ish
    best = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_len or h < min_len:
            continue
        area = w * h
        if best is None or area > best[0]:
            best = (area, x, y, w, h)

    if best is None:
        return None

    _, x, y, w, h = best

    # interior ROI (exclude the red border thickness)
    x0 = x + pad_in
    y0 = y + pad_in
    x1 = x + w - pad_in
    y1 = y + h - pad_in

    if x1 <= x0 + 10 or y1 <= y0 + 10:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _make_roi_mask(shape_hw, roi):
    H, W = shape_hw
    m = np.zeros((H, W), dtype=np.uint8)
    if roi is None:
        m[:, :] = 255
        return m
    x0, y0, x1, y1 = roi
    m[y0:y1, x0:x1] = 255
    return m


def detect_particles(
    bgr: np.ndarray,
    *,
    roi,  # (x0,y0,x1,y1) or None
    min_area: int,
    max_area: int,
    min_circularity: float,
    stripe_kx: int,
    log_sigma: float,
    use_tophat: bool,
    tophat_ksize: int,
    mag_prior_mode: str,
    mag_local_ksize: int,
    median_ksize: int,
    open_ksize: int,
    open_iter: int,
    close_ksize: int,
    close_iter: int,
    resp_k: float,
    nms_ksize: int,
    peak_window: int,
    snr_min: float,
    blob_k: float,
    elong_ratio_max: float,
    mag_window_k: float,
    area_k: float,
    # adaptive thresholding knobs
    v_min_pct: float,
    mag_pct: float,
):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.uint8)
    mag_raw = _magenta_score(bgr)  # int16

    roi_mask = _make_roi_mask(gray.shape, roi)

    # 1) suppress stripes
    hp = _suppress_vertical_stripes(gray, kx=stripe_kx)

    # 2) optional top-hat
    if use_tophat:
        k = _ensure_odd(tophat_ksize, min_k=3)
        thk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        hp = cv2.morphologyEx(hp, cv2.MORPH_TOPHAT, thk)

    # 3) blob response
    resp = _blob_response(hp, sigma=log_sigma)

    # 4) background-suppressed signals for robust percentiles
    v_feat = _background_suppress_u8(v, stripe_kx=stripe_kx, use_tophat=use_tophat, tophat_ksize=tophat_ksize)

    if mag_prior_mode == "local":
        mag_feat_raw = _magenta_local_contrast(mag_raw, mag_local_ksize)
    else:
        mag_feat_raw = mag_raw.astype(np.float32)

    mag_feat_u8 = _normalize_to_u8(mag_feat_raw, roi_mask)
    mag_feat = _background_suppress_u8(
        mag_feat_u8, stripe_kx=stripe_kx, use_tophat=use_tophat, tophat_ksize=tophat_ksize
    )

    # --- adaptive thresholds INSIDE ROI only ---
    roi_idx = roi_mask.astype(bool)
    v_thr = _percentile_robust(v_feat, roi_idx, v_min_pct)
    mag_thr = None
    if mag_prior_mode != "none":
        mag_thr = _percentile_robust(mag_feat, roi_idx, mag_pct)

    resp_roi = resp[roi_idx]
    resp_m = float(np.median(resp_roi)) if resp_roi.size else 0.0
    resp_s = _mad_sigma(resp_roi)
    if resp_s <= 1e-6:
        resp_s = float(np.std(resp_roi)) if resp_roi.size else 0.0
    resp_thr = resp_m + float(resp_k) * float(resp_s)

    prior = (v_feat >= v_thr)
    if mag_prior_mode != "none":
        prior = prior & (mag_feat >= mag_thr)

    cand = (resp >= resp_thr) & prior & roi_idx
    cand_u8 = (cand.astype(np.uint8) * 255)

    # optional cleanup on candidate mask
    if median_ksize and median_ksize > 0:
        k = _ensure_odd(median_ksize, min_k=3)
        cand_u8 = cv2.medianBlur(cand_u8, k)
    if open_ksize and open_ksize > 0:
        k = _ensure_odd(open_ksize, min_k=3)
        cand_u8 = cv2.morphologyEx(
            cand_u8,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=open_iter,
        )
    if close_ksize and close_ksize > 0:
        k = _ensure_odd(close_ksize, min_k=3)
        cand_u8 = cv2.morphologyEx(
            cand_u8,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=close_iter,
        )
    cand = cand_u8.astype(bool)

    # local maxima (NMS)
    nms_k = _ensure_odd(nms_ksize, min_k=3)
    nms_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_k, nms_k))
    dil = cv2.dilate(resp, nms_kernel)
    peak_mask = (resp >= dil) & cand

    # de-duplicate flat plateaus by taking the strongest pixel per component
    peak_u8 = (peak_mask.astype(np.uint8) * 255)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(peak_u8, connectivity=8)

    H, W = gray.shape
    particles = []
    r = int(peak_window)
    if r < 1:
        r = 1

    for i in range(1, num):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue
        idx = int(np.argmax(resp[ys, xs]))
        y = int(ys[idx])
        x = int(xs[idx])

        y0 = max(0, y - r)
        y1 = min(H, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(W, x + r + 1)

        patch = resp[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        med = float(np.median(patch))
        mad = _mad_sigma(patch)
        if mad <= 1e-6:
            mad = float(np.std(patch))
        if mad <= 1e-6:
            continue

        peak_val = float(resp[y, x])
        snr = (peak_val - med) / mad
        if snr < snr_min:
            continue

        # blobness: center should be brighter than ring
        if blob_k > 0:
            ph, pw = patch.shape
            ys2, xs2 = np.mgrid[0:ph, 0:pw]
            cy = y - y0
            cx = x - x0
            dist = np.sqrt((ys2 - cy) ** 2 + (xs2 - cx) ** 2)
            ring = (dist >= max(1, r - 1)) & (dist <= r + 0.5)
            if ring.any():
                ring_mean = float(np.mean(patch[ring]))
            else:
                ring_mean = med
            if (peak_val - ring_mean) < (blob_k * mad):
                continue

        # shape / elongation
        w = patch - med
        w[w < 0] = 0
        elong = _elongation_ratio(w)
        if elong > elong_ratio_max:
            continue

        # optional local magenta contrast check (not global)
        if mag_window_k > 0:
            mpatch = mag_feat[y0:y1, x0:x1].astype(np.float32)
            m_med = float(np.median(mpatch))
            m_mad = _mad_sigma(mpatch)
            if m_mad <= 1e-6:
                m_mad = float(np.std(mpatch))
            if m_mad > 1e-6:
                if (float(mag_feat[y, x]) - m_med) < (mag_window_k * m_mad):
                    continue

        area_thr = med + area_k * mad
        area_mask = patch >= area_thr
        area = int(np.sum(area_mask))
        if area < min_area or area > max_area:
            continue

        gray_patch = gray[y0:y1, x0:x1]
        if area_mask.any():
            mean_int = float(np.mean(gray_patch[area_mask]))
        else:
            mean_int = float(gray[y, x])

        circularity = float(1.0 / max(1.0, elong))
        if circularity < min_circularity:
            continue

        particles.append((float(x), float(y), float(area), float(mean_int), float(circularity)))

    debug = {
        "thresholds": {
            "v_thr": v_thr,
            "mag_thr": mag_thr,
            "resp_thr": resp_thr,
            "resp_m": resp_m,
            "resp_s": resp_s,
            "resp_k": resp_k,
        },
        "signals": {
            "v_raw": v,
            "v_feat": v_feat,
            "mag_raw": mag_raw.astype(np.float32),
            "mag_feat": mag_feat,
            "resp": resp,
        },
        "roi_mask": roi_mask,
        "roi": roi,
        "mag_prior_mode": mag_prior_mode,
    }
    return particles, cand_u8, debug


def draw_overlay(bgr: np.ndarray, particles, roi, radius=4):
    out = bgr.copy()
    for (x, y, area, _, _) in particles:
        cv2.circle(out, (int(round(x)), int(round(y))), int(radius), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    return out


def _write_diagnostics(base: str, debug: dict, args) -> tuple[str, str]:
    roi_idx = debug["roi_mask"].astype(bool)
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    def stats_row(name: str, arr: np.ndarray, thr: float | None, note: str):
        vals = arr[roi_idx]
        if vals.size == 0:
            qv = [np.nan] * len(qs)
            mean = std = vmin = vmax = np.nan
        else:
            qv = [float(np.percentile(vals, q)) for q in qs]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
        thr_val = "" if thr is None else thr
        return [name] + qv + [mean, std, vmin, vmax, thr_val, note]

    stats_path = base + "_diagnostics.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["signal"]
            + [f"q{q}" for q in qs]
            + ["mean", "std", "min", "max", "threshold", "note"]
        )
        settings_note = "; ".join(
            [
                f"mag_prior={debug['mag_prior_mode']}",
                f"v_min_pct={args.v_min_pct}",
                f"mag_pct={args.mag_pct}",
                f"stripe_kx={args.stripe_kx}",
                f"log_sigma={args.log_sigma}",
                f"use_tophat={not args.no_tophat}",
                f"tophat_ksize={args.tophat_ksize}",
                f"median_ksize={args.median_ksize}",
                f"open_ksize={args.open_ksize}",
                f"close_ksize={args.close_ksize}",
                f"resp_k={args.resp_k}",
                f"nms_ksize={args.nms_ksize}",
                f"peak_window={args.peak_window}",
                f"snr_min={args.snr_min}",
                f"blob_k={args.blob_k}",
                f"elong_ratio_max={args.elong_ratio_max}",
                f"mag_window_k={args.mag_window_k}",
                f"area_k={args.area_k}",
                f"min_area={args.min_area}",
                f"max_area={args.max_area}",
                f"min_circularity={args.min_circularity}",
            ]
        )
        w.writerow(["settings"] + [""] * len(qs) + ["", "", "", "", "", settings_note])
        w.writerow(
            stats_row(
                "v_raw",
                debug["signals"]["v_raw"],
                None,
                "V channel (raw)",
            )
        )
        w.writerow(
            stats_row(
                "v_feat",
                debug["signals"]["v_feat"],
                debug["thresholds"]["v_thr"],
                "V after stripe/tophat (used for threshold)",
            )
        )
        w.writerow(
            stats_row(
                "mag_raw",
                debug["signals"]["mag_raw"],
                None,
                "mag_score=(R+B)-2G (raw)",
            )
        )
        w.writerow(
            stats_row(
                "mag_feat",
                debug["signals"]["mag_feat"],
                debug["thresholds"]["mag_thr"],
                "mag prior after local/global + stripe/tophat (used for threshold)",
            )
        )
        w.writerow(
            stats_row(
                "resp",
                debug["signals"]["resp"],
                debug["thresholds"]["resp_thr"],
                f"LoG response (m={debug['thresholds']['resp_m']:.4g}, s={debug['thresholds']['resp_s']:.4g}, k={debug['thresholds']['resp_k']})",
            )
        )

    hist_path = base + "_hist.csv"
    bins = np.arange(257)
    v_hist, _ = np.histogram(debug["signals"]["v_feat"][roi_idx], bins=bins)
    mag_hist, _ = np.histogram(debug["signals"]["mag_feat"][roi_idx], bins=bins)
    resp_hist_src = _normalize_to_u8(debug["signals"]["resp"], debug["roi_mask"])
    resp_hist, _ = np.histogram(resp_hist_src[roi_idx], bins=bins)
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "v_feat", "mag_feat", "resp"])
        for i in range(256):
            w.writerow([i, int(v_hist[i]), int(mag_hist[i]), int(resp_hist[i])])

    return stats_path, hist_path


def process_file(path, args):
    bgr = load_bgr(path)

    # ROI logic:
    # - if debug image has a red crop rectangle: use its interior
    # - otherwise assume it is already a crop: ROI = full image
    roi = _find_red_rect_roi(bgr) if args.allow_debug_roi else None

    particles, mask, debug = detect_particles(
        bgr,
        roi=roi,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        stripe_kx=args.stripe_kx,
        log_sigma=args.log_sigma,
        use_tophat=not args.no_tophat,
        tophat_ksize=args.tophat_ksize,
        mag_prior_mode=args.mag_prior,
        mag_local_ksize=args.mag_local_ksize,
        median_ksize=args.median_ksize,
        open_ksize=args.open_ksize,
        open_iter=args.open_iter,
        close_ksize=args.close_ksize,
        close_iter=args.close_iter,
        resp_k=args.resp_k,
        nms_ksize=args.nms_ksize,
        peak_window=args.peak_window,
        snr_min=args.snr_min,
        blob_k=args.blob_k,
        elong_ratio_max=args.elong_ratio_max,
        mag_window_k=args.mag_window_k,
        area_k=args.area_k,
        v_min_pct=args.v_min_pct,
        mag_pct=args.mag_pct,
    )

    base, _ = os.path.splitext(path)
    overlay_png = None
    mask_png = None
    if not args.no_overlays:
        overlay = draw_overlay(bgr, particles, roi, radius=args.radius)
        overlay_png = base + "_overlay.png"
        cv2.imwrite(overlay_png, overlay)
    if not args.no_mask:
        mask_png = base + "_mask.png"
        cv2.imwrite(mask_png, mask)

    # If a matching *_crop_debug.png exists, also write an overlay on that image.
    # This keeps detections inside the red ROI for quick visual QA.
    debug_overlay_path = None
    debug_crop_path = None
    if base.endswith("_crop"):
        cand = base + "_debug.png"
        if os.path.exists(cand):
            debug_crop_path = cand

    if (
        (not args.no_overlays)
        and debug_crop_path is not None
        and os.path.abspath(debug_crop_path) != os.path.abspath(path)
    ):
        dbg_bgr = load_bgr(debug_crop_path)
        dbg_roi = _find_red_rect_roi(dbg_bgr) if args.allow_debug_roi else None
        if dbg_roi is not None:
            x0, y0, x1, y1 = dbg_roi
            particles_dbg = [p for p in particles if (x0 <= p[0] < x1 and y0 <= p[1] < y1)]
        else:
            particles_dbg = particles
        dbg_overlay = draw_overlay(dbg_bgr, particles_dbg, dbg_roi, radius=args.radius)
        debug_overlay_path = os.path.splitext(debug_crop_path)[0] + "_overlay.png"
        cv2.imwrite(debug_overlay_path, dbg_overlay)

    return (
        overlay_png,
        mask_png,
        len(particles),
        debug["thresholds"],
        debug_overlay_path,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Detect fluorescent particles on cropped images; optional debug-ROI support.")
    p.add_argument(
        "paths",
        nargs="*",
        help="Optional file(s) or folder(s). If omitted, searches for *_crop.* in the current tree.",
    )
    p.add_argument(
        "--small-particle-mode",
        "--small-particle",
        action="store_true",
        dest="small_particle_mode",
        help="Enable small-particle defaults (min_area=1, looser thresholds, no median/open).",
    )
    p.add_argument(
        "--preset",
        choices=["normal", "loose"],
        default="normal",
        help="Percentile preset for V/mag thresholds. normal=60/75, loose=35/55.",
    )

    p.add_argument("--min-area", type=int, default=None, help="Minimum component area. Default normal=6, small=1.")
    p.add_argument("--max-area", type=int, default=600)
    p.add_argument("--min-circularity", type=float, default=0.20)
    p.add_argument("--radius", type=int, default=4)

    p.add_argument("--stripe-kx", type=int, default=61, help="Horizontal blur kernel for stripe suppression.")
    p.add_argument("--log-sigma", type=float, default=1.2, help="Blob scale. 0.9-1.8 typical.")
    p.add_argument("--no-tophat", action="store_true")
    p.add_argument("--tophat-ksize", type=int, default=9)

    p.add_argument(
        "--mag-prior",
        choices=["local", "global", "none"],
        default="local",
        help="Magenta prior mode: local (contrast), global (raw mag), or none.",
    )
    p.add_argument(
        "--mag-local-ksize",
        type=int,
        default=51,
        help="Gaussian kernel size for local magenta contrast (odd).",
    )

    p.add_argument(
        "--median-ksize",
        type=int,
        default=None,
        help="Median blur ksize. 0 disables. Default normal=3, small=0.",
    )
    p.add_argument(
        "--open-ksize",
        type=int,
        default=None,
        help="Morph open ksize. 0 disables. Default normal=3, small=0.",
    )
    p.add_argument("--open-iter", type=int, default=1)
    p.add_argument("--close-ksize", type=int, default=5)
    p.add_argument("--close-iter", type=int, default=1)

    # Peak detection / response thresholding
    p.add_argument(
        "--resp-k",
        type=float,
        default=None,
        help="MAD threshold multiplier for LoG response. Typical range 4-8.",
    )
    p.add_argument("--nms-ksize", type=int, default=3, help="NMS kernel size (odd).")
    p.add_argument("--peak-window", type=int, default=4, help="Patch radius (px) for peak classification.")
    p.add_argument("--snr-min", type=float, default=3.0, help="Minimum peak SNR.")
    p.add_argument("--blob-k", type=float, default=0.5, help="Center-vs-ring threshold in MADs (0 to disable).")
    p.add_argument("--elong-ratio-max", type=float, default=3.0, help="Reject if elongation ratio exceeds this.")
    p.add_argument("--mag-window-k", type=float, default=0.0, help="Optional local magenta check in MADs (0 disables).")
    p.add_argument("--area-k", type=float, default=1.5, help="Area mask threshold in MADs for area estimate.")

    # Adaptive thresholds (percentiles inside ROI)
    p.add_argument(
        "--v-min-pct",
        type=float,
        default=None,
        help="Brightness percentile for V threshold inside ROI (higher = stricter).",
    )
    p.add_argument(
        "--mag-pct",
        type=float,
        default=None,
        help="Magenta score percentile inside ROI (higher = stricter).",
    )

    # File selection
    p.add_argument(
        "--include-debug",
        action="store_true",
        help="Also process *_debug.png (ROI will be inferred from red rectangle).",
    )
    p.add_argument("--no-overlays", action="store_true", help="Do not write *_overlay.png.")
    p.add_argument("--no-mask", action="store_true", help="Do not write *_mask.png.")
    p.add_argument(
        "--allow-debug-roi",
        action="store_true",
        default=True,
        help="If image contains a red crop rectangle, restrict detection to its interior.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Resolve defaults based on preset + small-particle mode
    if args.preset == "normal":
        preset_v, preset_mag = 60.0, 75.0
    else:
        preset_v, preset_mag = 35.0, 55.0

    if args.small_particle_mode and args.preset == "normal":
        preset_v, preset_mag = 35.0, 55.0

    if args.v_min_pct is None:
        args.v_min_pct = preset_v
    if args.mag_pct is None:
        args.mag_pct = preset_mag

    if args.resp_k is None:
        args.resp_k = 4.0 if args.small_particle_mode else 6.0

    if args.min_area is None:
        args.min_area = 1 if args.small_particle_mode else 6

    if args.median_ksize is None:
        args.median_ksize = 0 if args.small_particle_mode else 3
    if args.open_ksize is None:
        args.open_ksize = 0 if args.small_particle_mode else 3

    # Only process debug images with this exact suffix by default.
    patterns = [os.path.join("**", "*_S0.ome_bright_debug.png")]
    if args.include_debug:
        patterns.append(os.path.join("**", "*_debug.png"))

    files = []
    if args.paths:
        for pth in args.paths:
            if os.path.isdir(pth):
                for ptn in patterns:
                    files.extend(glob.glob(os.path.join(pth, ptn), recursive=True))
                continue
            hits = glob.glob(pth, recursive=True)
            if hits:
                files.extend(hits)
            elif os.path.isfile(pth):
                files.append(pth)
    else:
        for ptn in patterns:
            files.extend(glob.glob(ptn, recursive=True))

    if not files:
        print("No cropped images found (looked for *_crop.*). Use --include-debug to include *_debug.png.")
        return

    files = sorted(set(files))

    summary_rows = []
    for f in files:
        print(f"Processing {f} ...")
        png, mask, n, thr, dbg_overlay = process_file(f, args)
        outputs = []
        if png:
            outputs.append(png)
        if mask:
            outputs.append(mask)
        out_msg = ", ".join(outputs) if outputs else "(no images written)"
        print(f"  {n} detections -> {out_msg}  thr={thr}")
        if dbg_overlay:
            print(f"  debug crop overlay -> {dbg_overlay}")
        summary_rows.append({"filename": os.path.relpath(f, start="."), "n_particles": n})

    with open("summary_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "n_particles"])
        for row in summary_rows:
            w.writerow([row["filename"], row["n_particles"]])

    print("Summary saved to summary_counts.csv")
    print("Done.")


if __name__ == "__main__":
    main()
