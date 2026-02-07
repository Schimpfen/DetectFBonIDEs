#!/usr/bin/env python3
"""
vsi_to_ome_tiff.py
Wrapper to convert .vsi -> per-channel OME-TIFF using Fiji macro or Bio-Formats tools.

Usage:
  python vsi_to_ome_tiff.py --backend bftools --indir "C:\\path\\to\\in" --outdir "C:\\path\\to\\out"
  python vsi_to_ome_tiff.py --backend bftools --indir "C:\\path\\to\\in" --recursive --outdir "C:\\path\\to\\out"
  python vsi_to_ome_tiff.py --backend fiji --indir "C:\\path\\to\\in" --outdir "C:\\path\\to\\out" --fiji "C:\\Fiji.app\\ImageJ-win64.exe"
  python vsi_to_ome_tiff.py --backend fiji --indir "C:\\path\\to\\in" --no-split   (outdir defaults to indir)

You can also set FIJI_PATH or IMAGEJ_PATH instead of --fiji.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import glob

import numpy as np
from PIL import Image

DEFAULT_BFTOOLS_URL = "https://downloads.openmicroscopy.org/bio-formats/8.4.0/artifacts/bftools.zip"


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_macro = os.path.join(here, "vsi_to_ome_tiff.ijm")

    p = argparse.ArgumentParser(
        description="Convert .vsi -> per-channel OME-TIFF via Fiji macro or Bio-Formats tools."
    )
    p.add_argument("--backend", choices=["bftools", "fiji"], default="bftools",
                   help="Conversion backend. 'bftools' downloads Bio-Formats tools if missing (default).")
    p.add_argument("--indir", required=True, help="Input folder containing .vsi files.")
    p.add_argument("--outdir", help="Output folder (defaults to indir).")
    p.add_argument("--fiji", help="Path to Fiji/ImageJ executable (required for --backend fiji).")
    p.add_argument("--bftools-dir", help="Directory containing bfconvert(.bat) and showinf(.bat).")
    p.add_argument("--bftools-url", default=DEFAULT_BFTOOLS_URL,
                   help="Download URL for bftools.zip when --bftools-dir is not set.")
    p.add_argument("--macro", default=default_macro, help="Path to vsi_to_ome_tiff.ijm.")
    p.add_argument("--recursive", action="store_true",
                   help="Process .vsi files in all subfolders of indir.")
    p.add_argument("--flat", action="store_true",
                   help="When --recursive and --outdir is set, write all outputs into outdir (no subfolders).")
    p.add_argument("--run-mode", choices=["macro", "run"], default="macro",
                   help="How to launch the ImageJ macro: 'macro' uses -macro (ImageJ1), 'run' uses --run (ImageJ2).")
    p.add_argument("--debug-log", action="store_true",
                   help="Write a per-folder macro log file into each output folder.")
    p.add_argument("--wait-ms", type=int, default=60000,
                   help="Max wait (ms) for Bio-Formats to open images before continuing.")
    p.add_argument("--do-split", dest="do_split", action="store_true", default=True,
                   help="Split channels (default).")
    p.add_argument("--no-split", dest="do_split", action="store_false",
                   help="Do not split channels.")
    p.add_argument("--brighten", action="store_true",
                   help="Create extra brightened figure copies (separate files).")
    p.add_argument("--bright-factor", type=float, default=2.0,
                   help="Brightness multiplier for figure copies (higher = whiter).")
    p.add_argument("--bright-gamma", type=float, default=0.7,
                   help="Gamma correction for figure copies (<1 brightens shadows).")
    p.add_argument("--bright-suffix", default="_bright",
                   help="Suffix to add to brightened figure filenames.")
    p.add_argument("--bright-format", choices=["tif", "png"], default="tif",
                   help="Format for brightened figure copies.")
    return p.parse_args()


def resolve_fiji(path):
    if path:
        return path
    return os.getenv("FIJI_PATH") or os.getenv("IMAGEJ_PATH")

def find_executable(root, names):
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        lower = {f.lower(): f for f in filenames}
        for name in names:
            hit = lower.get(name.lower())
            if hit:
                return os.path.join(dirpath, hit)
    return None

def ensure_bftools(bftools_dir, url):
    base = bftools_dir or os.path.join(os.path.expanduser("~"), ".bftools")
    bfconvert = find_executable(base, ["bfconvert.bat", "bfconvert"])
    showinf = find_executable(base, ["showinf.bat", "showinf"])
    if bfconvert and showinf:
        return bfconvert, showinf

    os.makedirs(base, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "bftools.zip")
        print(f"Downloading bftools from {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)

        bfconvert_tmp = find_executable(tmp, ["bfconvert.bat", "bfconvert"])
        showinf_tmp = find_executable(tmp, ["showinf.bat", "showinf"])
        if not bfconvert_tmp or not showinf_tmp:
            raise RuntimeError("bftools.zip did not contain bfconvert/showinf.")

        src_root = os.path.dirname(bfconvert_tmp)
        shutil.copytree(src_root, base, dirs_exist_ok=True)

    bfconvert = find_executable(base, ["bfconvert.bat", "bfconvert"])
    showinf = find_executable(base, ["showinf.bat", "showinf"])
    if not bfconvert or not showinf:
        raise RuntimeError("Failed to install bftools correctly.")
    return bfconvert, showinf

def build_cmd(exe, *args):
    if exe.lower().endswith(".bat"):
        safe_args = [a.replace("%", "%%") for a in args]
        return ["cmd", "/c", exe, *safe_args]
    return [exe, *args]

def run_capture(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)

def iter_vsi_files(root, recursive):
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".vsi"):
                    yield os.path.join(dirpath, name)
    else:
        for name in os.listdir(root):
            if name.lower().endswith(".vsi"):
                yield os.path.join(root, name)

def brighten_image(in_path, out_path, factor=2.0, gamma=0.7):
    im = Image.open(in_path)
    arr = np.asarray(im)
    if arr.dtype == np.uint16:
        maxv = 65535.0
    elif arr.dtype == np.uint8:
        maxv = 255.0
    else:
        # fallback to float scaling
        arr = arr.astype(np.float32)
        maxv = float(arr.max()) if arr.max() > 0 else 1.0

    arr_f = arr.astype(np.float32) / maxv
    if gamma != 1.0:
        arr_f = np.power(arr_f, gamma)
    if factor != 1.0:
        arr_f = arr_f * factor
    arr_f = np.clip(arr_f, 0.0, 1.0)
    out = (arr_f * maxv).astype(arr.dtype)

    out_im = Image.fromarray(out)
    out_im.save(out_path)

def bright_output_path(in_path, suffix, fmt):
    base, _ = os.path.splitext(in_path)
    return f"{base}{suffix}.{fmt}"

def brighten_outputs(paths, factor, gamma, suffix, fmt):
    for p in paths:
        out_path = bright_output_path(p, suffix, fmt)
        brighten_image(p, out_path, factor=factor, gamma=gamma)

def glob_from_pattern(pattern):
    return glob.glob(pattern.replace("%c", "*").replace("%s", "*"))

def parse_series_info(showinf_output):
    series = []
    current = None
    for line in showinf_output.splitlines():
        line = line.strip()
        if line.startswith("Series #"):
            if current is not None:
                series.append(current)
            current = {"SizeC": None, "RGB": False}
            continue
        m = re.search(r"SizeC\\s*=\\s*(\\d+)", line)
        if m:
            if current is None:
                current = {"SizeC": None, "RGB": False}
            current["SizeC"] = int(m.group(1))
        if line.startswith("RGB ="):
            if current is None:
                current = {"SizeC": None, "RGB": False}
            current["RGB"] = "true" in line.lower()
    if current is not None:
        series.append(current)
    if not series:
        return [{"SizeC": 1, "RGB": False}]
    for s in series:
        if s["SizeC"] is None:
            s["SizeC"] = 1
    return series

def convert_with_bftools(bfconvert, showinf, in_path, out_dir, do_split):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_cmd(showinf, "-nopix", in_path)
    info = run_capture(cmd).stdout
    series_info = parse_series_info(info)

    base = os.path.splitext(os.path.basename(in_path))[0]
    for s, info_s in enumerate(series_info):
        if do_split:
            out_name = f"{base}_S{s}_C%c.ome.tif" if len(series_info) > 1 else f"{base}_C%c.ome.tif"
            out_path = os.path.join(out_dir, out_name)
            args = ["-overwrite", "-autoscale", "-series", str(s)]
            if info_s.get("RGB"):
                args.append("-separate")
            args += [in_path, out_path]
            cmd = build_cmd(bfconvert, *args)
            subprocess.run(cmd, check=True)
            yield from glob_from_pattern(out_path)
        else:
            out_name = f"{base}_S{s}.ome.tif" if len(series_info) > 1 else f"{base}.ome.tif"
            out_path = os.path.join(out_dir, out_name)
            cmd = build_cmd(bfconvert, "-overwrite", "-autoscale", "-series", str(s), in_path, out_path)
            subprocess.run(cmd, check=True)
            yield out_path

def has_vsi_files(folder):
    for name in os.listdir(folder):
        if name.lower().endswith(".vsi"):
            return True
    return False

def find_vsi_dirs(root):
    dirs = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith(".vsi") for f in filenames):
            dirs.append(dirpath)
    return dirs

def run_macro(fiji, macro, indir, outdir, do_split, run_mode, log_path=None, wait_ms=60000):
    args_str = f"indir={indir} outdir={outdir} doSplit={'true' if do_split else 'false'} waitMs={int(wait_ms)}"
    if log_path:
        args_str += f" log={log_path}"
    if run_mode == "run":
        cmd = [fiji, "--headless", "--ij2", "--run", macro, args_str]
    else:
        cmd = [fiji, "--ij1", "-batch", macro, args_str]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    indir = os.path.abspath(args.indir)
    outdir = os.path.abspath(args.outdir) if args.outdir else indir
    macro = os.path.abspath(args.macro)

    if not os.path.isdir(indir):
        print(f"ERROR: indir does not exist: {indir}")
        return 2
    if args.flat and (not args.recursive or not args.outdir):
        print("ERROR: --flat only applies when both --recursive and --outdir are provided.")
        return 2

    if args.backend == "bftools":
        try:
            bfconvert, showinf = ensure_bftools(args.bftools_dir, args.bftools_url)
        except Exception as e:
            print(f"ERROR: failed to set up bftools: {e}")
            return 2

        files = list(iter_vsi_files(indir, args.recursive))
        if not files:
            print(f"ERROR: no .vsi files found under: {indir}")
            return 2

        for f in files:
            if args.outdir:
                if args.flat:
                    target_out = outdir
                else:
                    rel = os.path.relpath(os.path.dirname(f), indir)
                    target_out = os.path.join(outdir, rel)
            else:
                target_out = os.path.dirname(f)
            outputs = list(convert_with_bftools(bfconvert, showinf, f, target_out, args.do_split))
            if args.brighten and outputs:
                brighten_outputs(
                    outputs,
                    factor=args.bright_factor,
                    gamma=args.bright_gamma,
                    suffix=args.bright_suffix,
                    fmt=args.bright_format,
                )
        return 0

    # Fiji backend
    fiji = resolve_fiji(args.fiji)
    if not fiji:
        print("ERROR: Fiji/ImageJ path not provided. Use --fiji or set FIJI_PATH/IMAGEJ_PATH.")
        return 2
    if not os.path.isfile(macro):
        print(f"ERROR: macro not found: {macro}")
        return 2

    if not args.recursive:
        if not has_vsi_files(indir):
            print(f"ERROR: no .vsi files found in: {indir}")
            return 2
        os.makedirs(outdir, exist_ok=True)
        log_path = os.path.join(outdir, "_vsi_to_ome_log.txt") if args.debug_log else None
        run_macro(
            fiji, macro, indir, outdir, args.do_split, args.run_mode,
            log_path=log_path, wait_ms=args.wait_ms
        )
        return 0

    vsi_dirs = find_vsi_dirs(indir)
    if not vsi_dirs:
        print(f"ERROR: no .vsi files found under: {indir}")
        return 2

    for d in vsi_dirs:
        if args.outdir:
            if args.flat:
                target_out = outdir
            else:
                rel = os.path.relpath(d, indir)
                target_out = os.path.join(outdir, rel)
        else:
            target_out = d

        os.makedirs(target_out, exist_ok=True)
        log_path = os.path.join(target_out, "_vsi_to_ome_log.txt") if args.debug_log else None
        run_macro(
            fiji, macro, d, target_out, args.do_split, args.run_mode,
            log_path=log_path, wait_ms=args.wait_ms
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
