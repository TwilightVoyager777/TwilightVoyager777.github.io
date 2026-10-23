"""Composite rendered frames onto the plate using a MOTION mask: only pixels that actually change somewhere in the
sequence come from the (processed) frames, everything static is the plate itself — so re-synthesis / super-resolution
tone shifts in static areas (cushion, collar) can never show.

usage: composite-motion.py <plate-clip.png> <raw-frames-dir> <processed-frames-dir> <out-dir> [--thresh 10] [--preview out.png]
  raw-frames-dir       : frames straight from the renderer (same size as the plate), used only to detect motion
  processed-frames-dir : the frames to composite (e.g. super-resolved; resized to the plate size)
"""
import sys, os, glob, cv2, numpy as np
plate_p, raw_dir, proc_dir, out_dir = sys.argv[1:5]
thresh = float(sys.argv[sys.argv.index('--thresh') + 1]) if '--thresh' in sys.argv else 10.0
preview = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None
DILATE, FEATHER = 8, 6
os.makedirs(out_dir, exist_ok=True)
plate = cv2.imread(plate_p); Hh, Ww = plate.shape[:2]
pb = cv2.GaussianBlur(plate, (0, 0), 2).astype(np.float32)
motion = np.zeros((Hh, Ww), np.float32)
raws = sorted(glob.glob(os.path.join(raw_dir, '*.png')))
for f in raws:
    fr = cv2.imread(f)
    if fr.shape[:2] != (Hh, Ww): fr = cv2.resize(fr, (Ww, Hh), interpolation=cv2.INTER_AREA)
    d = np.abs(cv2.GaussianBlur(fr, (0, 0), 2).astype(np.float32) - pb).max(2)
    motion = np.maximum(motion, d)
m = (motion > thresh).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))            # drop isolated specks
n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
keep = np.zeros_like(m)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 400: keep[lab == i] = 1                   # keep real regions only
m = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
flood = m.copy(); ff = np.zeros((Hh + 2, Ww + 2), np.uint8); cv2.floodFill(flood, ff, (0, 0), 1); m = m | (1 - flood)
m = cv2.dilate(m, np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8)).astype(np.float32)
M = cv2.GaussianBlur(m, (0, 0), FEATHER)[..., None]
if preview:
    vis = plate.copy(); vis[..., 1] = np.clip(vis[..., 1] * (0.35 + 0.65 * M[..., 0]), 0, 255)
    cv2.imwrite(preview, np.hstack([vis, cv2.cvtColor((M[..., 0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
procs = sorted(glob.glob(os.path.join(proc_dir, '*.png')))
for f in procs:
    fr = cv2.imread(f)
    if fr.shape[:2] != (Hh, Ww): fr = cv2.resize(fr, (Ww, Hh), interpolation=cv2.INTER_AREA)
    out = fr.astype(np.float32) * M + plate.astype(np.float32) * (1 - M)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), np.clip(out, 0, 255).astype(np.uint8))
print(f'motion mask: {int((M[...,0] > 0.5).sum())} px of {Hh*Ww}; composited {len(procs)} frames (thresh {thresh})')
