"""Give the (stitched, static) neck a twist that follows the chin.

usage: composite-frames5.py <plate-clip.png> <stitched-frames-dir> <out-dir> [--gain 1.0] [--preview out.png]
For each frame: measure how far the chin has moved sideways versus the plate (skin centroid in the chin band),
then shear the neck interior horizontally by that amount at the top of the neck, tapering to zero at the base.
The shift fades to zero at the neck silhouette (distance profile), so edges, cushion and collar stay untouched.
"""
import sys, os, glob, cv2, numpy as np
plate_p, in_dir, out_dir = sys.argv[1:4]
gain = float(sys.argv[sys.argv.index('--gain') + 1]) if '--gain' in sys.argv else 1.0
preview = sys.argv[sys.argv.index('--preview') + 1] if '--preview' in sys.argv else None
CHIN_Y0, CHIN_Y1 = 500, 522          # chin band used to measure sideways motion
NECK_TOP, NECK_BASE = 526, 590       # shear = full at NECK_TOP, zero at NECK_BASE
DEPTH = 40
NECK_X0, NECK_X1 = 165, 415
os.makedirs(out_dir, exist_ok=True)
plate = cv2.imread(plate_p); Hh, Ww = plate.shape[:2]

def skin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = ((hsv[..., 0] <= 25) & (hsv[..., 1] >= 35) & (hsv[..., 1] <= 190) & (hsv[..., 2] >= 90)).astype(np.uint8)
    s[:, :NECK_X0] = 0; s[:, NECK_X1:] = 0
    return s

def chin_x(img):
    s = skin(img)[CHIN_Y0:CHIN_Y1]
    ys, xs = np.nonzero(s)
    return xs.mean() if len(xs) else np.nan

# neck interior mask from the plate (neck only, below the chin)
s = skin(plate); s[:NECK_TOP - 6] = 0
s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
n, lab = cv2.connectedComponents(s); seed = lab[NECK_TOP + 8, 290]
neck = (lab == seed).astype(np.uint8) if seed > 0 else s
flood = neck.copy(); ff = np.zeros((Hh + 2, Ww + 2), np.uint8); cv2.floodFill(flood, ff, (0, 0), 1); neck = neck | (1 - flood)
# displacement profile: 0 on the neck silhouette, full at DEPTH px inside, so no pixel ever samples beyond the edge
dist = cv2.distanceTransform(neck, cv2.DIST_L2, 5)
prof = np.clip(dist / DEPTH, 0, 1); interior = (prof * prof * (3 - 2 * prof)).astype(np.float32)
yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
ramp = np.clip((NECK_BASE - yy) / (NECK_BASE - NECK_TOP), 0, 1)          # 1 at the top of the neck, 0 at the base
ramp = ramp * ramp * (3 - 2 * ramp)
cx0 = chin_x(plate)
files = sorted(glob.glob(os.path.join(in_dir, '*.png')))
for i, f in enumerate(files):
    fr = cv2.imread(f)
    if fr.shape[:2] != (Hh, Ww): fr = cv2.resize(fr, (Ww, Hh), interpolation=cv2.INTER_AREA)
    c = (chin_x(fr) - cx0) * gain
    shift = c * ramp * interior                                             # px to move each pixel (sample from x - shift)
    warped = cv2.remap(fr, (xx - shift).astype(np.float32), yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M = interior[..., None]
    out = warped.astype(np.float32) * M + fr.astype(np.float32) * (1 - M)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), np.clip(out, 0, 255).astype(np.uint8))
    if i in (0, len(files) // 2, len(files) - 1): print(f'{os.path.basename(f)}: chin shift {c:+.1f}px')
    if preview and i == 0: cv2.imwrite(preview, cv2.cvtColor((interior * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
print('neck interior px', int((interior > 0.5).sum()), 'rows', np.where((interior > 0.5).any(1))[0][[0, -1]].tolist())
