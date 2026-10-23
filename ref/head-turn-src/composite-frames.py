"""Blend head-turn frames back onto the still plate so everything outside the moving zone is exactly the plate.

usage: composite-frames.py <plate-clip.png (580x600)> <in-dir> <out-dir> [x0 y0 x1 y1 feather]
Frames in <in-dir> may be any size (e.g. 4x super-resolved); they are resized to the plate clip size first.
The moving zone (default x 100..480, y 30..565) is kept from the frame; it feathers out to the plate over
`feather` px (default 40). With stitching on, the neck and collar are static, so the feather sits on static content.
"""
import sys, os, glob, cv2, numpy as np
plate_p, in_dir, out_dir = sys.argv[1:4]
x0, y0, x1, y1, feather = (int(v) for v in (sys.argv[4:9] or [100, 30, 480, 565, 40]))
os.makedirs(out_dir, exist_ok=True)
plate = cv2.imread(plate_p).astype(np.float32)
Hh, Ww = plate.shape[:2]
# distance-based soft box: 1 inside the box, linear ramp to 0 over `feather` px outside it
yy, xx = np.mgrid[0:Hh, 0:Ww]
dx = np.maximum(np.maximum(x0 - xx, xx - x1), 0)
dy = np.maximum(np.maximum(y0 - yy, yy - y1), 0)
d = np.sqrt(dx * dx + dy * dy)
m = np.clip(1 - d / feather, 0, 1)
m = (m * m * (3 - 2 * m))[..., None].astype(np.float32)          # smoothstep
files = sorted(glob.glob(os.path.join(in_dir, '*.png')))
for f in files:
    fr = cv2.imread(f)
    if fr.shape[:2] != (Hh, Ww):
        fr = cv2.resize(fr, (Ww, Hh), interpolation=cv2.INTER_AREA)
    out = fr.astype(np.float32) * m + plate * (1 - m)
    cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), np.clip(out, 0, 255).astype(np.uint8))
print(f'composited {len(files)} frames, box ({x0},{y0})-({x1},{y1}) feather {feather}')
