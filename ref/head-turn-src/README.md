# Head-turn clip (assets/hero.mp4) — how it was made

Source: `src-960.png` = 960×960 crop of `ref/plate-1920x1080.png` at (460,0). The head crop used by the site is (190,150) 580×600 inside it.

Tool: [LivePortrait](https://github.com/KwaiVGI/LivePortrait) (KwaiVGI, open source), image-retargeting path, run locally on Apple Silicon.

Setup (≈3 GB, one-off):
```bash
git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git && cd LivePortrait
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r requirements_macOS.txt
uv pip install "huggingface_hub==0.25.2"        # newer versions break gradio 5.1's import
python -c "from huggingface_hub import snapshot_download; snapshot_download('KwaiVGI/LivePortrait', local_dir='pretrained_weights', allow_patterns=['insightface/*','liveportrait/*'])"
```

Render (from inside the LivePortrait directory; ~70 s for 120 frames):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python /path/to/lp-make-frames.py \
  --src /path/to/src-960.png --out /tmp/frames --frames 120 --yaw -22 --eye 0.8 --crop 190,150,580,600
```
`--yaw -22` makes frame 0 look screen-left and the last frame screen-right (the site maps cursor x to clip time). `--eye` makes the eyes lead the turn. Keep stitching ON (the default, `--stitch 1`): it holds the neck and collar still so the frames can be composited onto the plate. Without it the whole collar shifts ~12 px and no mask can hide the seam. (The "ghost chin" seen earlier came from the page's elliptical mask cutting through the jaw, not from stitching.)

Post-processing (one uniform chain for the whole frame, no masks, no compositing):
```bash
# 1. neck follows the chin, applied inside the 960x960 render at the head crop (190,150) 580x600
mkdir /tmp/crop && for f in /tmp/frames/*.png; do ffmpeg -i "$f" -vf crop=580:600:190:150 /tmp/crop/$(basename "$f"); done
python neck-twist.py plate-clip-580x600.png /tmp/crop /tmp/twist
# 2. paste the twisted crop back into each 960 render, and the 960 render into the full 1920x1080 plate at (460,0)
python - <<'PY'
import cv2, glob, os
plate = cv2.imread('../plate-1920x1080.png')
for f in sorted(glob.glob('/tmp/frames/*.png')):
    b = os.path.basename(f); fr = cv2.imread(f); fr[150:750, 190:770] = cv2.imread('/tmp/twist/' + b)
    out = plate.copy(); out[0:960, 460:1420] = fr; cv2.imwrite('/tmp/full/' + b, out)
PY
# 3. super-resolve EVERY FULL FRAME (Real-ESRGAN x4plus, scaled straight back to 1920x1080): ~30 s/frame, ~1 h for 120
./sr-full.sh /tmp/full /tmp/full-sr
# 4. encode, and use the centre frame as the poster so image and video are literally the same pixels
ffmpeg -framerate 24 -i /tmp/full-sr/f%03d.png -vf "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:v libx264 -preset slow -crf 21 -g 4 -keyint_min 4 -sc_threshold 0 -movflags +faststart hero.mp4     # ≈ 7 MB
ffmpeg -i /tmp/full-sr/f060.png -q:v 2 plate.jpg
```
Why no masks: every partial approach (head-only clip over the image, box mask, motion mask) left a boundary somewhere between "video pixels" and "image pixels" that showed up as a rectangle, dark bands or ghosting in some browser. Now the entire frame goes through the same chain and the poster is one of the frames.

The page shows `hero.mp4` covering the frame 1:1; `plate.jpg` is the clip's centre frame and only shows until the clip is ready.

`turn.html` + `render-turn.mjs` are the earlier pseudo-3D attempt (sphere-warp of the still image), kept for reference; `head-pseudo3d.mp4` is its output.
