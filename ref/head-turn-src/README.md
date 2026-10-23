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

Sharpen (LivePortrait renders the face at 512 px, so it comes out softer than the plate): run the official
[Real-ESRGAN ncnn macOS build](https://github.com/xinntao/Real-ESRGAN/releases) over the frames with the general x4plus model, then scale back:
```bash
./realesrgan-ncnn-vulkan -i /tmp/frames -o /tmp/frames-sr -n realesrgan-x4plus -s 4 -f png     # ≈ 8 min for 120 frames on an M-series GPU
for f in /tmp/frames-sr/*.png; do ffmpeg -i "$f" -vf scale=580:600:flags=lanczos /tmp/frames-580/$(basename "$f"); done
python composite-frames.py plate-clip-580x600.png /tmp/frames-sr /tmp/frames-comp 100 30 480 550 45   # also scales 4x frames back to 580x600
python neck-twist.py plate-clip-580x600.png /tmp/frames-comp /tmp/frames-twist                         # neck follows the chin (see below)
python - <<'PY'   # paste each 580x600 frame into the full 1920x1080 plate at (650,150)
import cv2, glob, os
plate = cv2.imread('../plate-1920x1080.png')
for f in sorted(glob.glob('/tmp/frames-twist/*.png')):
    out = plate.copy(); out[150:750, 650:1230] = cv2.imread(f); cv2.imwrite('/tmp/full/' + os.path.basename(f), out)
PY
ffmpeg -framerate 24 -i /tmp/full/f%03d.png -vf "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:v libx264 -preset slow -crf 21 -g 4 -keyint_min 4 -sc_threshold 0 -movflags +faststart hero.mp4     # ≈ 7 MB
```
`composite-frames.py` blends each frame back onto the plate crop (`plate-clip-580x600.png` = plate px (650,150) 580×600): the moving zone x 100–480 / y 30–550 is kept, everything else feathers to the plate over 45 px.

`neck-twist.py`: with stitching on, LivePortrait keeps the neck frozen while the head turns, which reads as "the Adam's apple doesn't move". The script measures each frame's chin displacement (skin centroid in the chin band) and shears the neck interior by that amount at the top of the neck, tapering to zero at the base; the shift also fades to zero at the neck silhouette (distance profile) so no pixel samples the cushion. Rendering with stitching OFF instead moves the collar ~12 px and smears the cushion; blending the two renders was tried and doesn't work (they don't line up).

The final clip is the **whole scene**, not a head-only crop: the page shows `hero.mp4` covering the frame 1:1 (the plate JPEG is only the poster until the clip is ready). A head-only clip pasted over the image showed its rectangle in Safari/macOS, which renders video and images with different tone curves; with one full-frame video there is nothing to mismatch. The BT.709 tags above keep that tone shift small anyway.

`turn.html` + `render-turn.mjs` are the earlier pseudo-3D attempt (sphere-warp of the still image), kept for reference; `head-pseudo3d.mp4` is its output.
