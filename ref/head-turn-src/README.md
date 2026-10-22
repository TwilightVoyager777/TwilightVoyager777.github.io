# Head-turn clip (assets/head.mp4) — how it was made

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
ffmpeg -framerate 24 -i /tmp/frames/f%03d.png -c:v libx264 -preset slow -crf 18 -g 4 -keyint_min 4 -sc_threshold 0 -pix_fmt yuv420p -movflags +faststart head.mp4
```
`--yaw -22` makes frame 0 look screen-left and the last frame screen-right (the site maps cursor x to clip time). `--eye` makes the eyes lead the turn. Increase `--scale` (default 2.3) if the paste-back boundary shows.

`turn.html` + `render-turn.mjs` are the earlier pseudo-3D attempt (sphere-warp of the still image), kept for reference; `head-pseudo3d.mp4` is its output.
