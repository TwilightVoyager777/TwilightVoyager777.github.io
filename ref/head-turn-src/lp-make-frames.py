"""Render a head-turn sequence from one portrait with LivePortrait's image-retargeting path.

Run from the LivePortrait repo directory (it resolves pretrained_weights relative to its own package):
  cd LivePortrait && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python <this file> --src IMG --out DIR [--frames 120] [--yaw 22] [--eye 0.8] [--crop x,y,w,h]

Frame i gets yaw = -YAW + 2*YAW*i/(N-1) (even speed). Positive/negative direction is checked afterwards
and the sequence is reversed at encode time if frame 0 does not look screen-left.
"""
import argparse, os, sys, time
import numpy as np, cv2, torch

sys.path.insert(0, os.getcwd())
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src import gradio_pipeline as gp


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--frames', type=int, default=120)
ap.add_argument('--yaw', type=float, default=22.0, help='max |yaw| in degrees')
ap.add_argument('--pitch', type=float, default=0.0, help='constant pitch offset')
ap.add_argument('--eye', type=float, default=0.8, help='eyeball_direction_x = eye * yaw')
ap.add_argument('--scale', type=float, default=2.3, help='cropper scale (bigger = more context, smaller face)')
ap.add_argument('--crop', default='', help='x,y,w,h to cut from the pasted-back frame')
ap.add_argument('--only', default='', help='comma list of frame indices to render (for quick checks)')
ap.add_argument('--no-stitch', action='store_true', help='skip the stitching module (head moves freely, neck not locked)')
ap.add_argument('--stitch', type=float, default=1.0, help='stitching strength 0..1 (1 = full neck lock, 0 = none); --no-stitch forces 0')
ap.add_argument('--vy', type=float, default=-0.125, help='cropper vy_ratio')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

args = ArgumentConfig()
args.scale = a.scale
args.vy_ratio = a.vy
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)
pipe = gp.GradioPipeline(inference_cfg, crop_cfg, args)
w = pipe.live_portrait_wrapper
device = w.device
print('device', device)

# source analysis once (yaw 0): features, keypoints, crop transform, paste-back mask
t0 = time.time()
pipe.init_retargeting_image(a.scale, 0, 0, a.src)
f_s, x_s, R_s, _R_d0, x_s_info, lmk, M_c2o, mask_ori, img_rgb = pipe.prepare_retargeting_image(a.src, 0.0, 0.0, 0.0, a.scale, flag_do_crop=True)
x_s, f_s, R_s = x_s.to(device), f_s.to(device), R_s.to(device)
x_c_s = x_s_info['kp'].to(device)
exp0 = x_s_info['exp'].to(device)
scale0 = x_s_info['scale'].to(device)
t_s = x_s_info['t'].to(device)
pitch0, yaw0, roll0 = x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll']
print(f'source pose pitch={float(pitch0):.1f} yaw={float(yaw0):.1f} roll={float(roll0):.1f}  prep {time.time()-t0:.1f}s  img {img_rgb.shape}')

crop = [int(v) for v in a.crop.split(',')] if a.crop else None
N = a.frames
want = set(int(v) for v in a.only.split(',')) if a.only else None
t0 = time.time()
with torch.no_grad():
    for i in range(N):
        if want is not None and i not in want:
            continue
        t = 0.0 if N == 1 else i / (N - 1)
        yaw = -a.yaw + 2 * a.yaw * t
        R_d = gp.get_rotation_matrix(pitch0 + a.pitch, yaw0 + yaw, roll0).to(device)
        R_new = (R_d @ R_s.permute(0, 2, 1)) @ R_s
        delta = exp0.clone()                                   # the helper edits in place, never reuse exp0
        ex = a.eye * yaw
        if ex != 0:
            delta = pipe.update_delta_new_eyeball_direction(torch.tensor(ex).to(device), torch.tensor(0.0).to(device), delta)
        x_d = scale0 * (x_c_s @ R_new + delta) + t_s
        strength = 0.0 if a.no_stitch else a.stitch
        if strength > 0:
            x_d = x_d + strength * (w.stitching(x_s, x_d) - x_d)
        out = w.warp_decode(f_s, x_s, x_d)
        out = w.parse_output(out['out'])[0]
        frame = gp.paste_back(out, M_c2o, img_rgb, mask_ori)   # RGB uint8, full source size
        if crop:
            x, y, cw, ch = crop
            frame = frame[y:y + ch, x:x + cw]
        cv2.imwrite(os.path.join(a.out, f'f{i:03d}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % 20 == 0 or want is not None:
            print(f'frame {i}/{N} yaw={yaw:+.1f} ex={ex:+.1f}  {time.time()-t0:.1f}s')
print('done', time.time() - t0)
