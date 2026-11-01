import os
import numpy as np
import shutil
import torch
import cv2


"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [batch_size, width, height, 2]
Output: flow_rgb: [batch_size, width, height, 3]
"""

def flow_viz_np(flow_x, flow_y, max_mag=None):
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)

    ang = np.arctan2(flow_y, flow_x)
    ang += np.pi
    ang *= 180. / np.pi / 2.
    ang = ang.astype(np.uint8)
    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=np.uint8)
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 255
    if max_mag is not None and max_mag > 0:
        hsv[:, :, 2] = np.clip((mag / max_mag) * 255.0, 0, 255).astype(np.uint8)
    else:
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. 'A Database and Evaluation Methodology for Optical Flow' (IJCV, 2011)
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_to_middlebury(flow_x, flow_y, clip_flow=None):
    """
    Renders optical flow in the standard Middlebury color wheel (Baker et al. 2011).
    Zero flow is pure white ([255, 255, 255]).
    Returns BGR uint8 image suitable for cv2.imwrite.
    """
    rad = np.sqrt(flow_x**2 + flow_y**2)
    rad_max = np.percentile(rad, 99.0) if clip_flow is None else clip_flow
    rad_max = max(float(rad_max), 1e-5)

    u_norm = flow_x / rad_max
    v_norm = flow_y / rad_max
    rad_norm = np.sqrt(u_norm**2 + v_norm**2)

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    # Angle normalized to [-pi, pi]
    a = np.arctan2(-v_norm, -u_norm) / np.pi
    fk = (a + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros((flow_x.shape[0], flow_x.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        col0 = colorwheel[:, i][k0] / 255.0
        col1 = colorwheel[:, i][k1] / 255.0
        col = (1.0 - f) * col0 + f * col1

        # Zero motion is pure white
        idx = (rad_norm <= 1.0)
        col[idx] = 1.0 - rad_norm[idx] * (1.0 - col[idx])
        col[~idx] = col[~idx] * 0.75  # Out of range / outliers

        # Channel 2-i converts RGB to BGR for OpenCV
        img[:, :, 2 - i] = np.clip(np.floor(col * 255.0), 0, 255).astype(np.uint8)

    return img


def events_to_bgr(pos_mask, neg_mask):
    """
    Renders bipolar event spikes on clean white background:
    Positive events: Red ([0, 0, 255] in BGR)
    Negative events: Blue ([255, 0, 0] in BGR)
    Background: White ([255, 255, 255] in BGR)
    """
    h, w = pos_mask.shape
    vis = np.full((h, w, 3), 255, dtype=np.uint8)

    # Negative events in Blue
    vis[neg_mask] = [255, 0, 0]
    # Positive events in Red
    vis[pos_mask] = [0, 0, 255]
    # Overlapping events: Purple
    both_mask = np.logical_and(pos_mask, neg_mask)
    vis[both_mask] = [180, 0, 180]

    return vis


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (
        flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path, filename),
                        os.path.join(save_path, 'model_best.pth.tar'))


def flat_model_params(model_params):
    flat_params= []
    for param in model_params:
        flat_param = param.clone().detach().cpu().numpy()
        flat_params.extend(flat_param)
    return flat_params
