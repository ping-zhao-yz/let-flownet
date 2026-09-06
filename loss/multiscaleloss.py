import torch
import numpy as np
import cv2

def smooth_loss(flow_predictions):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(flow_predictions) not in [tuple, list]:
        flow_predictions = [flow_predictions]

    loss = 0
    weight = 1.0

    for flow in flow_predictions:
        dx, dy = gradient(flow)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (
            dx2.abs().mean() 
            + dxdy.abs().mean() 
            + dydx.abs().mean() 
            + dy2.abs().mean()
        )*weight
        weight /= 2.0
    return loss

def smooth_loss_single(flow):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    dx, dy = gradient(flow)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss = (
        dx2.abs().mean() 
        + dxdy.abs().mean() 
        + dydx.abs().mean() 
        + dy2.abs().mean()
    )
    return loss

"""
DEPRECATED: Calculates per pixel flow error between flow_pred and flow_gt. event_img is used to mask out any pixels without events
"""
def flow_error_dense(flow_gt, flow_pred, event_img, is_car=False):
    max_row = flow_gt.shape[1]
    if is_car == True:
        max_row = 190

    flow_pred = np.array(flow_pred)
    event_img = np.array(event_img)

    event_img_cropped = np.squeeze(event_img)[:max_row, :]
    flow_gt_cropped = flow_gt[:max_row, :]
    flow_pred_cropped = flow_pred[:max_row, :]

    event_mask = event_img_cropped > 0

    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(
        np.logical_and(
            ~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])
        ),
        np.linalg.norm(flow_gt_cropped, axis=2) > 0,
    )
    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))

    gt_masked = flow_gt_cropped[total_mask, :]
    pred_masked = flow_pred_cropped[total_mask, :]

    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    EE_gt = np.linalg.norm(gt_masked, axis=-1)

    n_points = EE.shape[0]

    # Percentage of points with EE > 3 pixels.
    thresh = 3.0
    percent_Outlier = float((EE > thresh).sum()) / float(EE.shape[0] + 1e-5)

    EE = torch.from_numpy(EE)
    EE_gt = torch.from_numpy(EE_gt)

    if torch.sum(EE) == 0:
        AEE = 0
        AEE_sum_temp = 0

        AEE_gt = 0
        AEE_sum_temp_gt = 0
    else:
        AEE = torch.mean(EE)
        AEE_sum_temp = torch.sum(EE)

        AEE_gt = torch.mean(EE_gt)
        AEE_sum_temp_gt = torch.sum(EE_gt)

    return AEE, percent_Outlier, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt

"""
DSEC benchmark metrics (new)
"""
def flow_error_dense_dsec(gt_flow, pred_flow, mask_tensor, is_car=False):
    """
    Calculates standard DSEC benchmark metrics (backwards compatible with MVSEC).
    gt_flow: [H, W, 2] numpy array
    pred_flow: [H, W, 2] numpy array
    mask_tensor: [H, W] boolean array of valid pixels
    """
    max_row = gt_flow.shape[0]
    if is_car:
        max_row = 190

    gt_flow_cropped = gt_flow[:max_row, :]
    pred_flow_cropped = pred_flow[:max_row, :]
    mask_tensor_cropped = mask_tensor[:max_row, :]

    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(
        np.logical_and(
            ~np.isinf(gt_flow_cropped[:, :, 0]), ~np.isinf(gt_flow_cropped[:, :, 1])
        ),
        np.linalg.norm(gt_flow_cropped, axis=2) > 0,
    )
    total_mask = np.logical_and(mask_tensor_cropped, flow_mask)

    # Isolate valid pixels using the combined mask
    gt_u = gt_flow_cropped[:, :, 0][total_mask]
    gt_v = gt_flow_cropped[:, :, 1][total_mask]
    pred_u = pred_flow_cropped[:, :, 0][total_mask]
    pred_v = pred_flow_cropped[:, :, 1][total_mask]

    n_points = len(gt_u)
    if n_points == 0:
        return 0., 0., 0., 0., 0., 0

    # 1. EPE (Endpoint Error)
    epe = np.sqrt((pred_u - gt_u)**2 + (pred_v - gt_v)**2)
    mean_epe = np.mean(epe)

    # 2. AE (Angular Error)
    dot_product = (pred_u * gt_u) + (pred_v * gt_v) + 1.0
    norm_pred = np.sqrt(pred_u**2 + pred_v**2 + 1.0)
    norm_gt = np.sqrt(gt_u**2 + gt_v**2 + 1.0)
    
    # Clip to prevent NaN errors in arccos due to floating point precision limits
    cos_theta = np.clip(dot_product / (norm_pred * norm_gt), -1.0, 1.0)
    ae = np.arccos(cos_theta) * (180.0 / np.pi)
    mean_ae = np.mean(ae)

    # 3. 1PE, 2PE, 3PE (Outlier Percentages)
    pe1 = np.mean(epe > 1.0) * 100
    pe2 = np.mean(epe > 2.0) * 100
    pe3 = np.mean(epe > 3.0) * 100

    return mean_epe, mean_ae, pe1, pe2, pe3, n_points


"""Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow. x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement."""
def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow, x_indices, y_indices, cv2.INTER_NEAREST)
    flow_y_interp = cv2.remap(y_flow, x_indices, y_indices, cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False

    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor
    return


"""The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we need to propagate the ground truth flow over the time between two images.
This function assumes that the ground truth flow is in terms of pixel displacement, not velocity. Pseudo code for this process is as follows:
x_orig = range(cols)      y_orig = range(rows)
x_prop = x_orig           y_prop = y_orig
Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
for all of these flows:
  x_prop = x_prop + gt_flow_x(x_prop, y_prop)
  y_prop = y_prop + gt_flow_y(x_prop, y_prop)
The final flow, then, is x_prop - x-orig, y_prop - y_orig.
Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.
Inputs:
  x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at each timestamp.
  gt_timestamps - timestamp for each flow array.  start_time, end_time - gt flow will be estimated between start_time and end time."""
def estimate_corresponding_gt_flow(
    x_flow_in, y_flow_in, gt_timestamps, start_time, end_time
):
    # Only evaluate within the bounds of the Ground Truth timestamps
    start_time = max(start_time, gt_timestamps[0])
    end_time = min(end_time, gt_timestamps[-1])
    if start_time >= end_time:
        return np.zeros_like(x_flow_in[0]), np.zeros_like(y_flow_in[0])
    
    x_flow_in = np.array(x_flow_in, dtype=np.float64)
    y_flow_in = np.array(y_flow_in, dtype=np.float64)
    gt_timestamps = np.array(gt_timestamps, dtype=np.float64)
    start_time = np.array(start_time, dtype=np.float64)
    end_time = np.array(end_time, dtype=np.float64)

    # Each gt flow at timestamp gt_timestamps[gt_iter] represents the displacement between gt_iter and gt_iter+1.
    gt_iter = np.searchsorted(gt_timestamps, start_time, side="right") - 1
    gt_iter = max(0, gt_iter) # Ensure we don't grab the last frame

    gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]
    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    dt = end_time - start_time

    # No need to propagate if the desired dt is shorter than the time between gt timestamps.
    if gt_dt > dt:
        return x_flow * dt / gt_dt, y_flow * dt / gt_dt

    x_indices, y_indices = np.meshgrid(
        np.arange(x_flow.shape[1]), np.arange(x_flow.shape[0])
    )
    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (gt_timestamps[gt_iter + 1] - start_time) / gt_dt
    total_dt = gt_timestamps[gt_iter + 1] - start_time

    prop_flow(
        x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=scale_factor
    )
    gt_iter += 1

    while gt_timestamps[gt_iter + 1] < end_time:
        x_flow = np.squeeze(x_flow_in[gt_iter, ...])
        y_flow = np.squeeze(y_flow_in[gt_iter, ...])

        prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask)
        total_dt += gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

        gt_iter += 1

    final_dt = end_time - gt_timestamps[gt_iter]
    total_dt += final_dt

    final_gt_dt = gt_timestamps[gt_iter + 1] - gt_timestamps[gt_iter]

    x_flow = np.squeeze(x_flow_in[gt_iter, ...])
    y_flow = np.squeeze(y_flow_in[gt_iter, ...])

    scale_factor = final_dt / final_gt_dt

    prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return x_shift, y_shift

def supervised_loss_multiscale(flow_preds, flow_gt, mask_gt, weights=None):
    if weights is None:
        weights = [0.01, 0.02, 0.08, 1.0]

    if type(flow_preds) not in [tuple, list]:
        flow_preds = [flow_preds]

    loss = 0
    for i, pred in enumerate(flow_preds):
        b, c, h, w = pred.shape
        
        # Downsample ground truth flow and mask
        scale_x = w / flow_gt.size(3)
        scale_y = h / flow_gt.size(2)
        
        gt_scaled = torch.nn.functional.interpolate(flow_gt, size=(h, w), mode='bilinear', align_corners=False)
        # Scale the magnitude!
        gt_scaled[:, 0, :, :] *= scale_x
        gt_scaled[:, 1, :, :] *= scale_y
        
        mask_scaled = torch.nn.functional.interpolate(mask_gt, size=(h, w), mode='nearest')
        
        # ---> CRITICAL FIX: Isolate 2D flow (u, v) and ignore the 3rd channel <---
        pred_eff = pred[:, :2, :, :]
            
        # Calculate L1 loss over valid pixels
        diff = torch.abs(pred_eff - gt_scaled)
        # Average over all valid scalar components
        l1_loss = (diff * mask_scaled).sum() / (mask_scaled.sum() * 2 + 1e-6)
        
        loss += weights[i] * l1_loss
        
    return loss
