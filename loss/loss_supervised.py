import torch
import numpy as np

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
