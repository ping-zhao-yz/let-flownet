import torch
import numpy as np
import torch.nn.functional as F

def supervised_loss_multiscale(flow_preds, flow_gt, mask_gt, print_details, weights=None):
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
        
        gt_scaled = F.interpolate(flow_gt, size=(h, w), mode='bilinear', align_corners=False)
        # Scale the magnitude!
        gt_scaled[:, 0, :, :] *= scale_x
        gt_scaled[:, 1, :, :] *= scale_y
        
        # ---> CRITICAL FIX: Use max_pool to preserve sparse boolean points without aliasing <---
        mask_scaled = F.adaptive_max_pool2d(mask_gt, output_size=(h, w))

        # ---> FIX: Isolate 2D flow (u, v) and ignore the 3rd channel <---
        pred_eff = pred[:, :2, :, :]
            
        # Calculate L1 loss over valid pixels
        diff = torch.abs(pred_eff - gt_scaled)
        
        # ---> FIX: Safe Denominator Clamping <---
        valid_pixel_count = mask_scaled.sum()
        
        # Only calculate loss for this scale if there are valid ground-truth pixels!
        if valid_pixel_count > 10: 
            l1_loss = (diff * mask_scaled).sum() / (valid_pixel_count * 2)
            if print_details:
                print(f'supervised_loss (scale {i}): {l1_loss.item()}')

            loss += weights[i] * l1_loss
        else:
            if print_details:
                print(f'supervised_loss (scale {i}): SKIPPED (Not enough valid pixels)')

            # If the mask is empty at this resolution, append 0 loss to maintain the gradient graph
            loss += weights[i] * (pred_eff * 0.0).sum()

    if print_details:
        print('total_supervised_loss: {0}'.format(loss.item()))

    return loss
