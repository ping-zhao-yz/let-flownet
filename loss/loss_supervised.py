import torch
import torch.nn.functional as F

def supervised_loss_multiscale(flow_preds, flow_gt, mask_gt, print_details=False, weights=None):
    if weights is None:
        weights = [0.01, 0.02, 0.08, 1.0]

    if type(flow_preds) not in [tuple, list]:
        flow_preds = [flow_preds]

    loss = 0
    # The mask is no longer downsampled, so the valid count is perfectly accurate
    valid_pixel_count = mask_gt.sum()

    for i, pred in enumerate(flow_preds):
        b, c, h, w = pred.shape
        
        pred_eff = pred[:, :2, :, :]

        # ---> CRITICAL FIX: Upsample Prediction instead of Downsampling Ground Truth <---
        # This prevents the sparse zeros in the GT from diluting the flow targets.
        pred_up = F.interpolate(pred_eff, size=(flow_gt.size(2), flow_gt.size(3)), mode='bilinear', align_corners=False)
        
        # Scale the prediction magnitudes up to the full resolution space
        pred_up[:, 0, :, :] *= (flow_gt.size(3) / w)
        pred_up[:, 1, :, :] *= (flow_gt.size(2) / h)
        
        if valid_pixel_count > 10: 
            diff = torch.abs(pred_up - flow_gt)
            l1_loss = (diff * mask_gt).sum() / (valid_pixel_count * 2)
            loss += weights[i] * l1_loss
            
            if print_details:
                print(f'supervised_loss (scale {i}): {l1_loss.item():.3f}')
        else:
            loss += weights[i] * (pred_up * 0.0).sum()

            if print_details:
                print(f'supervised_loss (scale {i}): SKIPPED (Not enough valid pixels)')

    if print_details:
        print(f'total_supervised_loss: {loss.item():.3f}')
        
    return loss
