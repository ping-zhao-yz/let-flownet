import torch

def sequence_loss(flow_preds, flow_gt, mask_gt, gamma=0.8, print_details=False):
    """
    Loss function defined over sequence of flow predictions
    """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # Ensure mask and gt are matching shapes
    B, C, H, W = flow_preds[0].shape
    
    # Scale up gt and mask to full resolution? 
    # Wait, the flow_preds are already upsampled to full resolution (H, W) by upflow8 inside snn_raft.
    # The GT should be full resolution [B, 2, H, W]
    valid_pixel_count = mask_gt.sum()

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (torch.abs(flow_preds[i] - flow_gt) * mask_gt).sum() / (valid_pixel_count * 2 + 1e-8)
        
        flow_loss += i_weight * i_loss
        
        if print_details:
            print(f'seq_loss (iter {i}): {i_loss.item():.3f} (weight {i_weight:.3f})')
            
    if print_details:
        print(f'total_seq_loss: {flow_loss.item():.3f}')

    return flow_loss
