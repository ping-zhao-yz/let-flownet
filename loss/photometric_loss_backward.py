import torch
import numpy as np
import cv2
import torch.nn as nn


"""
Robust Charbonnier loss (Updated to support spatial masking).
"""
def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3, mask=None):
    loss = torch.pow(torch.mul(delta, delta) + torch.mul(epsilon, epsilon), alpha)
    
    # If a mask is provided, zero out the loss in regions without events
    if mask is not None:
        loss = loss * mask
        
    return torch.sum(loss)


"""
warp an image/tensor (im2) back to im1, according to the optical flow
x: [B, C, H, W] (im2), flo: [B, 2, H, W] flow
"""
def backward_warp(x, flo):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=False)
    mask = torch.ones_like(x)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


"""
Multi-scale photometric loss, as defined in equation (3) of the paper.
"""
def photometric_loss_multiscale(prev_images_temp, next_images_temp, event_images, output, device, print_details, weights=None):
    # 1. Expand dimensions from [Batch, H, W] to [Batch, 1, H, W] 
    prev_images_base = prev_images_temp.unsqueeze(1)
    next_images_base = next_images_temp.unsqueeze(1)
    
    # Expand event mask for interpolation
    event_mask_base = event_images.unsqueeze(1).float()

    total_photometric_loss = 0.0

    # Iterate through the multi-scale flow predictions [flow0, flow1, flow2, flow3]
    for i in range(len(output)):
        flow = output[i]
        
        height = flow.size(2)
        width = flow.size(3)

        # 2. Resize images and mask directly on the GPU for the current scale
        prev_images_scaled = nn.functional.interpolate(
            prev_images_base, size=(height, width), mode='bilinear', align_corners=False
        )
        next_images_scaled = nn.functional.interpolate(
            next_images_base, size=(height, width), mode='bilinear', align_corners=False
        )
        
        # Replace 'nearest' interpolation with max pooling to preserve sparse events
        event_mask_scaled = nn.functional.adaptive_max_pool2d(
            event_mask_base, output_size=(height, width)
        )

        valid_mask = (event_mask_scaled > 0).float()

        # Boundary Margin: Ignore the outer 5% to prevent out-of-frame warping outliers
        margin_y = max(1, int(height * 0.05))
        margin_x = max(1, int(width * 0.05))
        valid_mask[:, :, :margin_y, :] = 0
        valid_mask[:, :, -margin_y:, :] = 0
        valid_mask[:, :, :, :margin_x] = 0
        valid_mask[:, :, :, -margin_x:] = 0

        # 3. Calculate Loss
        next_images_warped = backward_warp(next_images_scaled, flow)
        error_temp_backward = next_images_warped - prev_images_scaled
        
        # Pass the mask to charbonnier to ignore blank regions (Returns a sum)
        photometric_loss_scale = charbonnier_loss(error_temp_backward, mask=valid_mask)

        if print_details:
            print(f'photometric_loss (scale {i}): {photometric_loss_scale.item()}')

        # ---> CRITICAL MULTI-SCALE FIXES <---
        # A. Use standard forward indexing (flow0 gets weights[0])
        current_weight = weights[i]
        
        # B. Compensate for the torch.sum() pixel drop-off by normalizing to the 256x256 base size.
        #    flow3 multiplier = 1. flow0 multiplier = 64. 
        pixel_scale_multiplier = (256 * 256) / (height * width)
        
        # C. Accumulate the corrected loss (No division by 4 at the end)
        total_photometric_loss += current_weight * (photometric_loss_scale * pixel_scale_multiplier)

    if print_details:
        print('total_photometric_loss: {0}'.format(total_photometric_loss.item()))

    return total_photometric_loss


"""
Single-scale photometric loss, as defined in equation (3) of the paper.
"""
def photometric_loss_single(prev_images_temp, next_images_temp, event_images, output, device, print_details, weights=None):
    flow = output

    height = flow.size(2)
    width = flow.size(3)

    # 1. Expand dimensions from [Batch, H, W] to [Batch, 1, H, W]
    prev_images_base = prev_images_temp.unsqueeze(1)
    next_images_base = next_images_temp.unsqueeze(1)
    event_mask_base = event_images.unsqueeze(1).float()

    # 2. Resize directly on the GPU using PyTorch
    prev_images_scaled = nn.functional.interpolate(
        prev_images_base, size=(height, width), mode='bilinear', align_corners=False
    )
    next_images_scaled = nn.functional.interpolate(
        next_images_base, size=(height, width), mode='bilinear', align_corners=False
    )
    
    # Replace 'nearest' interpolation with max pooling to preserve sparse events
    event_mask_scaled = nn.functional.adaptive_max_pool2d(
        event_mask_base, output_size=(height, width)
    )
    
    valid_mask = (event_mask_scaled > 0).float()

    # Boundary Margin: Ignore the outer 5% to prevent out-of-frame warping outliers
    margin_y = max(1, int(height * 0.05))
    margin_x = max(1, int(width * 0.05))
    valid_mask[:, :, :margin_y, :] = 0
    valid_mask[:, :, -margin_y:, :] = 0
    valid_mask[:, :, :, :margin_x] = 0
    valid_mask[:, :, :, -margin_x:] = 0
    
    # 3. Calculate Loss
    next_images_warped = backward_warp(next_images_scaled, flow)
    error_temp_backward = next_images_warped - prev_images_scaled
    
    # Pass the mask to charbonnier to ignore blank regions
    photometric_loss = charbonnier_loss(error_temp_backward, mask=valid_mask)

    return photometric_loss