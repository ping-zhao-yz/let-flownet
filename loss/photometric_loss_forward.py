import torch
import numpy as np
import cv2
import torch.nn as nn

from models.warp.Forward_Warp.forward_warp import forward_warp_fn


"""
Robust Charbonnier loss.
"""
def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = torch.sum(
        torch.pow(torch.mul(delta, delta) + torch.mul(epsilon, epsilon), alpha)
    )
    return loss


"""
warp an image/tensor (im2) back to im1, according to the optical flow
x: [B, C, H, W] (im2), flo: [B, 2, H, W] flow
"""
def backward_warp(x, flo, device):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    grid = grid.to(device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=False)
    mask = torch.ones(x.size()).to(device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=False)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


"""
Multi-scale photometric loss, as defined in equation (3) of the paper.
"""
def photometric_loss_forward(prev_images_temp, next_images_temp, event_images, output, device, print_details, weights=None):
    prev_images = np.array(prev_images_temp)
    next_images = np.array(next_images_temp)

    total_photometric_loss = 0.0
    print_cost_details = True

    for i in range(len(output)):
        flow = output[i]

        m_batch = flow.size(0)
        height = flow.size(2)
        width = flow.size(3)

        prev_images_resize = torch.zeros(m_batch, 1, height, width)
        next_images_resize = torch.zeros(m_batch, 1, height, width)

        for b in range(m_batch):
            prev_images_resize[b, 0, :, :] = torch.from_numpy(
                cv2.resize(
                    prev_images[b, :, :], 
                    (height, width), 
                    interpolation=cv2.INTER_LINEAR
                )
            )
            next_images_resize[b, 0, :, :] = torch.from_numpy(
                cv2.resize(
                    next_images[b, :, :], 
                    (height, width), 
                    interpolation=cv2.INTER_LINEAR
                )
            )

        prev_images_warped = forward_warp_fn.apply(prev_images_resize.to(device), flow.to(device))
        error_temp_forward = prev_images_warped - next_images_resize.to(device)
        photometric_loss_forward = charbonnier_loss(error_temp_forward)

        photometric_loss_scale = torch.log(photometric_loss_forward)

        if print_details & print_cost_details:
            print(f'photometric_loss_forward: {photometric_loss_forward}, photometric_loss_scale: {photometric_loss_scale}')

        total_photometric_loss += weights[len(weights) - i - 1] * photometric_loss_scale

    if print_details & print_cost_details:
        print('total_photometric_loss: {0}'.format(total_photometric_loss))

    return total_photometric_loss
