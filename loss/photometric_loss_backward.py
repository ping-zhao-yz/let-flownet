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
def photometric_loss_backward(prev_images_temp, next_images_temp, event_images, output, device, print_details, weights=None):
    prev_images = np.array(prev_images_temp)
    next_images = np.array(next_images_temp)

    total_photometric_loss = 0.0
    loss_weight_sum = 0.0

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

        prev_images_gpu = prev_images_resize.to(device)
        next_images_gpu = next_images_resize.to(device)

        next_images_warped = backward_warp(next_images_gpu, flow)
        error_temp_backward = next_images_warped - prev_images_gpu
        photometric_loss_backward = charbonnier_loss(error_temp_backward)

        if print_details:
            print(f'photometric_loss_backward: {photometric_loss_backward}')

        total_photometric_loss += weights[len(weights) - i -1] * photometric_loss_backward
        loss_weight_sum += 1.

    total_photometric_loss = total_photometric_loss / loss_weight_sum

    if print_details:
        print('total_photometric_loss: {0}'.format(total_photometric_loss))

    return total_photometric_loss


"""
Single-scale photometric loss, as defined in equation (3) of the paper.
"""
def photometric_loss_backward_single(prev_images_temp, next_images_temp, event_images, output, device, print_details, weights=None):
    prev_images = np.array(prev_images_temp)
    next_images = np.array(next_images_temp)

    flow = output

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

    prev_images_gpu = prev_images_resize.to(device)
    next_images_gpu = next_images_resize.to(device)

    next_images_warped = backward_warp(next_images_gpu, flow)
    error_temp_backward = next_images_warped - prev_images_gpu
    photometric_loss_backward = charbonnier_loss(error_temp_backward)

    total_photometric_loss = photometric_loss_backward

    if print_details:
        print('total_photometric_loss: {0}'.format(total_photometric_loss))

    return total_photometric_loss
