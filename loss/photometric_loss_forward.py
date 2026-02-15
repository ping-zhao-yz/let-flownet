import torch
import numpy as np
import cv2
import torch.nn as nn

from models.warp.Forward_Warp.forward_warp import forward_warp_fn


"""
Robust Charbonnier loss.
"""
def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = torch.mean(
        torch.pow(torch.mul(delta, delta) + torch.mul(epsilon, epsilon), alpha)
    )
    return loss


"""
Multi-scale photometric loss, as defined in equation (3) of the paper.
"""
def photometric_loss_forward(prev_images_temp, next_images_temp, event_images, output, device, print_details, weights=None):
    prev_images = np.array(prev_images_temp)
    next_images = np.array(next_images_temp)

    total_photometric_loss = 0.0

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

        prev_images_warped = forward_warp_fn.apply(prev_images_gpu, flow)
        error_temp_forward = prev_images_warped - next_images_gpu
        photometric_loss_forward = charbonnier_loss(error_temp_forward)

        if print_details:
            print(f'photometric_loss_forward: {photometric_loss_forward}')

        total_photometric_loss += weights[i] * photometric_loss_forward

    if print_details:
        print(f'total_photometric_loss: {total_photometric_loss}')

    return total_photometric_loss
