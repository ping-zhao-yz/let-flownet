import torch

device = torch.device("cuda")

def events_to_voxel_grid(events, num_bins, height, width, device=device):
    """
    Build a voxel grid with bilinear interpolation in the time domain.
    Assumes all input tensors are already on the GPU.
    """
    assert num_bins > 0
    assert width > 0
    assert height > 0

    with torch.no_grad():
        events_torch = torch.from_numpy(events).to(device)

        # Initialize the voxel grid on GPU
        voxel_grid = torch.zeros(num_bins, height, width, 2, dtype=torch.float32, device=device).flatten(end_dim=2)

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events_torch[-1, 0]
        first_stamp = events_torch[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT

        ts = events_torch[:, 0]
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()
        pols[pols == 0] = -1  # polarity should be +1 / -1

        mask_pos = (pols == 1)
        mask_neg = (pols == -1)

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = 1.0 - dts.float()
        vals_right = dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0

        datatype = torch.cuda.LongTensor

        voxel_grid[:,0].index_add_(dim=0,
                                index=(xs[valid_indices*mask_pos] + ys[valid_indices*mask_pos]
                                        * width + tis_long[valid_indices*mask_pos] * width * height).type(
                                    datatype),
                                source=vals_left[valid_indices*mask_pos])
        voxel_grid[:,1].index_add_(dim=0,
                                    index=(xs[valid_indices * mask_neg] + ys[valid_indices * mask_neg]
                                            * width + tis_long[valid_indices * mask_neg] * width * height).type(
                                        datatype),
                                    source=vals_left[valid_indices*mask_neg])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid[:,0].index_add_(dim=0,
                                index=(xs[valid_indices*mask_pos] + ys[valid_indices*mask_pos] * width
                                        + (tis_long[valid_indices*mask_pos] + 1) * width * height).type(datatype),
                                source=vals_right[valid_indices*mask_pos])
        voxel_grid[:,1].index_add_(dim=0,
                                index=(xs[valid_indices*mask_neg] + ys[valid_indices*mask_neg] * width
                                        + (tis_long[valid_indices*mask_neg] + 1) * width * height).type(datatype),
                                source=vals_right[valid_indices*mask_neg])

        voxel_grid = voxel_grid.view(num_bins, height, width, 2).permute(0, 3, 1, 2)

    return voxel_grid