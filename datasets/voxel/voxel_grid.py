import torch

def events_to_voxel_grid(events_t, events_x, events_y, events_p, num_bins, height, width):
    """
    Build a voxel grid with bilinear interpolation in the time domain.
    Assumes all input tensors are already on the GPU.
    """
    assert num_bins > 0
    assert width > 0
    assert height > 0

    with torch.no_grad():
        # Initialize the voxel grid on GPU
        voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device='cuda').flatten()

        # Ensure tensors are float/long as needed for indexing and accumulation
        ts = events_t.float()
        xs = events_x.long()
        ys = events_y.long()
        pols = events_p.float()
        pols[pols == 0] = -1

        # Normalize timestamps to [0, num_bins - 1]
        last_stamp = ts[-1]
        first_stamp = ts[0]
        deltaT = last_stamp - first_stamp
        if deltaT == 0:
            deltaT = 1.0

        ts = (num_bins - 1) * (ts - first_stamp) / deltaT

        tis = torch.floor(ts)
        tis_long = tis.long()
        dts = ts - tis
        vals_left = pols * (1.0 - dts.float())
        vals_right = pols * dts.float()

        valid_indices = tis < num_bins
        valid_indices &= tis >= 0
        voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices]
                                * width + tis_long[valid_indices] * width * height,
                                source=vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        valid_indices &= tis >= 0

        voxel_grid.index_add_(dim=0,
                                index=xs[valid_indices] + ys[valid_indices] * width
                                + (tis_long[valid_indices] + 1) * width * height,
                                source=vals_right[valid_indices])

        voxel_grid = voxel_grid.view(num_bins, height, width)

    return voxel_grid