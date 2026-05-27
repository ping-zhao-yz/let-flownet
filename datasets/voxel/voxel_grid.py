import torch

def events_to_voxel_grid(events, num_bins, height, width, device=torch.device('cuda')):
    """
    Builds a 2-channel voxel grid separating positive and negative events.
    
    :param events: a [N x 4] NumPy array or PyTorch Tensor in the form: [timestamp, x, y, polarity]
    :param num_bins: number of temporal bins
    :param height: height of the voxel grid
    :param width: width of the voxel grid
    :param device: target device (defaults to cuda)
    :return: PyTorch tensor of shape [num_bins, 2, height, width]
    """
    
    # 1. Initialize with 2 channels [num_bins, 2, height, width]
    # Total elements = num_bins * 2 * height * width (e.g., 1,310,720 elements)
    voxel_grid = torch.zeros(num_bins, 2, height, width, dtype=torch.float32, device=device).flatten()

    # Safety check for empty windows
    if events is None or len(events) == 0:
        return voxel_grid.view(num_bins, 2, height, width)

    with torch.no_grad():
        # 2. Convert to tensor and move to device
        if not isinstance(events, torch.Tensor):
            events_torch = torch.from_numpy(events)
        else:
            events_torch = events
            
        events_torch = events_torch.to(device)

        # 3. Extract components (Assuming order: t, x, y, p based on your stacking)
        # KEEP TIMESTAMP IN DOUBLE (float64) PRECISION
        ts_double = events_torch[:, 0].double()
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
        pols = events_torch[:, 3].float()

        # 4. Time normalization
        last_stamp = ts_double[-1]
        first_stamp = ts_double[0]
        deltaT = last_stamp - first_stamp
        if deltaT == 0:
            deltaT = 1.0

        # Compute normalized time, THEN safely cast to float32
        ts = ((num_bins - 1) * (ts_double - first_stamp) / deltaT).float()
        
        tis = torch.floor(ts).long()
        dts = ts - tis.float()

        # 5. Interpolation Weights (Not multiplied by polarity)
        vals_left = 1.0 - dts
        vals_right = dts

        # 6. Masks for routing to the correct channel safely
        mask_pos = (pols > 0.0)
        mask_neg = (pols <= 0.0)    # Catches 0.0 (MVSEC default) and -1.0

        # Base flat index: x + y*W + t*(2*H*W)
        # Channel offsets: Pos = 0, Neg = H*W
        channel_offset = height * width
        base_idx_left = xs + ys * width + tis * (2 * height * width)
        base_idx_right = xs + ys * width + (tis + 1) * (2 * height * width)

        # --- LEFT BIN ACCUMULATION ---
        valid_left = (tis >= 0) & (tis < num_bins)
        
        valid_pos_left = valid_left & mask_pos
        if valid_pos_left.any():
            voxel_grid.index_add_(dim=0, index=base_idx_left[valid_pos_left], source=vals_left[valid_pos_left])
            
        valid_neg_left = valid_left & mask_neg
        if valid_neg_left.any():
            voxel_grid.index_add_(dim=0, index=base_idx_left[valid_neg_left] + channel_offset, source=vals_left[valid_neg_left])

        # --- RIGHT BIN ACCUMULATION ---
        valid_right = ((tis + 1) >= 0) & ((tis + 1) < num_bins)
        
        valid_pos_right = valid_right & mask_pos
        if valid_pos_right.any():
            voxel_grid.index_add_(dim=0, index=base_idx_right[valid_pos_right], source=vals_right[valid_pos_right])
            
        valid_neg_right = valid_right & mask_neg
        if valid_neg_right.any():
            voxel_grid.index_add_(dim=0, index=base_idx_right[valid_neg_right] + channel_offset, source=vals_right[valid_neg_right])

    # Return the correctly shaped tensor: [num_bins, 2, height, width]
    return voxel_grid.view(num_bins, 2, height, width)