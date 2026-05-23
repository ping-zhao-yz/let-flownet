import torch

def generate_voxel_grid(events_t, events_x, events_y, events_p, num_bins, height, width):
    """
    Dynamically converts raw events to a bilinear voxel grid.
    Adapted from SDformerFlow loader_utils.
    """
    # 1. Normalize timestamps to [0, num_bins - 1]
    t_start = events_t[0]
    t_end = events_t[-1]
    
    if t_start == t_end: # Edge case protection
        return torch.zeros((2 * num_bins, height, width), dtype=torch.float32)

    t_norm = (events_t - t_start) / (t_end - t_start)
    t_norm = t_norm * (num_bins - 1)

    # 2. Get upper and lower temporal bin indices for bilinear interpolation
    t_floor = torch.floor(t_norm)
    t_ceil = torch.ceil(t_norm)
    
    # Calculate interpolation weights (how close the event is to the bin)
    weight_ceil = t_norm - t_floor
    weight_floor = 1.0 - weight_ceil

    # Ensure indices are integers
    t_floor = t_floor.long()
    t_ceil = t_ceil.long()
    x = events_x.long()
    y = events_y.long()
    
    # Split polarities to separate channels (e.g., pos=0, neg=1)
    # If p is [-1, 1], map to [1, 0]
    pol = (events_p > 0).long() 

    # 3. Create flat empty voxel grid: shape (num_bins * 2_polarities * H * W)
    voxel_grid_flat = torch.zeros(num_bins * 2 * height * width, dtype=torch.float32)

    # 4. Calculate flattened 1D indices for index_add_
    # Channel layout: Time Bin -> Polarity -> Y -> X
    index_floor = (t_floor * 2 * height * width) + (pol * height * width) + (y * width) + x
    index_ceil = (t_ceil * 2 * height * width) + (pol * height * width) + (y * width) + x

    # 5. Scatter the weights using index_add_ (Extremely fast PyTorch C++ backend)
    voxel_grid_flat.index_add_(0, index_floor, weight_floor.float())
    voxel_grid_flat.index_add_(0, index_ceil, weight_ceil.float())

    # 6. Reshape back to 3D tensor: (num_bins * 2, height, width)
    voxel_grid = voxel_grid_flat.view(num_bins * 2, height, width)
    
    return voxel_grid
