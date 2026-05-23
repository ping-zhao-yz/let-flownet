import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset
from datasets.voxel.voxel_grid import generate_voxel_grid

def get_raw_events_for_window(dataset_file, index, dt, xoff=45, yoff=2, orig_w=346, orig_h=260):
    """ Helper to fetch and crop raw events. """
    with h5py.File(dataset_file, 'r') as d_set:
        event_inds = d_set['davis']['left']['image_raw_event_inds']
        if index + dt >= len(event_inds):
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        start_idx = event_inds[index]
        end_idx = event_inds[index + dt]
        if start_idx >= end_idx:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        events = d_set['davis']['left']['events'][start_idx:end_idx]

    events_x = events[:, 0]
    events_y = events[:, 1]
    events_t = events[:, 2]
    events_p = events[:, 3]

    # Crop spatial boundaries
    mask = (events_x >= xoff) & (events_x < orig_w - xoff) & (events_y >= yoff) & (events_y < orig_h - yoff)
    events_x = events_x[mask] - xoff
    events_y = events_y[mask] - yoff
    events_t = events_t[mask]
    events_p = events_p[mask]

    return events_t, events_x, events_y, events_p


class DatasetTrain(Dataset):
    def __init__(self, dt, dataset_file, train_dir, transform=None, num_bins=10):
        self.transform = transform
        self.dt = dt
        self.num_bins = num_bins
        self.dataset_file = dataset_file

        with h5py.File(dataset_file, 'r') as d_set:
            self.length = d_set['davis']['left']['image_raw'].shape[0]

    def __getitem__(self, index):
        # Empty tensor fallback: [Channels, H, W, Time] -> [2, 256, 256, num_bins]
        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        gray_0 = torch.zeros(1, 256, 256)

        if index + 100 < self.length and index > 100:
            # 1. Fetch raw events using the helper function
            events_t, events_x, events_y, events_p = get_raw_events_for_window(self.dataset_file, index, self.dt)
            
            # Handle empty windows
            if len(events_t) == 0:
                return voxel_0, gray_0, gray_0

            # 2. Generate the Voxel Grid
            # Returns shape: [num_bins * 2, H, W]
            voxel_flat = generate_voxel_grid(
                torch.from_numpy(events_t).float(),
                torch.from_numpy(events_x).float(),
                torch.from_numpy(events_y).float(),
                torch.from_numpy(events_p).float(),
                num_bins=self.num_bins, height=256, width=256
            )
            
            # 3. Reshape and Permute for SNN 
            # [Channels, Height, Width, Time] -> [2, 256, 256, num_bins]
            voxel_tensor = voxel_flat.view(self.num_bins, 2, 256, 256).permute(1, 2, 3, 0)
            
            # Fetch gray images
            with h5py.File(self.dataset_file, 'r') as d_set:
                gray_f_raw = d_set['davis']['left']['image_raw'][index]
                gray_l_raw = d_set['davis']['left']['image_raw'][index + self.dt]
            gray_f = np.uint8(gray_f_raw)
            gray_l = np.uint8(gray_l_raw)

            if self.transform:
                seed = np.random.randint(2147483647)
                
                # Pre-crop the images to 256x256 so spatial transforms match the pre-cropped event grid
                if gray_f.shape == (260, 346):
                    gray_f = gray_f[2:258, 45:301]
                    gray_l = gray_l[2:258, 45:301]
                
                voxel_transformed = torch.zeros_like(voxel_tensor)
                
                # Apply transformations consistently to each time bin and polarity
                for c in range(2):
                    for b in range(self.num_bins):
                        random.seed(seed)
                        torch.manual_seed(seed)
                        slice_cb = voxel_tensor[c, :, :, b].numpy()

                        scale_val = slice_cb.max()
                        if scale_val > 0:
                            normalized_slice = slice_cb / scale_val
                            t_slice = self.transform(normalized_slice)
                            # Restore fractional scale
                            if torch.max(t_slice) > 0:
                                voxel_transformed[c, :, :, b] = scale_val * t_slice / torch.max(t_slice)
                        else:
                            voxel_transformed[c, :, :, b] = self.transform(slice_cb)
                
                voxel_tensor = voxel_transformed
                
                random.seed(seed)
                torch.manual_seed(seed)
                gray_f = self.transform(gray_f)

                random.seed(seed)
                torch.manual_seed(seed)
                gray_l = self.transform(gray_l)
            else:
                if gray_f.shape == (260, 346):
                    gray_f = gray_f[2:258, 45:301]
                    gray_l = gray_l[2:258, 45:301]
                gray_f = torch.from_numpy(gray_f).float().unsqueeze(0)
                gray_l = torch.from_numpy(gray_l).float().unsqueeze(0)

            if torch.max(voxel_tensor) > 0 and torch.max(gray_f) > 0 and torch.max(gray_l) > 0:
                return voxel_tensor, gray_f/torch.max(gray_f), gray_l/torch.max(gray_l)
            else:
                return voxel_0, gray_0, gray_0
        else:
            return voxel_0, gray_0, gray_0

    def __len__(self):
        return self.length


class DatasetTest(Dataset):
    def __init__(self, dt, dataset_file, test_dir, gt_start_time=0, num_bins=10):
        self.dt = dt
        self.num_bins = num_bins
        self.dataset_file = dataset_file
        self.gt_start_time = gt_start_time

        with h5py.File(dataset_file, 'r') as d_set:
            self.gray_image_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
            self.length = d_set['davis']['left']['image_raw'].shape[0]

    def __getitem__(self, index):
        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        
        ts_f = self.gray_image_ts[index]
        ts_l = self.gray_image_ts[index + self.dt] if index + self.dt < self.length else 0.0

        # Skip if timestamp is before ground truth starts
        if ts_f < self.gt_start_time:
            return voxel_0, ts_f, ts_l

        if (index + 20 < self.length) and (index > 20):
            # 1. Fetch raw events using the helper function
            events_t, events_x, events_y, events_p = get_raw_events_for_window(self.dataset_file, index, self.dt)

            # Handle empty windows
            if len(events_t) == 0:
                return voxel_0, ts_f, ts_l

            # 2. Generate the Voxel Grid
            # Returns shape: (2 * num_bins, 256, 256)
            voxel_flat = generate_voxel_grid(
                torch.from_numpy(events_t).float(),
                torch.from_numpy(events_x).float(),
                torch.from_numpy(events_y).float(),
                torch.from_numpy(events_p).float(),
                num_bins=self.num_bins, height=256, width=256
            )

            # 3. Reshape and Permute for SNN 
            # [Channels, Height, Width, Time] -> [2, 256, 256, num_bins]
            voxel_tensor = voxel_flat.view(self.num_bins, 2, 256, 256).permute(1, 2, 3, 0)

            return voxel_tensor, ts_f, ts_l
        else:
            return voxel_0, ts_f, ts_l

    def __len__(self):
        return self.length
