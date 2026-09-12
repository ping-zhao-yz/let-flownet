import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset
from datasets.voxel.voxel_grid import events_to_voxel_grid

def get_raw_events_for_window(d_set, index, dt, xoff, yoff, crop_w, crop_h):
    event_inds = d_set['davis']['left']['image_raw_event_inds']
    if index + dt >= len(event_inds):
        return None
        
    start_idx = event_inds[index]
    end_idx = event_inds[index + dt]
    if start_idx >= end_idx:
        return None
        
    events = d_set['davis']['left']['events'][start_idx:end_idx]

    events_x = events[:, 0]
    events_y = events[:, 1]
    events_t = events[:, 2].astype(np.float64)
    events_p = events[:, 3].astype(np.float32)
    events_p = 2*events_p - 1

    mask = (events_x >= xoff) & (events_x < xoff + crop_w) & (events_y >= yoff) & (events_y < yoff + crop_h)

    events_packed = np.stack([
        events_t[mask],
        events_x[mask] - xoff,
        events_y[mask] - yoff,
        events_p[mask]
    ], axis=1)

    return events_packed


class DatasetTrain(Dataset):
    def __init__(self, dt, dataset_file, transform=None, num_bins=10):
        self.dt = dt
        self.dataset_file = dataset_file
        self.transform = transform
        self.num_bins = num_bins
        self.d_set = None

        with h5py.File(dataset_file, 'r') as d_set:
            self.length = d_set['davis']['left']['image_raw'].shape[0]

    def __getitem__(self, index):
        if self.d_set is None:
            self.d_set = h5py.File(self.dataset_file, 'r')

        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        gray_0 = torch.zeros(1, 256, 256)

        if index + 100 < self.length and index > 100:
            try:
                gray_f_raw = self.d_set['davis']['left']['image_raw'][index]
                gray_l_raw = self.d_set['davis']['left']['image_raw'][index + self.dt]
                
                gray_f = np.squeeze(np.asarray(gray_f_raw, dtype=np.uint8))
                gray_l = np.squeeze(np.asarray(gray_l_raw, dtype=np.uint8))
                
                orig_h, orig_w = gray_f.shape[0], gray_f.shape[1]
                
                xoff = random.randint(0, max(0, orig_w - 256))
                yoff = random.randint(0, max(0, orig_h - 256))

                events_packed = get_raw_events_for_window(self.d_set, index, self.dt, xoff, yoff, 256, 256)
            except (OSError, IndexError, Exception) as e:
                print(f"WARNING: HDF5 image/event read failed at index {index}. Recovering handle.")
                try:
                    self.d_set.close()
                except Exception:
                    pass
                self.d_set = None
                return voxel_0, gray_0, gray_0

            if events_packed is None or len(events_packed) == 0:
                return voxel_0, gray_0, gray_0

            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=256,
                width=256,
                device=torch.device('cpu')
            )

            voxel_flat = voxel_tensor.view(2 * self.num_bins, 256, 256)
            
            gray_f = gray_f[yoff:yoff+256, xoff:xoff+256]
            gray_l = gray_l[yoff:yoff+256, xoff:xoff+256]
            
            gray_f_t = torch.from_numpy(gray_f).float().unsqueeze(0)
            gray_l_t = torch.from_numpy(gray_l).float().unsqueeze(0)
            
            gray_f_t = gray_f_t / 255.0
            gray_l_t = gray_l_t / 255.0

            combo = torch.cat([voxel_flat, gray_f_t, gray_l_t], dim=0)

            combo_transformed = self.transform(combo) if self.transform else combo

            voxel_transformed_flat = combo_transformed[0:2*self.num_bins]
            
            gray_f_final = combo_transformed[-2:-1]
            gray_l_final = combo_transformed[-1:]

            voxel_transformed_flat = torch.clamp(voxel_transformed_flat, max=5.0) / 5.0

            voxel_tensor = voxel_transformed_flat.view(self.num_bins, 2, 256, 256)

            if random.random() > 0.5:
                voxel_tensor = torch.flip(voxel_tensor, dims=[0, 1])
                
                temp_gray = gray_f_final
                gray_f_final = gray_l_final
                gray_l_final = temp_gray

            voxel_tensor = voxel_tensor.permute(1, 2, 3, 0)

            if torch.max(voxel_tensor) > 0 and torch.max(gray_f_final) > 0 and torch.max(gray_l_final) > 0:
                return voxel_tensor, gray_f_final, gray_l_final
            else:
                return voxel_0, gray_0, gray_0
        else:
            return voxel_0, gray_0, gray_0

    def __len__(self):
        return self.length


class DatasetTest(Dataset):
    def __init__(self, dt, dataset_file, gt_start_time=0, num_bins=10):
        self.dt = dt
        self.num_bins = num_bins
        self.dataset_file = dataset_file
        self.gt_start_time = gt_start_time
        self.d_set = None

        with h5py.File(dataset_file, 'r') as d_set:
            self.gray_image_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
            image_shape = d_set['davis']['left']['image_raw'].shape
            self.length = image_shape[0]
            if len(image_shape) >= 3:
                self.orig_h, self.orig_w = image_shape[1], image_shape[2]
            else:
                self.orig_h, self.orig_w = 260, 346

    def __getitem__(self, index):
        if self.d_set is None:
            self.d_set = h5py.File(self.dataset_file, 'r')

        out_h = 256
        out_w = 256
        voxel_0 = torch.zeros(2, out_h, out_w, self.num_bins)

        ts_f = self.gray_image_ts[index]
        ts_l = self.gray_image_ts[index + self.dt] if index + self.dt < self.length else 0.0

        if ts_f < self.gt_start_time:
            return voxel_0, ts_f, ts_l

        if (index + 20 < self.length) and (index > 20):
            try:
                crop_w, crop_h = 256, 256
                xoff = max(0, (self.orig_w - crop_w) // 2)
                yoff = max(0, (self.orig_h - crop_h) // 2)

                events_packed = get_raw_events_for_window(self.d_set, index, self.dt, xoff, yoff, crop_w, crop_h)
            except (OSError, IndexError, Exception) as e:
                print(f"WARNING: HDF5 events read failed at index {index}. Recovering handle.")
                try:
                    self.d_set.close()
                except Exception:
                    pass
                self.d_set = None
                return voxel_0, ts_f, ts_l
            
            if events_packed is None or len(events_packed) == 0:
                return voxel_0, ts_f, ts_l

            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=crop_h,
                width=crop_w,
                device=torch.device('cpu')
            )

            voxel_tensor = torch.clamp(voxel_tensor, max=5.0) / 5.0
            
            voxel_tensor = voxel_tensor.permute(1, 2, 3, 0)

            return voxel_tensor, ts_f, ts_l
        else:
            return voxel_0, ts_f, ts_l

    def __len__(self):
        return self.length
