import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset
from datasets.voxel.voxel_grid import events_to_voxel_grid

def get_raw_events_for_window(dataset_file, index, dt, xoff=45, yoff=2, orig_w=346, orig_h=260):
    with h5py.File(dataset_file, 'r') as d_set:
        event_inds = d_set['davis']['left']['image_raw_event_inds']
        if index + dt >= len(event_inds):
            return None
            
        start_idx = event_inds[index]
        end_idx = event_inds[index + dt]
        if start_idx >= end_idx:
            return None
            
        events = d_set['davis']['left']['events'][start_idx:end_idx]

    # Map to [N, 4] where: 0=t, 1=x, 2=y, 3=p
    # Based testing, raw data is [x, y, t, p]
    events_x = events[:, 0]
    events_y = events[:, 1]
    events_t = events[:, 2].astype(np.float64)
    events_p = events[:, 3].astype(np.float32)
    events_p = 2*events_p - 1

    # Spatial cropping
    mask = (events_x >= xoff) & (events_x < orig_w - xoff) & (events_y >= yoff) & (events_y < orig_h - yoff)
    
    events_packed = np.stack([
        events_t[mask],
        events_x[mask] - xoff,
        events_y[mask] - yoff,
        events_p[mask]
    ], axis=1)

    return events_packed

class DatasetTrain(Dataset):
    def __init__(self, dt, dataset_file, transform=None, is_fine_tune=False, num_bins=10, teacher_train_dir=None):
        self.dt = dt
        self.dataset_file = dataset_file
        self.transform = transform
        self.is_fine_tune = is_fine_tune
        self.num_bins = num_bins
        self.teacher_train_dir = teacher_train_dir
        
        if self.teacher_train_dir is not None:
            self.x = 260
            self.y = 346
            self.split = 10
            self.half_split = int(self.split/2)
            self.range = int(self.half_split * self.dt)

        with h5py.File(dataset_file, 'r') as d_set:
            self.length = d_set['davis']['left']['image_raw'].shape[0]

    def __getitem__(self, index):
        # 1. Update the shape to match our new 2-channel grid (Keep it on CPU)
        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        gray_0 = torch.zeros(1, 256, 256)
        
        if self.teacher_train_dir is not None:
            event_0 = torch.zeros(256, 256, self.range)

        if index + 100 < self.length and index > 100:
            # 1. Fetch raw events using the helper function
            events_packed = get_raw_events_for_window(self.dataset_file, index, self.dt)

            # Handle empty windows
            if events_packed is None or len(events_packed) == 0:
                if self.teacher_train_dir is not None:
                    return voxel_0, event_0, event_0, event_0, event_0, gray_0, gray_0
                return voxel_0, gray_0, gray_0

            # 2. Generate the Voxel Grid
            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=256,
                width=256,
                device=torch.device('cpu')
            )

            # Fetch gray images
            with h5py.File(self.dataset_file, 'r') as d_set:
                gray_f_raw = d_set['davis']['left']['image_raw'][index]
                gray_l_raw = d_set['davis']['left']['image_raw'][index + self.dt]
            gray_f = np.uint8(gray_f_raw)
            gray_l = np.uint8(gray_l_raw)
            
            # ---> LOAD EVT COUNT DATA <---
            if self.teacher_train_dir is not None:
                aa = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
                bb = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
                cc = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
                dd = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
                
                if self.dt == 1:
                    im_onoff = np.load(self.teacher_train_dir + '/event_data/' + str(int(index + 1)) + '.npy')
                    aa[:, :, :] = im_onoff[0, :, :, 0:5]
                    bb[:, :, :] = im_onoff[1, :, :, 0:5]
                    cc[:, :, :] = im_onoff[0, :, :, 5:10]
                    dd[:, :, :] = im_onoff[1, :, :, 5:10]
                else:
                    for k in range(int(self.dt/2)):
                        im_former = np.load(self.teacher_train_dir + '/event_data/' + str(int(index + k + 1)) + '.npy')
                        im_latter = np.load(self.teacher_train_dir + '/event_data/' + str(int(index + self.dt/2 + k + 1)) + '.npy')
                        aa[:, :, self.split * k: self.split * (k + 1)] = im_former[0, :, :, :]
                        bb[:, :, self.split * k: self.split * (k + 1)] = im_former[1, :, :, :]
                        cc[:, :, self.split * k: self.split * (k + 1)] = im_latter[0, :, :, :]
                        dd[:, :, self.split * k: self.split * (k + 1)] = im_latter[1, :, :, :]

            # Flatten the 4D tensor to 3D for concatenation
            voxel_flat = voxel_tensor.view(2 * self.num_bins, 256, 256)
            
            if self.teacher_train_dir is not None:
                aa_t = torch.from_numpy(aa).float().permute(2, 0, 1)
                bb_t = torch.from_numpy(bb).float().permute(2, 0, 1)
                cc_t = torch.from_numpy(cc).float().permute(2, 0, 1)
                dd_t = torch.from_numpy(dd).float().permute(2, 0, 1)
            
            # Keep images on CPU
            if gray_f.shape == (260, 346):
                gray_f = gray_f[2:258, 45:301]
                gray_l = gray_l[2:258, 45:301]
            
            gray_f_t = torch.from_numpy(gray_f).float().unsqueeze(0)
            gray_l_t = torch.from_numpy(gray_l).float().unsqueeze(0)
            
            # Normalize images
            gray_f_t = gray_f_t / 255.0
            gray_l_t = gray_l_t / 255.0

            # Concatenate on the CPU safely
            if self.teacher_train_dir is not None:
                combo = torch.cat([voxel_flat, aa_t, bb_t, cc_t, dd_t, gray_f_t, gray_l_t], dim=0)
            else:
                combo = torch.cat([voxel_flat, gray_f_t, gray_l_t], dim=0)

            # Apply spatial transformation
            combo_transformed = self.transform(combo)

            voxel_transformed_flat = combo_transformed[0:2*self.num_bins]
            if self.teacher_train_dir is not None:
                r = self.range
                offset = 2 * self.num_bins
                aaa = combo_transformed[offset:offset+r]
                bbb = combo_transformed[offset+r:offset+2*r]
                ccc = combo_transformed[offset+2*r:offset+3*r]
                ddd = combo_transformed[offset+3*r:offset+4*r]
            
            gray_f_final = combo_transformed[-2:-1]
            gray_l_final = combo_transformed[-1:]

            # Compress outliers and normalize
            voxel_transformed_flat = torch.clamp(voxel_transformed_flat, max=5.0) / 5.0
            if self.teacher_train_dir is not None:
                aaa = torch.clamp(aaa, max=5.0) / 5.0
                bbb = torch.clamp(bbb, max=5.0) / 5.0
                ccc = torch.clamp(ccc, max=5.0) / 5.0
                ddd = torch.clamp(ddd, max=5.0) / 5.0

            # Reshape back to 4D [num_bins, 2, H, W] so we can safely manipulate Time
            voxel_tensor = voxel_transformed_flat.view(self.num_bins, 2, 256, 256)

            # ---> NEW: Temporal Reversal Augmentation <---
            if (not self.is_fine_tune) and (random.random() > 0.5):
                # Time is dim=0, Polarity is dim=1 at this stage - [num_bins, 2, H, W]
                # We must flip BOTH to maintain true event camera physics!
                voxel_tensor = torch.flip(voxel_tensor, dims=[0, 1])
                
                if self.teacher_train_dir is not None:
                    # For aaa, bbb, ccc, ddd, Time is dim=0 [Bins, H, W]
                    temp_aaa = torch.flip(ddd, dims=[0])
                    temp_bbb = torch.flip(ccc, dims=[0])
                    temp_ccc = torch.flip(bbb, dims=[0])
                    temp_ddd = torch.flip(aaa, dims=[0])
                    aaa, bbb, ccc, ddd = temp_aaa, temp_bbb, temp_ccc, temp_ddd
                
                # Swap the photometric target images
                temp_gray = gray_f_final
                gray_f_final = gray_l_final
                gray_l_final = temp_gray

            # Move Time to the last dimension for the SNN [2, 256, 256, num_bins]
            voxel_tensor = voxel_tensor.permute(1, 2, 3, 0)
            
            if self.teacher_train_dir is not None:
                # Move Time to last dimension [H, W, Bins]
                aaa = aaa.permute(1, 2, 0)
                bbb = bbb.permute(1, 2, 0)
                ccc = ccc.permute(1, 2, 0)
                ddd = ddd.permute(1, 2, 0)

            # Final Return
            if self.teacher_train_dir is not None:
                if torch.max(voxel_tensor) > 0 and torch.max(aaa) > 0 and torch.max(bbb) > 0 and torch.max(ccc) > 0 and torch.max(ddd) > 0 and torch.max(gray_f_final) > 0 and torch.max(gray_l_final) > 0:
                    return voxel_tensor, aaa, bbb, ccc, ddd, gray_f_final, gray_l_final
                else:
                    return voxel_0, event_0, event_0, event_0, event_0, gray_0, gray_0
            else:
                if torch.max(voxel_tensor) > 0 and torch.max(gray_f_final) > 0 and torch.max(gray_l_final) > 0:
                    return voxel_tensor, gray_f_final, gray_l_final
                else:
                    return voxel_0, gray_0, gray_0
        else:
            if self.teacher_train_dir is not None:
                return voxel_0, event_0, event_0, event_0, event_0, gray_0, gray_0
            return voxel_0, gray_0, gray_0

    def __len__(self):
        return self.length


class DatasetTest(Dataset):
    def __init__(self, dt, dataset_file, gt_start_time=0, num_bins=10):
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
            events_packed = get_raw_events_for_window(self.dataset_file, index, self.dt)
            
            # Handle empty windows
            if events_packed is None or len(events_packed) == 0:
                return voxel_0, ts_f, ts_l

            # 2. Generate the Voxel Grid
            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=256,
                width=256,
                device=torch.device('cpu')
            )

            # Compress outliers to preserve normal 1.0 signals and normalize to [0, 1] for Transformer stability
            voxel_tensor = torch.clamp(voxel_tensor, max=5.0) / 5.0
            
            # ---> Move back to CPU <---
            voxel_tensor = voxel_tensor.permute(1, 2, 3, 0)

            return voxel_tensor, ts_f, ts_l
        else:
            return voxel_0, ts_f, ts_l

    def __len__(self):
        return self.length
