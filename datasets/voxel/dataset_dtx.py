import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset
from datasets.voxel.voxel_grid import events_to_voxel_grid
from loss.multiscaleloss import estimate_corresponding_gt_flow

def get_raw_events_for_window(d_set, index, dt, xoff, yoff, orig_w, orig_h):
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

    # Spatial cropping (Strict 256x256 window)
    mask = (events_x >= xoff) & (events_x < xoff + 256) & (events_y >= yoff) & (events_y < yoff + 256)

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

        # 1. Update the shape to match our new 2-channel grid (Keep it on CPU)
        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        gray_0 = torch.zeros(1, 256, 256)

        if index + 100 < self.length and index > 100:
            # Fetch gray images first to dynamically extract orig_h, orig_w
            try:
                gray_f_raw = self.d_set['davis']['left']['image_raw'][index]
                gray_l_raw = self.d_set['davis']['left']['image_raw'][index + self.dt]
                
                gray_f = np.squeeze(np.asarray(gray_f_raw, dtype=np.uint8))
                gray_l = np.squeeze(np.asarray(gray_l_raw, dtype=np.uint8))
                
                orig_h, orig_w = gray_f.shape[0], gray_f.shape[1]
                
                # Generate dynamic random offsets for spatial augmentation
                xoff = random.randint(0, max(0, orig_w - 256))
                yoff = random.randint(0, max(0, orig_h - 256))

                # 1. Fetch raw events using the exact dynamic random crop
                events_packed = get_raw_events_for_window(self.d_set, index, self.dt, xoff, yoff, orig_w, orig_h)
            except (OSError, IndexError, Exception) as e:
                print(f"WARNING: HDF5 image/event read failed at index {index}. Recovering handle.")
                try:
                    self.d_set.close()
                except Exception:
                    pass
                self.d_set = None
                return voxel_0, gray_0, gray_0

            # Handle empty windows
            if events_packed is None or len(events_packed) == 0:
                return voxel_0, gray_0, gray_0

            # 2. Generate the Voxel Grid
            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=256,
                width=256,
                device=torch.device('cpu')
            )

            # Flatten the 4D tensor to 3D for concatenation
            voxel_flat = voxel_tensor.view(2 * self.num_bins, 256, 256)
            
            # Crop images dynamically using the exact same offsets
            gray_f = gray_f[yoff:yoff+256, xoff:xoff+256]
            gray_l = gray_l[yoff:yoff+256, xoff:xoff+256]
            
            gray_f_t = torch.from_numpy(gray_f).float().unsqueeze(0)
            gray_l_t = torch.from_numpy(gray_l).float().unsqueeze(0)
            
            # Normalize images
            gray_f_t = gray_f_t / 255.0
            gray_l_t = gray_l_t / 255.0

            # Concatenate on the CPU safely
            combo = torch.cat([voxel_flat, gray_f_t, gray_l_t], dim=0)

            # Apply spatial transformation
            combo_transformed = self.transform(combo)

            voxel_transformed_flat = combo_transformed[0:2*self.num_bins]
            
            gray_f_final = combo_transformed[-2:-1]
            gray_l_final = combo_transformed[-1:]

            # Compress outliers and normalize
            voxel_transformed_flat = torch.clamp(voxel_transformed_flat, max=5.0) / 5.0

            # Reshape back to 4D [num_bins, 2, H, W] so we can safely manipulate Time
            voxel_tensor = voxel_transformed_flat.view(self.num_bins, 2, 256, 256)

            # ---> Temporal Reversal Augmentation <---
            if random.random() > 0.5:
                # Time is dim=0, Polarity is dim=1 at this stage - [num_bins, 2, H, W]
                # We must flip BOTH to maintain true event camera physics!
                voxel_tensor = torch.flip(voxel_tensor, dims=[0, 1])
                
                # Swap the photometric target images
                temp_gray = gray_f_final
                gray_f_final = gray_l_final
                gray_l_final = temp_gray

            # Move Time to the last dimension for the SNN [2, 256, 256, num_bins]
            voxel_tensor = voxel_tensor.permute(1, 2, 3, 0)

            # Final Return
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

        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        
        ts_f = self.gray_image_ts[index]
        ts_l = self.gray_image_ts[index + self.dt] if index + self.dt < self.length else 0.0

        # Skip if timestamp is before ground truth starts
        if ts_f < self.gt_start_time:
            return voxel_0, ts_f, ts_l

        if (index + 20 < self.length) and (index > 20):
            try:
                # 1. Fetch raw events using strict mathematical center crop
                xoff = max(0, (self.orig_w - 256) // 2)
                yoff = max(0, (self.orig_h - 256) // 2)
                events_packed = get_raw_events_for_window(self.d_set, index, self.dt, xoff, yoff, self.orig_w, self.orig_h)
            except (OSError, IndexError, Exception) as e:
                print(f"WARNING: HDF5 events read failed at index {index}. Recovering handle.")
                try:
                    self.d_set.close()
                except Exception:
                    pass
                self.d_set = None
                return voxel_0, ts_f, ts_l
            
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

class DatasetTrainDSEC_Supervised(Dataset):
    def __init__(self, dt, dataset_file, gt_file, transform=None, num_bins=10):
        self.dt = dt
        self.dataset_file = dataset_file
        self.gt_file = gt_file
        self.transform = transform # Kept for API compatibility, but unused as we manually augment
        self.num_bins = num_bins
        self.d_set = None
        self.d_label = None

        with h5py.File(dataset_file, 'r') as d_set:
            self.length = len(d_set['davis']['left']['image_raw_ts'])
            
        with h5py.File(gt_file, 'r') as d_label:
            self.gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
            
        self.orig_h, self.orig_w = 480, 640

    def __getitem__(self, index):
        if self.d_set is None:
            self.d_set = h5py.File(self.dataset_file, 'r')
        if self.d_label is None:
            self.d_label = h5py.File(self.gt_file, 'r')

        voxel_0 = torch.zeros(2, 256, 256, self.num_bins)
        gt_flow_0 = torch.zeros(2, 256, 256)
        mask_0 = torch.zeros(1, 256, 256)

        if index + 100 < self.length and index > 100:
            try:
                ts_f = np.float64(self.d_set['davis']['left']['image_raw_ts'][index])
                ts_l = np.float64(self.d_set['davis']['left']['image_raw_ts'][index + self.dt])
                
                event_inds = self.d_set['davis']['left']['image_raw_event_inds']
                start_idx = event_inds[index]
                end_idx = event_inds[index + self.dt]
                if start_idx >= end_idx:
                    return voxel_0, gt_flow_0, mask_0
                    
                events = self.d_set['davis']['left']['events'][start_idx:end_idx]
            except (OSError, IndexError, Exception) as e:
                print(f"WARNING: HDF5 events read failed at index {index}. Recovering handle.")
                try:
                    self.d_set.close()
                except Exception:
                    pass
                self.d_set = None
                return voxel_0, gt_flow_0, mask_0

            xoff = random.randint(0, max(0, self.orig_w - 256))
            yoff = random.randint(0, max(0, self.orig_h - 256))

            events_x = events[:, 0]
            events_y = events[:, 1]
            events_t = events[:, 2].astype(np.float64)
            events_p = events[:, 3].astype(np.float32)
            events_p = 2*events_p - 1

            mask = (events_x >= xoff) & (events_x < xoff + 256) & (events_y >= yoff) & (events_y < yoff + 256)
            events_packed = np.stack([
                events_t[mask],
                events_x[mask] - xoff,
                events_y[mask] - yoff,
                events_p[mask]
            ], axis=1)

            if len(events_packed) == 0:
                return voxel_0, gt_flow_0, mask_0

            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=256,
                width=256,
                device=torch.device('cpu')
            )

            # Safe Targeted HDF5 Read with Recovery Shield
            try:
                start_flow_idx = max(0, np.searchsorted(self.gt_ts_temp, ts_f, side="right") - 1)
                end_flow_idx = min(len(self.gt_ts_temp), np.searchsorted(self.gt_ts_temp, ts_l, side="right") + 2)
                
                gt_temp_slice = np.float32(self.d_label['davis']['left']['flow_dist'][start_flow_idx:end_flow_idx])
                gt_ts_slice = self.gt_ts_temp[start_flow_idx:end_flow_idx]

                if len(gt_ts_slice) < 2 or len(gt_temp_slice) < 2:
                    return voxel_0, gt_flow_0, mask_0

            except (OSError, IndexError, Exception) as e:
                print(f"WARNING: HDF5 flow read failed at index {index}. Recovering handle.")
                try:
                    self.d_label.close()
                except Exception:
                    pass
                self.d_label = None
                return voxel_0, gt_flow_0, mask_0
            
            u_gt_all = gt_temp_slice[:, 0, :, :].copy()
            v_gt_all = gt_temp_slice[:, 1, :, :].copy()
            
            invalid_mask = (u_gt_all == -256.0) & (v_gt_all == -256.0)
            u_gt_all[invalid_mask] = 0.0
            v_gt_all[invalid_mask] = 0.0

            u_gt, v_gt = estimate_corresponding_gt_flow(
                u_gt_all, v_gt_all, gt_ts_slice, ts_f, ts_l)
            gt_flow = np.stack((u_gt, v_gt), axis=2)

            gt_flow_cropped = gt_flow[yoff:yoff+256, xoff:xoff+256, :]
            
            valid_mask = np.linalg.norm(gt_flow_cropped, axis=2) > 0
            
            gt_flow_t = torch.from_numpy(gt_flow_cropped).permute(2, 0, 1).float()
            mask_t = torch.from_numpy(valid_mask).unsqueeze(0).float()
            
            voxel_tensor = torch.clamp(voxel_tensor, max=5.0) / 5.0
            
            # voxel_tensor is [2, 256, 256, num_bins] from events_to_voxel_grid
            # Permute to [num_bins, 2, H, W] for easier augmentation along H/W
            voxel_tensor = voxel_tensor.permute(3, 0, 1, 2)

            # --- Strict Vector-Aware Spatial Augmentation ---
            if random.random() > 0.5: # Horizontal Flip
                voxel_tensor = torch.flip(voxel_tensor, dims=[3])
                gt_flow_t = torch.flip(gt_flow_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                gt_flow_t[0, :, :] *= -1.0 # Invert U

            if random.random() > 0.5: # Vertical Flip
                voxel_tensor = torch.flip(voxel_tensor, dims=[2])
                gt_flow_t = torch.flip(gt_flow_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
                gt_flow_t[1, :, :] *= -1.0 # Invert V

            # --- Temporal Reversal Augmentation ---
            if random.random() > 0.5:
                # Flip Time (dim=0) and Polarity (dim=1)
                voxel_tensor = torch.flip(voxel_tensor, dims=[0, 1])
                # Invert both U and V
                gt_flow_t *= -1.0

            # Move Time to the last dimension for the SNN: [2, 256, 256, num_bins]
            voxel_tensor = voxel_tensor.permute(1, 2, 3, 0)
            
            event_mask = (torch.sum(voxel_tensor, dim=3) > 0).float()
            event_mask_2d = torch.sum(event_mask, dim=0, keepdim=True) > 0
            
            final_mask = mask_t * event_mask_2d.float()

            if torch.max(voxel_tensor) > 0 and final_mask.sum() > 0:
                return voxel_tensor, gt_flow_t, final_mask
            else:
                return voxel_0, gt_flow_0, mask_0

        else:
            return voxel_0, gt_flow_0, mask_0

    def __len__(self):
        return self.length
