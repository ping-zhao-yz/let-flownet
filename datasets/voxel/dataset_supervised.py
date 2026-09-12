import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset
from datasets.voxel.voxel_grid import events_to_voxel_grid
from loss.metrics import estimate_corresponding_gt_flow

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


class DatasetTestDSEC(Dataset):
    def __init__(self, dt, dataset_file, gt_start_time=0, num_bins=10):
        self.dt = dt
        self.num_bins = num_bins
        self.dataset_file = dataset_file
        self.gt_start_time = gt_start_time
        self.d_set = None

        with h5py.File(dataset_file, 'r') as d_set:
            self.gray_image_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
            self.length = d_set['davis']['left']['image_raw'].shape[0]
            
        self.orig_h, self.orig_w = 480, 640

    def __getitem__(self, index):
        if self.d_set is None:
            self.d_set = h5py.File(self.dataset_file, 'r')

        voxel_0 = torch.zeros(2, self.orig_h, self.orig_w, self.num_bins)

        ts_f = self.gray_image_ts[index]
        ts_l = self.gray_image_ts[index + self.dt] if index + self.dt < self.length else 0.0

        if ts_f < self.gt_start_time:
            return voxel_0, ts_f, ts_l

        if (index + 20 < self.length) and (index > 20):
            try:
                events_packed = get_raw_events_for_window(self.d_set, index, self.dt, 0, 0, self.orig_w, self.orig_h)
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
                height=self.orig_h,
                width=self.orig_w,
                device=torch.device('cpu')
            )

            voxel_tensor = torch.clamp(voxel_tensor, max=5.0) / 5.0
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
        self.transform = transform
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

        voxel_0 = torch.zeros(2, 480, 640, self.num_bins)
        gt_flow_0 = torch.zeros(2, 480, 640)
        mask_0 = torch.zeros(1, 480, 640)

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

            events_x = events[:, 0]
            events_y = events[:, 1]
            events_t = events[:, 2].astype(np.float64)
            events_p = events[:, 3].astype(np.float32)
            events_p = 2*events_p - 1

            mask = (events_x >= 0) & (events_x < 640) & (events_y >= 0) & (events_y < 480)
            events_packed = np.stack([
                events_t[mask],
                events_x[mask],
                events_y[mask],
                events_p[mask]
            ], axis=1)

            if len(events_packed) == 0:
                return voxel_0, gt_flow_0, mask_0

            voxel_tensor = events_to_voxel_grid(
                events_packed,
                num_bins=self.num_bins,
                height=480,
                width=640,
                device=torch.device('cpu')
            )

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

            gt_flow_cropped = gt_flow[:480, :640, :]
            
            valid_mask = np.linalg.norm(gt_flow_cropped, axis=2) > 0
            
            gt_flow_t = torch.from_numpy(gt_flow_cropped).permute(2, 0, 1).float()
            mask_t = torch.from_numpy(valid_mask).unsqueeze(0).float()
            
            voxel_tensor = torch.clamp(voxel_tensor, max=5.0) / 5.0

            if random.random() > 0.5:
                voxel_tensor = torch.flip(voxel_tensor, dims=[3])
                gt_flow_t = torch.flip(gt_flow_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                gt_flow_t[0, :, :] *= -1.0 

            if random.random() > 0.5:
                voxel_tensor = torch.flip(voxel_tensor, dims=[2])
                gt_flow_t = torch.flip(gt_flow_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
                gt_flow_t[1, :, :] *= -1.0 

            if random.random() > 0.5:
                voxel_tensor = torch.flip(voxel_tensor, dims=[0, 1])
                gt_flow_t *= -1.0

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
