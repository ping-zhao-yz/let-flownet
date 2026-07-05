import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset


class DatasetTrain(Dataset):
    def __init__(self, dt, dataset_file, train_dir, transform=None, train_phase=1):
        self.dt = dt
        self.transform = transform
        self.train_phase = train_phase

        self.x = 260
        self.y = 346

        self.split = 10
        self.half_split = int(self.split/2)
        # Dynamically scales the depth of the tensor: 5 bins for dt=1, 20 bins for dt=4
        self.range = int(self.half_split * self.dt)

        self.train_dir = train_dir

        with h5py.File(dataset_file, 'r') as d_set:
            self.length = d_set['davis']['left']['image_raw'].shape[0]

    def __getitem__(self, index):
        event_0 = torch.zeros(256, 256, self.range)
        gray_0 = torch.zeros(1, 256, 256)

        if index + 100 < self.length and index > 100:
            aa = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
            bb = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
            cc = np.zeros((self.x, self.y, self.range), dtype=np.uint8)
            dd = np.zeros((self.x, self.y, self.range), dtype=np.uint8)

            # ---> DATA LOADING DIVERGENCE HANDLED HERE <---
            if self.dt == 1:
                im_onoff = np.load(self.train_dir + '/event_data/' + str(int(index + 1)) + '.npy')
                aa[:, :, :] = im_onoff[0, :, :, 0:5]
                bb[:, :, :] = im_onoff[1, :, :, 0:5]
                cc[:, :, :] = im_onoff[0, :, :, 5:10]
                dd[:, :, :] = im_onoff[1, :, :, 5:10]
            else:
                for k in range(int(self.dt/2)):
                    im_former = np.load(self.train_dir + '/event_data/' + str(int(index + k + 1)) + '.npy')
                    im_latter = np.load(self.train_dir + '/event_data/' + str(int(index + self.dt/2 + k + 1)) + '.npy')
                    aa[:, :, self.split * k: self.split * (k + 1)] = im_former[0, :, :, :]
                    bb[:, :, self.split * k: self.split * (k + 1)] = im_former[1, :, :, :]
                    cc[:, :, self.split * k: self.split * (k + 1)] = im_latter[0, :, :, :]
                    dd[:, :, self.split * k: self.split * (k + 1)] = im_latter[1, :, :, :]

            gray_f = np.uint8(np.load(self.train_dir + '/gray_image/' + str(int(index)) + '.npy'))
            gray_l = np.uint8(np.load(self.train_dir + '/gray_image/' + str(int(index + self.dt)) + '.npy'))

            # ---> UNIFIED OPTIMIZED PIPELINE <---
            if self.transform:
                # 1. Permute to [Channels/Bins, H, W] for PyTorch Transforms
                aa_t = torch.from_numpy(aa).float().permute(2, 0, 1)
                bb_t = torch.from_numpy(bb).float().permute(2, 0, 1)
                cc_t = torch.from_numpy(cc).float().permute(2, 0, 1)
                dd_t = torch.from_numpy(dd).float().permute(2, 0, 1)
                
                gray_f_t = torch.from_numpy(gray_f).float().unsqueeze(0) / 255.0
                gray_l_t = torch.from_numpy(gray_l).float().unsqueeze(0) / 255.0

                # 2. Stack everything into one tensor for a SINGLE lightning-fast transform
                combo = torch.cat([aa_t, bb_t, cc_t, dd_t, gray_f_t, gray_l_t], dim=0)
                combo_transformed = self.transform(combo)

                # 3. Unpack the transformed data using self.range
                r = self.range
                aaa = combo_transformed[0:r]
                bbb = combo_transformed[r:2*r]
                ccc = combo_transformed[2*r:3*r]
                ddd = combo_transformed[3*r:4*r]
                
                gray_f_final = combo_transformed[-2:-1]
                gray_l_final = combo_transformed[-1:]

                # 4. ALIGNED NORMALIZATION: Global clamp to preserve temporal density
                aaa = torch.clamp(aaa, max=5.0) / 5.0
                bbb = torch.clamp(bbb, max=5.0) / 5.0
                ccc = torch.clamp(ccc, max=5.0) / 5.0
                ddd = torch.clamp(ddd, max=5.0) / 5.0

                # 5. ALIGNED TEMPORAL REVERSAL: The 4-Way Cross-Swap
                if (self.train_phase != 4) and (random.random() > 0.5):
                    # Reverse the order of the temporal bins (dim=0 since shape is [Bins, H, W])
                    temp_aaa = torch.flip(ddd, dims=[0])
                    temp_bbb = torch.flip(ccc, dims=[0])
                    temp_ccc = torch.flip(bbb, dims=[0])
                    temp_ddd = torch.flip(aaa, dims=[0])
                    
                    aaa, bbb, ccc, ddd = temp_aaa, temp_bbb, temp_ccc, temp_ddd
                    
                    # Swap the photometric target images
                    temp_gray = gray_f_final
                    gray_f_final = gray_l_final
                    gray_l_final = temp_gray
                
                # 6. Permute back to [H, W, Bins] expected by the Event Count Model
                aaa = aaa.permute(1, 2, 0)
                bbb = bbb.permute(1, 2, 0)
                ccc = ccc.permute(1, 2, 0)
                ddd = ddd.permute(1, 2, 0)

            else:
                # 1. Apply the standard Center Crop manually
                if aa.shape[0] == 260 and aa.shape[1] == 346:
                    aaa = torch.from_numpy(aa[2:258, 45:301, :]).float()
                    bbb = torch.from_numpy(bb[2:258, 45:301, :]).float()
                    ccc = torch.from_numpy(cc[2:258, 45:301, :]).float()
                    ddd = torch.from_numpy(dd[2:258, 45:301, :]).float()
                    
                    gray_f_final = (torch.from_numpy(gray_f[2:258, 45:301]).float() / 255.0).unsqueeze(0)
                    gray_l_final = (torch.from_numpy(gray_l[2:258, 45:301]).float() / 255.0).unsqueeze(0)
                else:
                    aaa = torch.from_numpy(aa).float()
                    bbb = torch.from_numpy(bb).float()
                    ccc = torch.from_numpy(cc).float()
                    ddd = torch.from_numpy(dd).float()
                    
                    gray_f_final = (torch.from_numpy(gray_f).float() / 255.0).unsqueeze(0)
                    gray_l_final = (torch.from_numpy(gray_l).float() / 255.0).unsqueeze(0)

                # 2. ALIGNED NORMALIZATION: Global clamp to preserve temporal density
                aaa = torch.clamp(aaa, max=5.0) / 5.0
                bbb = torch.clamp(bbb, max=5.0) / 5.0
                ccc = torch.clamp(ccc, max=5.0) / 5.0
                ddd = torch.clamp(ddd, max=5.0) / 5.0

                # 3. ALIGNED TEMPORAL REVERSAL: The 4-Way Cross-Swap
                if (self.train_phase != 4) and (random.random() > 0.5):
                    # Time is the LAST dimension here [H, W, Bins], so we flip on dim=2
                    temp_aaa = torch.flip(ddd, dims=[2])
                    temp_bbb = torch.flip(ccc, dims=[2])
                    temp_ccc = torch.flip(bbb, dims=[2])
                    temp_ddd = torch.flip(aaa, dims=[2])
                    
                    aaa, bbb, ccc, ddd = temp_aaa, temp_bbb, temp_ccc, temp_ddd
                    
                    # Swap the photometric target images
                    temp_gray = gray_f_final
                    gray_f_final = gray_l_final
                    gray_l_final = temp_gray

            if torch.max(aaa) > 0 and torch.max(bbb) > 0 and torch.max(ccc) > 0 and torch.max(ddd) > 0 and torch.max(gray_f_final) > 0 and torch.max(gray_l_final) > 0:
                return aaa, bbb, ccc, ddd, gray_f_final, gray_l_final
            else:
                return event_0, event_0, event_0, event_0, gray_0, gray_0
        else:
            return event_0, event_0, event_0, event_0, gray_0, gray_0

    def __len__(self):
        return self.length


class DatasetTest(Dataset):
    def __init__(self, dt, dataset_file, test_dir, gt_start_time=0):
        self.xoff = 45
        self.yoff = 2

        self.dt = dt
        self.split = 10
        self.half_split = int(self.split / 2)
        self.range = int(self.half_split * self.dt)

        self.test_dir = test_dir

        with h5py.File(dataset_file, 'r') as d_set:
            self.gray_image_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
            self.length = d_set['davis']['left']['image_raw'].shape[0]

        self.gt_start_time = gt_start_time

    def __getitem__(self, index):
        event_0 = np.zeros((256, 256, self.range))
        gray_0 = np.zeros((self.gray_image_ts[index].shape))

        # Skip if timestamp is before ground truth starts
        if self.gray_image_ts[index] < self.gt_start_time:
            return event_0, event_0, event_0, event_0, gray_0, gray_0

        if (index + 20 < self.length) and (index > 20):
            aa = np.zeros((256, 256, self.range), dtype=np.float32)
            bb = np.zeros((256, 256, self.range), dtype=np.float32)
            cc = np.zeros((256, 256, self.range), dtype=np.float32)
            dd = np.zeros((256, 256, self.range), dtype=np.float32)

            # ---> DATA LOADING DIVERGENCE HANDLED HERE <---
            if self.dt == 1:
                im_onoff = np.load(self.test_dir + '/event_data/' + str(int(index + 1)) + '.npy')
                aa[:, :, :] = im_onoff[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:5].astype(float)
                bb[:, :, :] = im_onoff[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:5].astype(float)
                cc[:, :, :] = im_onoff[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 5:10].astype(float)
                dd[:, :, :] = im_onoff[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 5:10].astype(float)
            else:
                for k in range(int(self.dt/2)):
                    im_former = np.load(self.test_dir + '/event_data/' + str(int(index + k + 1)) + '.npy')
                    im_latter = np.load(self.test_dir + '/event_data/' + str(int(index + self.dt/2 + k + 1)) + '.npy')
                    aa[:, :, self.split * k: self.split * (k + 1)] = im_former[0, self.yoff: -self.yoff, self.xoff: -self.xoff, :].astype(float)
                    bb[:, :, self.split * k: self.split * (k + 1)] = im_former[1, self.yoff: -self.yoff, self.xoff: -self.xoff, :].astype(float)
                    cc[:, :, self.split * k: self.split * (k + 1)] = im_latter[0, self.yoff: -self.yoff, self.xoff: -self.xoff, :].astype(float)
                    dd[:, :, self.split * k: self.split * (k + 1)] = im_latter[1, self.yoff: -self.yoff, self.xoff: -self.xoff, :].astype(float)

            # ---> FIX: Convert to tensors and permute to [Channels, H, W]
            aaa = torch.from_numpy(aa).float()
            bbb = torch.from_numpy(bb).float()
            ccc = torch.from_numpy(cc).float()
            ddd = torch.from_numpy(dd).float()

            return aaa, bbb, ccc, ddd, self.gray_image_ts[index], self.gray_image_ts[index + self.dt]
        else:
            return event_0, event_0, event_0, event_0, gray_0, gray_0

    def __len__(self):
        return self.length
