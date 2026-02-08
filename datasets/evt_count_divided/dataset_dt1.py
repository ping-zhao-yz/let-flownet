import torch
import numpy as np
import h5py
import random
from torch.utils.data import Dataset


class DatasetTrain(Dataset):
    def __init__(self, dataset_file, train_dir, transform=None):
        self.transform = transform

        self.x = 260
        self.y = 346

        self.split = 10
        self.half_split = int(self.split/2)

        self.train_dir = train_dir

        d_set = h5py.File(dataset_file, 'r')
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        event_0 = torch.zeros(256, 256, self.half_split)
        gray_0 = torch.zeros(1, 256, 256)

        if index + 100 < self.length and index > 100:
            aa = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)
            bb = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)
            cc = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)
            dd = np.zeros((self.x, self.y, self.half_split), dtype=np.uint8)

            im_onoff = np.load(self.train_dir + '/event_data/' +
                               str(int(index + 1))+'.npy')

            # Data mapping between raw events and gray images:
            # events[0][2]              1504645177.4254043
            # event_inds[0]             126
            # events[event_inds[0]][2]  1504645177.4490044
            # image_raw_ts[0]           1504645177.4490874
            # former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off, former_gray, latter_gray = data

            aa[:, :, :] = im_onoff[0, :, :, 0:5]
            bb[:, :, :] = im_onoff[1, :, :, 0:5]
            cc[:, :, :] = im_onoff[0, :, :, 5:10]
            dd[:, :, :] = im_onoff[1, :, :, 5:10]

            gray_f = np.uint8(
                np.load(self.train_dir + '/gray_image/' + str(int(index))+'.npy'))
            gray_l = np.uint8(
                np.load(self.train_dir + '/gray_image/' + str(int(index + 1))+'.npy'))

            if self.transform:
                seed = np.random.randint(2147483647)

                aaa = torch.zeros(256, 256, int(aa.shape[2]))
                bbb = torch.zeros(256, 256, int(bb.shape[2]))
                ccc = torch.zeros(256, 256, int(cc.shape[2]))
                ddd = torch.zeros(256, 256, int(dd.shape[2]))

                for p in range(self.half_split):
                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_a = aa[:, :, p].max()
                    aaa[:, :, p] = self.transform(aa[:, :, p])
                    if torch.max(aaa[:, :, p]) > 0:
                        aaa[:, :, p] = scale_a * aaa[:, :, p] / \
                            torch.max(aaa[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_b = bb[:, :, p].max()
                    bbb[:, :, p] = self.transform(bb[:, :, p])
                    if torch.max(bbb[:, :, p]) > 0:
                        bbb[:, :, p] = scale_b * bbb[:, :, p] / \
                            torch.max(bbb[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_c = cc[:, :, p].max()
                    ccc[:, :, p] = self.transform(cc[:, :, p])
                    if torch.max(ccc[:, :, p]) > 0:
                        ccc[:, :, p] = scale_c * ccc[:, :, p] / \
                            torch.max(ccc[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_d = dd[:, :, p].max()
                    ddd[:, :, p] = self.transform(dd[:, :, p])
                    if torch.max(ddd[:, :, p]) > 0:
                        ddd[:, :, p] = scale_d * ddd[:, :, p] / \
                            torch.max(ddd[:, :, p])

                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                gray_f = self.transform(gray_f)

                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                gray_l = self.transform(gray_l)

            if torch.max(aaa) > 0 and torch.max(bbb) > 0 and torch.max(ccc) > 0 and torch.max(ddd) > 0 and torch.max(gray_f) > 0 and torch.max(gray_l) > 0:
                return aaa, bbb, ccc, ddd, gray_f/torch.max(gray_f), gray_l/torch.max(gray_l)
            else:
                return event_0, event_0, event_0, event_0, gray_0, gray_0
        else:
            return event_0, event_0, event_0, event_0, gray_0, gray_0

    def __len__(self):
        return self.length


class DatasetTest(Dataset):
    def __init__(self, dataset_file, test_dir):
        self.xoff = 45
        self.yoff = 2

        self.split = 10
        self.half_split = int(self.split / 2)

        self.test_dir = test_dir

        d_set = h5py.File(dataset_file, 'r')
        self.gray_image_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        event_0 = np.zeros((256, 256, self.half_split), dtype=np.uint8)
        gray_0 = np.zeros((self.gray_image_ts[index].shape), dtype=np.uint8)

        if (index + 20 < self.length) and (index > 20):
            aa = np.zeros((256, 256, self.half_split), dtype=np.uint8)
            bb = np.zeros((256, 256, self.half_split), dtype=np.uint8)
            cc = np.zeros((256, 256, self.half_split), dtype=np.uint8)
            dd = np.zeros((256, 256, self.half_split), dtype=np.uint8)

            im_onoff = np.load(self.test_dir + '/event_data/' +
                               str(int(index + 1)) + '.npy')

            aa[:, :, :] = im_onoff[0, self.yoff:-self.yoff,
                                   self.xoff:-self.xoff, 0:5].astype(float)
            bb[:, :, :] = im_onoff[1, self.yoff:-self.yoff,
                                   self.xoff:-self.xoff, 0:5].astype(float)
            cc[:, :, :] = im_onoff[0, self.yoff:-self.yoff,
                                   self.xoff:-self.xoff, 5:10].astype(float)
            dd[:, :, :] = im_onoff[1, self.yoff:-self.yoff,
                                   self.xoff:-self.xoff, 5:10].astype(float)

            return aa, bb, cc, dd, self.gray_image_ts[index], self.gray_image_ts[index + 1]
        else:
            return event_0, event_0, event_0, event_0, gray_0, gray_0

    def __len__(self):
        return self.length
