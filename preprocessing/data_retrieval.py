import numpy as np
import h5py
import cv2


def main():

    # loadEventFrames()

    loadRawEvents()


def loadEventFrames():

    aa = np.zeros((256, 256, 5), dtype=np.uint8)
    bb = np.zeros((256, 256, 5), dtype=np.uint8)
    cc = np.zeros((256, 256, 5), dtype=np.uint8)
    dd = np.zeros((256, 256, 5), dtype=np.uint8)

    # shape: 2 x 260 x 346 x 10
    event_group = np.load(
        '../../dataset/Event/mvsec/preprocessed/indoor_flying1/event_data/' + '0.npy')

    # shape: 260 x 346
    gray_image = np.load(
        '../../dataset/Event/mvsec/preprocessed/indoor_flying1/gray_image/' + '0.npy')

    aa[:, :, :] = event_group[0, 2:-2, 45:-45, 0:5].astype(float)
    bb[:, :, :] = event_group[1, 2:-2, 45:-45, 0:5].astype(float)
    cc[:, :, :] = event_group[0, 2:-2, 45:-45, 5:10].astype(float)
    dd[:, :, :] = event_group[1, 2:-2, 45:-45, 5:10].astype(float)
    print(aa.shape)


def loadRawEvents():

    indoor_flying1 = h5py.File(
        "../../dataset/Event/mvsec/original/indoor_flying1/indoor_flying1_data.hdf5", 'r')

    # https://daniilidis-group.github.io/mvsec/
    # https://pythonnumericalmethods.berkeley.edu/notebooks/chapter11.05-HDF5-Files.html

    # dataset structure of 'davis' group and subgroups
    # - left
    #   - events
    #   - image_raw
    #   - image_raw_event_inds
    #   - image_raw_ts
    #   - imu
    #   - imu_ts
    # - right
    #   - events
    #   - image_raw
    #   - image_raw_event_inds
    #   - image_raw_ts
    #   - imu
    #   - imu_ts

    # (2206, 260, 346)
    image_raw = indoor_flying1['davis']['left']['image_raw']
    # (2206,)
    image_raw_ts = indoor_flying1['davis']['left']['image_raw_ts']
    # (14071304, 4)
    events = indoor_flying1['davis']['left']['events']
    # (2206,)
    event_inds = indoor_flying1['davis']['left']['image_raw_event_inds']

    # https://www.geeksforgeeks.org/python-opencv-cv2-imshow-method/
    cv2.imshow('Gray Image', image_raw[0])
    cv2.waitKey(0)

    cv2.destroyAllWindows()


# gray_image_0 = torch.sum(torch.sum(gray_image_0, 0), 2)

# spike_image = image_raw[0]
# spike_image[spike_image > 0] = 255

# cv2.imshow('Spike Image', np.array(spike_image, dtype=np.uint8))


if __name__ == "__main__":
    main()
