import os
import h5py
import hdf5plugin
import numpy as np
import imageio.v3 as iio
import cv2
import glob

# Set your paths here
DSEC_ROOT = "/path/to/DSEC"
OUTPUT_DIR = "/path/to/DSEC_formatted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_sequence(seq_name):
    print(f"Processing {seq_name}...")
    seq_path = os.path.join(DSEC_ROOT, seq_name)

    # 1. Exact Event Path
    events_file = os.path.join(seq_path, "events", "left", "events.h5")

    # 2. Exact Image Paths
    img_dir = os.path.join(seq_path, "images", "left", "rectified")

    # DSEC timestamp files sometimes drop the .txt extension when unzipped
    img_ts_file = os.path.join(seq_path, "images", "timestamps.txt")
    if not os.path.exists(img_ts_file):
        img_ts_file = os.path.join(seq_path, "images", "timestamps")

    if not os.path.exists(events_file) or not os.path.exists(img_dir) or not os.path.exists(img_ts_file):
        print(f"Skipping {seq_name}: Missing core event or image data.")
        return

    # --- Load Events and align timestamps ---
    with h5py.File(events_file, 'r') as f_ev:
        p = np.array(f_ev['events/p'])
        t = np.array(f_ev['events/t'])
        x = np.array(f_ev['events/x'])
        y = np.array(f_ev['events/y'])
        t_offset = f_ev['t_offset'][()]

    # Align event time to image time
    t = t + t_offset

    # Pack events into [x, y, t, p] as your dataset_dtx.py expects
    events_packed = np.column_stack((x, y, t, p))

    # --- Load Images and Timestamps ---
    img_timestamps = np.loadtxt(img_ts_file, dtype=np.float64)

    # Find all PNGs in the image directory
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    images_raw = []
    for img_p in img_files:
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        images_raw.append(img)
    images_raw = np.stack(images_raw, axis=0)

    # --- Calculate image_raw_event_inds (Mapping images to event indices) ---
    event_inds = np.searchsorted(t, img_timestamps)

    # --- Write Unified Data HDF5 ---
    out_data_path = os.path.join(OUTPUT_DIR, f"{seq_name}_data.hdf5")
    with h5py.File(out_data_path, 'w') as f_out:
        grp = f_out.create_group('davis/left')
        grp.create_dataset('events', data=events_packed, compression="lzf")
        grp.create_dataset('image_raw', data=images_raw, compression="lzf")
        grp.create_dataset('image_raw_ts', data=img_timestamps)
        grp.create_dataset('image_raw_event_inds', data=event_inds)

    print(f"  -> Saved {out_data_path}")

    # --- Process Ground Truth Flow (if available) ---
    flow_dir = os.path.join(seq_path, "flow", "forward")
    flow_ts_file = os.path.join(seq_path, "flow", "forward_timestamps.txt")
    if not os.path.exists(flow_ts_file):
        flow_ts_file = os.path.join(seq_path, "flow", "forward_timestamps")

    if os.path.exists(flow_dir) and os.path.exists(flow_ts_file):
        # DSEC flow timestamps are comma-separated [start_time, end_time]
        flow_timestamps_raw = np.loadtxt(flow_ts_file, delimiter=',', dtype=np.float64)

        # Your pipeline expects a 1D array of timestamps, so extract the first column (start_time)
        if flow_timestamps_raw.ndim == 2:
            flow_timestamps = flow_timestamps_raw[:, 0]
        else:
            flow_timestamps = flow_timestamps_raw

        flow_files = sorted(glob.glob(os.path.join(flow_dir, "*.png")))

        flow_dist = []
        # DSEC flow is at 10Hz, parse the 16-bit PNGs
        for flow_p in flow_files:
            flow_16bit = iio.imread(flow_p, extension=".png")

            # Convert DSEC format to standard optical flow
            flow_x = (flow_16bit[:, :, 0].astype(np.float32) - 2**15) / 128.0
            flow_y = (flow_16bit[:, :, 1].astype(np.float32) - 2**15) / 128.0

            # Stack into [2, H, W] for your validate() function
            flow_dist.append(np.stack((flow_x, flow_y), axis=0))

        flow_dist = np.stack(flow_dist, axis=0)

        out_gt_path = os.path.join(OUTPUT_DIR, f"{seq_name}_gt.hdf5")
        with h5py.File(out_gt_path, 'w') as f_gt:
            grp_gt = f_gt.create_group('davis/left')
            grp_gt.create_dataset('flow_dist', data=flow_dist, compression="lzf")
            grp_gt.create_dataset('flow_dist_ts', data=flow_timestamps)

        print(f"  -> Saved {out_gt_path}")

# Iterate over all extracted sequences
for seq in os.listdir(DSEC_ROOT):
    if os.path.isdir(os.path.join(DSEC_ROOT, seq)):
        process_sequence(seq)
