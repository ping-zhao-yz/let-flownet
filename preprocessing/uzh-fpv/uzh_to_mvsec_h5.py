import argparse
import os
import h5py
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='uzh-fpv event data conversion from zip to h5 format',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', default=None, help='dataset location')

args = parser.parse_args()

event_data_path = '../../../dataset/Event/uzh-fpv/data/'

def convert_uzh_to_mvsec_h5(data_dir, output_h5):
    events_file = os.path.join(data_dir, 'events.txt')
    images_file = os.path.join(data_dir, 'images.txt')

    print(f"1. Loading events from {events_file} (This may take a minute for 900MB...)")
    # UZH-FPV events.txt format: timestamp x y polarity
    # Using pandas as it is significantly faster than np.loadtxt for massive text files
    events_df = pd.read_csv(events_file, sep=r'\s+', header=None, names=['t', 'x', 'y', 'p'], comment='#')

    print("2. Formatting events array to [x, y, t, p]...")
    # Your dataset_dtx.py expects the HDF5 event columns in the order: [x, y, timestamp, polarity]
    events_array = np.zeros((len(events_df), 4), dtype=np.float64)
    events_array[:, 0] = events_df['x'].values
    events_array[:, 1] = events_df['y'].values
    events_array[:, 2] = events_df['t'].values
    events_array[:, 3] = events_df['p'].values

    print("3. Loading image timestamps...")
    images_df = pd.read_csv(images_file, sep=r'\s+', header=None, names=['t', 'filename'], comment='#')
    image_timestamps = images_df['t'].values
    image_filenames = images_df['filename'].values

    print("4. Computing event-to-image synchronization indices...")
    # Extract just the timestamps column to search against
    event_timestamps = events_array[:, 2]
    # np.searchsorted finds the index of the first event that occurred >= the image timestamp
    image_raw_event_inds = np.searchsorted(event_timestamps, image_timestamps)

    print("5. Loading grayscale frames...")
    image_list = []
    for fname in tqdm(image_filenames, desc="Reading Images"):
        # Handle cases where the filename string might or might not include 'img/'
        img_path = os.path.join(data_dir, fname)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, 'img', os.path.basename(fname))
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image_list.append(img)

    # Stack images into a single 3D numpy array [M, H, W]
    image_raw = np.stack(image_list, axis=0)

    print(f"6. Writing data to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        # Create the MVSEC internal folder structure
        davis = f.create_group('davis')
        left = davis.create_group('left')

        # Save datasets with GZIP compression to keep the file size manageable
        left.create_dataset('events', data=events_array, compression='gzip')
        left.create_dataset('image_raw', data=image_raw, dtype=np.uint8, compression='gzip')
        left.create_dataset('image_raw_ts', data=image_timestamps, dtype=np.float64)
        left.create_dataset('image_raw_event_inds', data=image_raw_event_inds, dtype=np.uint64)

    print(f"\nSuccess! Built {output_h5}.")
    print(f"Total Events: {len(events_array):,}")
    print(f"Total Frames: {len(image_raw):,}")

if __name__ == "__main__":
    # Define input folder and output filename
    INPUT_DIR = event_data_path + args.dataset
    OUTPUT_FILE = INPUT_DIR + '.h5'
    
    convert_uzh_to_mvsec_h5(INPUT_DIR, OUTPUT_FILE)
