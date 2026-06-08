import os
import h5py
import argparse
from event_parser import EventParser


def main():
    parser = argparse.ArgumentParser(description='Event Pre-processing')
    parser.add_argument('--input-dataset-path', type=str, default='../../dataset/Event/mvsec/original/indoor_flying3/indoor_flying3_data.hdf5',
                        metavar='PARAMS', help='HDF5 dataset file path to load raw data from')
    parser.add_argument('--output-dataset-dir', type=str, default='../../dataset/Event/mvsec/preprocessed/indoor_flying3',
                        metavar='PARAMS', help='Directory to save encoded dataset')
    args = parser.parse_args()

    output_dataset_dir = args.output_dataset_dir
    os.makedirs(output_dataset_dir, exist_ok=True)

    event_dir = os.path.join(output_dataset_dir, 'event_data')
    os.makedirs(event_dir, exist_ok=True)

    gray_dir = os.path.join(output_dataset_dir, 'gray_image')
    os.makedirs(gray_dir, exist_ok=True)

    d_set = h5py.File(args.input_dataset_path, 'r')

    # (14071304, 4)
    events = d_set['davis']['left']['events']
    # (2206,)
    event_inds = d_set['davis']['left']['image_raw_event_inds']
    # (2206, 260, 346)
    gray_image = d_set['davis']['left']['image_raw']
    # (2206,)
    gray_image_ts = d_set['davis']['left']['image_raw_ts']

    event_parser = EventParser(event_dir, gray_dir)
    event_parser.generate_event_frames(
        events, gray_image, event_inds, gray_image_ts)

    d_set = None
    events = None
    event_inds = None
    gray_image = None
    gray_image_ts = None

    print('Event Pre-processing Complete!')


if __name__ == "__main__":
    main()
