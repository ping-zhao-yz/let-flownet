#!/usr/bin/env python3
import os
import hdf5plugin  # CRITICAL: Must be imported before or alongside h5py to load DSEC compression filters
import h5py
import numpy as np
from tqdm import tqdm

DSEC_ROOT = "/path/to/DSEC"
OUTPUT_DIR = "/path/to/DSEC_formatted"

def rebuild_sequence_no_images(seq_name):
    print("=" * 60)
    print(f"Rebuilding {seq_name} (Chunked Streaming / Safe RAM)...")
    print("=" * 60)
    seq_path = os.path.join(DSEC_ROOT, seq_name)

    events_file = os.path.join(seq_path, "events", "left", "events.h5")
    img_ts_file = os.path.join(seq_path, "images", "timestamps.txt")
    if not os.path.exists(img_ts_file):
        img_ts_file = os.path.join(seq_path, "images", "timestamps")

    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Missing events file: {events_file}")
    if not os.path.exists(img_ts_file):
        raise FileNotFoundError(f"Missing timestamps file: {img_ts_file}")

    # --- 1. Load Timestamps & Handle 2-Column Exposures ---
    print("=> Loading and standardizing frame timestamps...")
    raw_ts = np.loadtxt(img_ts_file, dtype=np.float64)
    if raw_ts.ndim == 2:
        img_timestamps = np.mean(raw_ts, axis=1)
    else:
        img_timestamps = raw_ts

    num_frames = len(img_timestamps)
    print(f"   Found {num_frames} frames in timestamp record.")

    # --- 2. Inspect Event Metadata ---
    with h5py.File(events_file, 'r') as f_ev:
        num_events = f_ev['events/t'].shape[0]
        t_offset = np.int64(f_ev['t_offset'][()])
    print(f"   Total events to pack: {num_events:,}")

    # --- 3. Compute image_raw_event_inds Safely (Chunked) ---
    print("=> Computing image_raw_event_inds (Chunked to save RAM)...")
    event_inds = np.zeros(num_frames, dtype=np.int64)
    
    with h5py.File(events_file, 'r') as f_ev:
        t_dset = f_ev['events/t']
        frame_idx = 0
        t_chunk_size = 10_000_000 # Read 10 million timestamps at a time (~80MB RAM)
        
        for start_idx in tqdm(range(0, num_events, t_chunk_size), desc="Mapping frames"):
            end_idx = min(start_idx + t_chunk_size, num_events)
            
            # Load chunk and align timestamp
            t_chunk = np.array(t_dset[start_idx:end_idx], dtype=np.int64) + t_offset
            
            while frame_idx < num_frames:
                # If the frame's timestamp falls inside this chunk (or it's the final chunk)
                if img_timestamps[frame_idx] <= t_chunk[-1] or end_idx == num_events:
                    local_idx = np.searchsorted(t_chunk, img_timestamps[frame_idx])
                    event_inds[frame_idx] = start_idx + local_idx
                    frame_idx += 1
                else:
                    break # Frame belongs in a future chunk

    # --- 4. Stream and Write HDF5 in Chunks ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_data_path = os.path.join(OUTPUT_DIR, f"{seq_name}_data.hdf5")

    chunk_size = 5_000_000 # Stream 5 million full events at a time

    with h5py.File(out_data_path, 'w') as f_out:
        grp = f_out.create_group('davis/left')

        # Create resizable or pre-allocated events dataset
        events_dset = grp.create_dataset(
            'events',
            shape=(num_events, 4),
            dtype=np.int64,
            chunks=(1_000_000, 4),
            compression="lzf"
        )
        grp.create_dataset('image_raw_ts', data=img_timestamps)
        grp.create_dataset('image_raw_event_inds', data=event_inds)

        print("\n=> Streaming events in chunks directly to disk...")
        with h5py.File(events_file, 'r') as f_ev:
            p_ref = f_ev['events/p']
            t_ref = f_ev['events/t']
            x_ref = f_ev['events/x']
            y_ref = f_ev['events/y']

            for start in tqdm(range(0, num_events, chunk_size), desc="Streaming chunks"):
                end = min(start + chunk_size, num_events)

                x_c = np.array(x_ref[start:end], dtype=np.int64)
                y_c = np.array(y_ref[start:end], dtype=np.int64)
                t_c = np.array(t_ref[start:end], dtype=np.int64) + t_offset
                p_c = np.array(p_ref[start:end], dtype=np.int64)

                # Stack only the current slice (~150 MB buffer)
                chunk_packed = np.column_stack((x_c, y_c, t_c, p_c))
                events_dset[start:end] = chunk_packed

        # Write dummy zero-filled image dataset (1080x1440)
        print("\n=> Generating zeroed image canvas (1080x1440)...")
        grp.create_dataset(
            'image_raw',
            shape=(num_frames, 1080, 1440),
            dtype=np.uint8,
            fillvalue=0,
            chunks=(1, 1080, 1440),
            compression="lzf"
        )

    print(f"\n=> Successfully rebuilt: {out_data_path}")
    print(f"   Final size on disk: {os.path.getsize(out_data_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    rebuild_sequence_no_images("zurich_city_10_a")
