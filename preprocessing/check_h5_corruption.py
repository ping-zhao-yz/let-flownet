#!/usr/bin/env python3
import os
import glob
import argparse
import h5py
from tqdm import tqdm

def verify_dataset(name, obj, filepath):
    """Attempt chunked reading through the dataset to trigger decompression filters."""
    if isinstance(obj, h5py.Dataset):
        shape = obj.shape
        if len(shape) == 0 or obj.size == 0:
            return True
        
        # Dimension-aware chunking to prevent I/O thrashing and RAM overflow
        if len(shape) >= 3:
            chunk_step = 50  # 3D/4D data (images, flow) - 50 frames at a time
        else:
            chunk_step = 500000  # 1D/2D data (events, timestamps) - 500k rows at a time
            
        chunk_step = min(chunk_step, shape[0])
        
        # Inner progress bar to show real-time dataset scanning
        for start in tqdm(range(0, shape[0], chunk_step), desc=f"  Scanning {name}", leave=False):
            end = min(start + chunk_step, shape[0])
            try:
                _ = obj[start:end]
            except Exception as e:
                print(f"\n[CORRUPTED] {filepath}")
                print(f"       -> Dataset: '{name}' | Slice: [{start}:{end}] | Error: {e}")
                return False
    return True

def scan_file(filepath):
    """Scans all internal datasets inside an HDF5 file."""
    try:
        with h5py.File(filepath, 'r') as h5_file:
            is_valid = True
            def visitor(name, obj):
                nonlocal is_valid
                # Only scan if the previous datasets were valid
                if is_valid and not verify_dataset(name, obj, filepath):
                    is_valid = False
            
            h5_file.visititems(visitor)
            return is_valid
    except Exception as e:
        print(f"\n[CORRUPTED HEADER/FILE] {filepath}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Scan and verify HDF5 files for corruption.")
    parser.add_argument("target_dir", type=str, help="Directory containing .h5 or .hdf5 files")
    parser.add_argument("--ext", default="*.hdf5", choices=["*.hdf5", "*.h5", "*.*"], help="File pattern to match")
    args = parser.parse_args()

    pattern = os.path.join(args.target_dir, "**", args.ext)
    files = glob.glob(pattern, recursive=True)

    if not files:
        # Fallback check for alternate extensions
        alt_pattern = os.path.join(args.target_dir, "**", "*.h5")
        files = glob.glob(alt_pattern, recursive=True)

    print(f"=> Found {len(files)} file(s) to verify in: {args.target_dir}")
    
    corrupted_files = []
    
    # Outer progress bar for files
    for f in tqdm(files, desc="Total Progress", position=0):
        if not scan_file(f):
            corrupted_files.append(f)

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    if corrupted_files:
        print(f"Found {len(corrupted_files)} corrupted file(s):")
        for cf in corrupted_files:
            print(f"  - {cf}")
        print("\nRe-transmit or regenerate the files listed above.")
    else:
        print("All scanned HDF5 files are intact and readable.")

if __name__ == "__main__":
    main()
