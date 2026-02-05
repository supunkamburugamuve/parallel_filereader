#!/usr/bin/env python3
"""
Create VDS by dividing an existing HDF5 file into N real source files

Takes one large HDF5 file, copies slices of data into N separate source files,
and creates a VDS that maps back to those source files.
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path


def create_vds_from_existing(source_file, vds_file, dataset_name, num_divisions,
                             source_pattern=None, verbose=True):
    """
    Copy data from an existing HDF5 file into N source files and create a VDS.

    Args:
        source_file: Existing HDF5 file to divide
        vds_file: Output VDS file
        dataset_name: Dataset name in source file
        num_divisions: Number of source files (N)
        source_pattern: Pattern for source file names (e.g. 'slice_{:03d}.h5')
        verbose: Print progress
    """

    if verbose:
        print(f"Creating VDS from existing file: {source_file}")
        print(f"Output VDS: {vds_file}")
        print(f"Dataset: {dataset_name}")
        print(f"Number of source files: {num_divisions}")
        print()

    # Open source file and get info
    with h5py.File(source_file, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in {source_file}")

        source_shape = f[dataset_name].shape
        dtype = f[dataset_name].dtype

        if verbose:
            print(f"Source dataset shape: {source_shape}")
            print(f"Source dataset dtype: {dtype}")
            print(f"Source file size: {np.prod(source_shape) * dtype.itemsize / (1024**3):.2f} GB")

    # Calculate division along slowest (first) dimension
    total_frames = source_shape[0]
    frames_per_division = total_frames // num_divisions
    remainder = total_frames % num_divisions

    if verbose:
        print()
        print(f"Division along axis 0 (slowest dimension):")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames per file: {frames_per_division}")
        print(f"  Remainder: {remainder}")
        print()

    # VDS has same shape as source
    vds_shape = source_shape

    # Generate source file names
    if source_pattern is None:
        vds_stem = Path(vds_file).stem
        source_pattern = f"{vds_stem}_source_{{:03d}}.h5"

    # Determine output directory (source files go next to VDS file)
    vds_dir = Path(vds_file).parent

    # Create VDS layout
    try:
        layout = h5py.VirtualLayout(shape=vds_shape, dtype=dtype)
    except TypeError:
        layout = h5py.VirtualLayout(shape=vds_shape, dtype=dtype)

    # Copy data into source files and build VDS layout
    offset = 0
    division_info = []
    source_files = []

    with h5py.File(source_file, 'r') as src:
        src_dset = src[dataset_name]

        for i in range(num_divisions):
            # Calculate frames for this division
            n_frames = frames_per_division + (1 if i < remainder else 0)

            # Source file name
            if '{}' in source_pattern or '{:' in source_pattern:
                src_name = source_pattern.format(i)
            else:
                src_name = f"{source_pattern}_{i:03d}.h5"

            src_path = str(vds_dir / src_name)
            source_files.append(src_name)

            # Shape for this source file
            src_shape = (n_frames,) + source_shape[1:]

            if verbose:
                print(f"Source file {i+1}/{num_divisions}: {src_name}")
                print(f"  Copying frames [{offset}:{offset + n_frames}] -> shape {src_shape}")

            # Copy data into the new source file
            with h5py.File(src_path, 'w') as dst:
                dst.create_dataset(dataset_name, data=src_dset[offset:offset + n_frames])

            if verbose:
                file_size = Path(src_path).stat().st_size / (1024**2)
                print(f"  Written: {file_size:.2f} MB")

            # Create virtual source pointing to the new source file
            vsource = h5py.VirtualSource(src_name, dataset_name, shape=src_shape)

            # Map entire source file into this slice of the VDS
            layout[offset:offset + n_frames] = vsource

            division_info.append({
                'source_file': src_name,
                'start': offset,
                'end': offset + n_frames,
                'frames': n_frames
            })

            offset += n_frames

    # Create VDS file
    with h5py.File(vds_file, 'w') as f:
        try:
            f.create_virtual_dataset(dataset_name, layout)
        except TypeError:
            f.create_virtual_dataset(dataset_name, layout)

        # Add metadata
        f.attrs['vds_original_file'] = str(source_file)
        f.attrs['vds_num_sources'] = num_divisions
        f.attrs['vds_source_pattern'] = source_pattern

        f[dataset_name].attrs['original_file'] = str(source_file)
        f[dataset_name].attrs['num_sources'] = num_divisions
        f[dataset_name].attrs['source_files'] = source_files

        # Store division info
        for i, info in enumerate(division_info):
            f[dataset_name].attrs[f'source_{i}_file'] = info['source_file']
            f[dataset_name].attrs[f'source_{i}_start'] = info['start']
            f[dataset_name].attrs[f'source_{i}_end'] = info['end']
            f[dataset_name].attrs[f'source_{i}_frames'] = info['frames']

    if verbose:
        print()
        print(f"✓ VDS created: {vds_file}")
        print(f"  Total shape: {vds_shape}")
        print(f"  Source files: {num_divisions}")
        for sf in source_files:
            print(f"    - {sf}")
        print()
        print("Data has been copied from the original into the source files.")
        print("The VDS maps to the source files (not the original).")

    return division_info


def verify_vds(vds_file, source_file, dataset_name='data', verbose=True):
    """Verify VDS can be read and data matches the original source."""

    if verbose:
        print(f"\nVerifying VDS: {vds_file}")

    with h5py.File(vds_file, 'r') as f:
        dset = f[dataset_name]

        if verbose:
            print(f"  Shape: {dset.shape}")
            print(f"  Dtype: {dset.dtype}")
            print(f"  Original file: {f.attrs.get('vds_original_file', 'N/A')}")
            print(f"  Source files: {f.attrs.get('vds_num_sources', 'N/A')}")

        # Try reading first and last frame
        try:
            first = dset[0]
            last = dset[-1]
            if verbose:
                print(f"  ✓ Can read first frame: {first.shape}")
                print(f"  ✓ Can read last frame: {last.shape}")

            # Try reading middle
            middle_idx = dset.shape[0] // 2
            middle = dset[middle_idx]
            if verbose:
                print(f"  ✓ Can read middle frame: {middle.shape}")

        except Exception as e:
            print(f"  ✗ Read failed: {e}")
            return False

        # Verify data matches the original file
        if source_file and Path(source_file).exists():
            if verbose:
                print(f"  Comparing with original: {source_file}")
            with h5py.File(source_file, 'r') as orig:
                orig_dset = orig[dataset_name]
                # Compare a few frames
                for idx in [0, middle_idx, dset.shape[0] - 1]:
                    if not np.array_equal(dset[idx], orig_dset[idx]):
                        print(f"  ✗ Data mismatch at frame {idx}")
                        return False
                if verbose:
                    print(f"  ✓ Data matches original at sampled frames")

    if verbose:
        print("  ✓ VDS verification passed")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Copy data from an existing HDF5 file into N source files and create a VDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Divide existing file into 8 source files + VDS
  %(prog)s -i data.h5 -o data_vds.h5 -n 8

  # Specify dataset name
  %(prog)s -i data.h5 -o data_vds.h5 -d jungfrau1M_data -n 10

  # Custom source file naming
  %(prog)s -i data.h5 -o data_vds.h5 -n 8 -p "slice_{:02d}.h5"

  # With verification against original
  %(prog)s -i data.h5 -o data_vds.h5 -n 8 --verify

How it works:
  - Takes ONE existing HDF5 file
  - Copies slices of data into N separate source files
  - Creates a VDS that maps to those source files
  - Useful for parallel I/O testing or distributed processing

  Example: 1000-frame file divided into 8 source files:
    - source_000.h5: frames [0:125]   (copied from original)
    - source_001.h5: frames [125:250] (copied from original)
    - ...
    - source_007.h5: frames [875:1000] (copied from original)
    - data_vds.h5: VDS mapping all source files
        """
    )

    parser.add_argument('-i', '--input', required=True,
                       help='Input HDF5 file (existing file to divide)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output VDS file')
    parser.add_argument('-d', '--dataset', default='data',
                       help='Dataset name in input file (default: data)')
    parser.add_argument('-n', '--num-divisions', type=int, required=True,
                       help='Number of source files to create (N)')
    parser.add_argument('-p', '--pattern',
                       help='Source file naming pattern (default: <vds_name>_source_{:03d}.h5)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify VDS data matches original after creation')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode')

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Check num_divisions is valid
    if args.num_divisions < 1:
        print("Error: num_divisions must be at least 1", file=sys.stderr)
        return 1

    # Create VDS
    try:
        division_info = create_vds_from_existing(
            source_file=args.input,
            vds_file=args.output,
            dataset_name=args.dataset,
            num_divisions=args.num_divisions,
            source_pattern=args.pattern,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Verify if requested
    if args.verify:
        if not verify_vds(args.output, args.input, args.dataset, verbose=not args.quiet):
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())