#!/usr/bin/env python3
"""
Create HDF5 VDS by dividing shape among N source files

Divides the VDS along the slowest (first) dimension across source files.
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path


def create_vds_from_params(vds_file, shape, num_files, dtype='float64',
                           dataset_name='data', source_pattern=None,
                           fillvalue=0, verbose=True):
    """
    Create VDS by dividing shape among N source files.

    Args:
        vds_file: Output VDS file path
        shape: Full VDS shape tuple (e.g., (1000, 1064, 1030))
        num_files: Number of source files
        dtype: Data type (e.g., 'float64', 'float32', 'int32')
        dataset_name: Dataset name
        source_pattern: Pattern for source file names (e.g., 'data_{:03d}.h5')
        fillvalue: Fill value for unmapped regions
        verbose: Print info
    """

    if verbose:
        print(f"Creating VDS: {vds_file}")
        print(f"  Shape: {shape}")
        print(f"  Dtype: {dtype}")
        print(f"  Number of source files: {num_files}")
        print(f"  Dataset name: {dataset_name}")
        print()

    # Parse dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    # Calculate division along slowest (first) dimension
    total_frames = shape[0]
    frames_per_file = total_frames // num_files
    remainder = total_frames % num_files

    if verbose:
        print(f"Division along axis 0 (slowest dimension):")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames per file: {frames_per_file}")
        print(f"  Remainder: {remainder}")
        print()

    # Generate source file names if pattern not provided
    if source_pattern is None:
        vds_stem = Path(vds_file).stem
        source_pattern = f"{vds_stem}_source_{{:03d}}.h5"

    # Create VDS layout
    # Note: fillvalue parameter added in h5py 3.2, use try/except for compatibility
    try:
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype, fillvalue=fillvalue)
    except TypeError:
        # Older h5py version without fillvalue support
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
        if verbose and fillvalue != 0:
            print(f"Warning: h5py version doesn't support fillvalue, using default (0)")

    # Map each source file
    offset = 0
    source_files = []

    for i in range(num_files):
        # Calculate frames for this file
        # Distribute remainder across first files
        n_frames = frames_per_file + (1 if i < remainder else 0)

        # Source file name
        if '{}' in source_pattern or '{:' in source_pattern:
            source_file = source_pattern.format(i)
        else:
            source_file = f"{source_pattern}_{i:03d}.h5"

        source_files.append(source_file)

        # Source shape (same spatial dims, different frame count)
        source_shape = (n_frames,) + shape[1:]

        # Create virtual source
        vsource = h5py.VirtualSource(source_file, dataset_name, shape=source_shape)

        # Map to VDS
        layout[offset:offset + n_frames] = vsource

        if verbose:
            print(f"Source file {i+1}/{num_files}: {source_file}")
            print(f"  Shape: {source_shape}")
            print(f"  Frames: {n_frames}")
            print(f"  VDS mapping: [{offset}:{offset + n_frames}]")

        offset += n_frames

    # Create VDS file
    with h5py.File(vds_file, 'w') as f:
        # Try with fillvalue, fall back if not supported
        try:
            f.create_virtual_dataset(dataset_name, layout, fillvalue=fillvalue)
        except TypeError:
            # Older h5py without fillvalue support
            f.create_virtual_dataset(dataset_name, layout)

        # Add metadata
        f.attrs['vds_shape'] = shape
        f.attrs['vds_dtype'] = str(dtype)
        f.attrs['vds_num_sources'] = num_files
        f.attrs['vds_source_pattern'] = source_pattern
        f[dataset_name].attrs['source_files'] = source_files
        f[dataset_name].attrs['frames_per_file'] = frames_per_file

    if verbose:
        print()
        print(f"✓ VDS created: {vds_file}")
        print(f"  Total shape: {shape}")
        print(f"  Total size: {np.prod(shape) * dtype.itemsize / (1024**3):.2f} GB")
        print(f"  Source files: {len(source_files)}")
        print()
        print("Source files that should exist:")
        for sf in source_files:
            print(f"  - {sf}")

    return source_files


def main():
    parser = argparse.ArgumentParser(
        description='Create HDF5 VDS by dividing shape among N source files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create VDS with shape (1000, 1064, 1030) divided among 10 files
  %(prog)s -o output.h5 -s 1000 1064 1030 -n 10

  # With specific dtype
  %(prog)s -o output.h5 -s 2000 512 512 -n 8 -t float32

  # With custom dataset name
  %(prog)s -o output.h5 -s 1000 1064 1030 -n 10 -d jungfrau1M_data

  # With custom source file pattern
  %(prog)s -o output.h5 -s 1000 1064 1030 -n 10 -p "run_{:03d}.h5"

  # With fill value
  %(prog)s -o output.h5 -s 1000 1064 1030 -n 10 -f -1

  # Real example: 8000 Jungfrau frames across 8 files
  %(prog)s -o all_data.h5 -s 8000 1064 1030 -n 8 -d jungfrau1M_data -t float64

How it works:
  - VDS shape is divided along the FIRST dimension (slowest)
  - Each source file gets approximately shape[0] / num_files frames
  - Remainder frames distributed to first files
  - Example: 1000 frames / 8 files = 125 frames each
            First 1000 % 8 = 0 files get no extra frames
  - Example: 1003 frames / 8 files = 125 frames each
            First 3 files get 126 frames (125 + 1)
        """
    )

    parser.add_argument('-o', '--output', required=True,
                       help='Output VDS file path')
    parser.add_argument('-s', '--shape', nargs='+', type=int, required=True,
                       metavar='DIM',
                       help='VDS shape as space-separated dimensions (e.g., 1000 1064 1030)')
    parser.add_argument('-n', '--num-files', type=int, required=True,
                       help='Number of source files to divide data among')
    parser.add_argument('-t', '--dtype', default='float64',
                       help='Data type (default: float64). Options: float32, float64, int32, int64, uint16, etc.')
    parser.add_argument('-d', '--dataset', default='data',
                       help='Dataset name (default: data)')
    parser.add_argument('-p', '--pattern',
                       help='Source file name pattern (default: <vds_name>_source_{:03d}.h5). Use {} or {:03d} for numbering.')
    parser.add_argument('-f', '--fillvalue', type=float, default=0,
                       help='Fill value for unmapped regions (default: 0)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode')

    args = parser.parse_args()

    # Parse shape
    if len(args.shape) < 2:
        print("Error: Shape must have at least 2 dimensions", file=sys.stderr)
        return 1

    shape = tuple(args.shape)

    # Validate num_files
    if args.num_files < 1:
        print("Error: Number of files must be at least 1", file=sys.stderr)
        return 1

    if args.num_files > shape[0]:
        print(f"Warning: num_files ({args.num_files}) > first dimension ({shape[0]})",
              file=sys.stderr)
        print(f"Some source files will be empty or have very few frames", file=sys.stderr)

    # Create VDS
    try:
        create_vds_from_params(
            vds_file=args.output,
            shape=shape,
            num_files=args.num_files,
            dtype=args.dtype,
            dataset_name=args.dataset,
            source_pattern=args.pattern,
            fillvalue=args.fillvalue,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())