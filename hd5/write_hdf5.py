#!/usr/bin/env python3
"""
Create actual HDF5 source files for VDS

Generates the source files that the VDS points to.
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path
import json


def create_source_files(vds_file, fill_with_data=False, verbose=True):
    """
    Create source files based on VDS metadata.

    Args:
        vds_file: Path to VDS file
        fill_with_data: If True, fill with random data. If False, create empty files.
        verbose: Print progress
    """

    # Read VDS metadata
    with h5py.File(vds_file, 'r') as f:
        if 'data' in f:
            dataset_name = 'data'
        else:
            # Find first dataset
            dataset_name = list(f.keys())[0]

        dset = f[dataset_name]

        # Get source file list
        if 'source_files' in dset.attrs:
            source_files = list(dset.attrs['source_files'])
        else:
            print("Error: VDS doesn't contain source file information", file=sys.stderr)
            return 1

        vds_shape = dset.shape
        dtype = dset.dtype

    if verbose:
        print(f"Creating {len(source_files)} source files for VDS: {vds_file}")
        print(f"Dataset: {dataset_name}")
        print(f"VDS shape: {vds_shape}")
        print(f"Dtype: {dtype}")
        print()

    vds_dir = Path(vds_file).parent

    # Create each source file
    for i, source_file in enumerate(source_files):
        source_path = vds_dir / source_file

        # Get the shape for this source from VDS
        with h5py.File(vds_file, 'r') as f:
            # Read virtual sources to get shape
            # This is tricky - we need to infer from VDS structure
            pass

        # Calculate shape (divide along first dimension)
        total_frames = vds_shape[0]
        num_files = len(source_files)
        frames_per_file = total_frames // num_files
        remainder = total_frames % num_files

        # This file gets extra frame if within remainder
        n_frames = frames_per_file + (1 if i < remainder else 0)
        source_shape = (n_frames,) + vds_shape[1:]

        if verbose:
            print(f"Creating {i+1}/{len(source_files)}: {source_path}")
            print(f"  Shape: {source_shape}")
            print(f"  Size: {np.prod(source_shape) * dtype.itemsize / (1024**3):.2f} GB")

        # Create file
        with h5py.File(source_path, 'w') as f:
            if fill_with_data:
                # Create with random data
                if verbose:
                    print(f"  Filling with random data...")
                data = np.random.randn(*source_shape).astype(dtype)
                f.create_dataset(dataset_name, data=data)
            else:
                # Create empty (faster)
                f.create_dataset(dataset_name, shape=source_shape, dtype=dtype)

        if verbose:
            print(f"  ✓ Created")

    if verbose:
        print()
        print(f"✓ All {len(source_files)} source files created")

    return 0


def create_from_spec(shape, num_files, dtype, dataset_name, output_dir,
                     pattern=None, fill_with_data=False, verbose=True):
    """
    Create source files from specification (without needing VDS first).

    Args:
        shape: Full shape tuple
        num_files: Number of files
        dtype: Data type
        dataset_name: Dataset name
        output_dir: Directory to create files
        pattern: File name pattern
        fill_with_data: Fill with random data
        verbose: Print progress
    """

    if verbose:
        print(f"Creating {num_files} HDF5 source files")
        print(f"  Total shape: {shape}")
        print(f"  Dtype: {dtype}")
        print(f"  Dataset: {dataset_name}")
        print()

    # Parse dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    # Calculate division
    total_frames = shape[0]
    frames_per_file = total_frames // num_files
    remainder = total_frames % num_files

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate file names
    if pattern is None:
        pattern = "source_{:03d}.h5"

    for i in range(num_files):
        # Calculate frames for this file
        n_frames = frames_per_file + (1 if i < remainder else 0)
        source_shape = (n_frames,) + shape[1:]

        # File name
        if '{}' in pattern or '{:' in pattern:
            filename = pattern.format(i)
        else:
            filename = f"{pattern}_{i:03d}.h5"

        filepath = output_dir / filename

        if verbose:
            print(f"Creating {i+1}/{num_files}: {filepath}")
            print(f"  Shape: {source_shape}")
            print(f"  Size: {np.prod(source_shape) * dtype.itemsize / (1024**3):.2f} GB")

        # Create file
        with h5py.File(filepath, 'w') as f:
            if fill_with_data:
                if verbose:
                    print(f"  Generating random data...")
                data = np.random.randn(*source_shape).astype(dtype)
                f.create_dataset(dataset_name, data=data)
            else:
                # Create empty dataset
                f.create_dataset(dataset_name, shape=source_shape, dtype=dtype)

        if verbose:
            print(f"  ✓ Created")

    if verbose:
        print()
        print(f"✓ All {num_files} files created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create HDF5 source files for VDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create source files from existing VDS
  %(prog)s --vds output.h5

  # Create source files with random data
  %(prog)s --vds output.h5 --fill

  # Create source files from specification (no VDS needed)
  %(prog)s -s 10000 1000 1000 -n 8 -d data -o ./source_files/

  # With custom pattern and random data
  %(prog)s -s 10000 1000 1000 -n 8 -d data -o ./data/ -p "run_{:02d}.h5" --fill
        """
    )

    # VDS-based creation
    parser.add_argument('--vds', help='Path to VDS file (read metadata from here)')

    # Specification-based creation
    parser.add_argument('-s', '--shape', nargs='+', type=int,
                       help='Shape (e.g., 10000 1000 1000)')
    parser.add_argument('-n', '--num-files', type=int,
                       help='Number of files')
    parser.add_argument('-t', '--dtype', default='float64',
                       help='Data type (default: float64)')
    parser.add_argument('-d', '--dataset', default='data',
                       help='Dataset name (default: data)')
    parser.add_argument('-o', '--output', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('-p', '--pattern',
                       help='File name pattern (e.g., run_{:03d}.h5)')

    # Options
    parser.add_argument('--fill', action='store_true',
                       help='Fill files with random data (slower, for testing)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode')

    args = parser.parse_args()

    try:
        if args.vds:
            # Create from VDS metadata
            return create_source_files(
                vds_file=args.vds,
                fill_with_data=args.fill,
                verbose=not args.quiet
            )

        elif args.shape and args.num_files:
            # Create from specification
            shape = tuple(args.shape)
            return create_from_spec(
                shape=shape,
                num_files=args.num_files,
                dtype=args.dtype,
                dataset_name=args.dataset,
                output_dir=args.output,
                pattern=args.pattern,
                fill_with_data=args.fill,
                verbose=not args.quiet
            )

        else:
            print("Error: Must specify either --vds or (-s and -n)", file=sys.stderr)
            parser.print_help()
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())