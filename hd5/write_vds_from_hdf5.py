#!/usr/bin/env python3
"""
Create VDS by dividing an existing HDF5 file into N virtual source files

Takes one large HDF5 file and creates a VDS that divides it along the slowest axis.
The VDS points back to the original file, creating N virtual "slices".
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path


def create_vds_from_existing(source_file, vds_file, dataset_name, num_divisions,
                             virtual_source_pattern=None, verbose=True):
    """
    Create VDS by dividing existing file into N virtual slices.
    
    Args:
        source_file: Existing HDF5 file to divide
        vds_file: Output VDS file
        dataset_name: Dataset name in source file
        num_divisions: Number of virtual divisions (N)
        virtual_source_pattern: Pattern for virtual source names in metadata
        verbose: Print progress
    """
    
    if verbose:
        print(f"Creating VDS from existing file: {source_file}")
        print(f"Output VDS: {vds_file}")
        print(f"Dataset: {dataset_name}")
        print(f"Number of divisions: {num_divisions}")
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
        print(f"  Frames per division: {frames_per_division}")
        print(f"  Remainder: {remainder}")
        print()
    
    # VDS has same shape as source
    vds_shape = source_shape
    
    # Create VDS layout
    try:
        layout = h5py.VirtualLayout(shape=vds_shape, dtype=dtype)
    except TypeError:
        layout = h5py.VirtualLayout(shape=vds_shape, dtype=dtype)
    
    # Map divisions from source file
    offset = 0
    division_info = []
    
    # Generate virtual source names for metadata
    if virtual_source_pattern is None:
        vds_stem = Path(vds_file).stem
        virtual_source_pattern = f"{vds_stem}_virtual_{{:03d}}"
    
    for i in range(num_divisions):
        # Calculate frames for this division
        n_frames = frames_per_division + (1 if i < remainder else 0)
        
        # Virtual source name (for documentation/metadata only)
        if '{}' in virtual_source_pattern or '{:' in virtual_source_pattern:
            virtual_name = virtual_source_pattern.format(i)
        else:
            virtual_name = f"{virtual_source_pattern}_{i:03d}"
        
        # Create virtual source pointing to slice of original file
        vsource = h5py.VirtualSource(
            source_file, 
            dataset_name, 
            shape=source_shape
        )
        
        # Map this slice of the source to this slice of VDS
        layout[offset:offset + n_frames] = vsource[offset:offset + n_frames]
        
        division_info.append({
            'virtual_name': virtual_name,
            'start': offset,
            'end': offset + n_frames,
            'frames': n_frames
        })
        
        if verbose:
            print(f"Virtual division {i+1}/{num_divisions}: {virtual_name}")
            print(f"  Frames: {n_frames}")
            print(f"  VDS mapping: [{offset}:{offset + n_frames}]")
            print(f"  Source mapping: {source_file}[{offset}:{offset + n_frames}]")
        
        offset += n_frames
    
    # Create VDS file
    with h5py.File(vds_file, 'w') as f:
        try:
            f.create_virtual_dataset(dataset_name, layout)
        except TypeError:
            f.create_virtual_dataset(dataset_name, layout)
        
        # Add metadata
        f.attrs['vds_source_file'] = str(source_file)
        f.attrs['vds_num_divisions'] = num_divisions
        f.attrs['vds_division_pattern'] = virtual_source_pattern
        
        f[dataset_name].attrs['source_file'] = str(source_file)
        f[dataset_name].attrs['num_divisions'] = num_divisions
        
        # Store division info
        for i, info in enumerate(division_info):
            f[dataset_name].attrs[f'division_{i}_name'] = info['virtual_name']
            f[dataset_name].attrs[f'division_{i}_start'] = info['start']
            f[dataset_name].attrs[f'division_{i}_end'] = info['end']
            f[dataset_name].attrs[f'division_{i}_frames'] = info['frames']
    
    if verbose:
        print()
        print(f"✓ VDS created: {vds_file}")
        print(f"  Total shape: {vds_shape}")
        print(f"  Points to: {source_file}")
        print(f"  Virtual divisions: {num_divisions}")
        print()
        print("Note: This VDS points to the original file.")
        print("      No data was copied - the VDS is just a view of slices.")
    
    return division_info


def verify_vds(vds_file, dataset_name='data', verbose=True):
    """Verify VDS can be read and matches source."""
    
    if verbose:
        print(f"\nVerifying VDS: {vds_file}")
    
    with h5py.File(vds_file, 'r') as f:
        dset = f[dataset_name]
        
        if verbose:
            print(f"  Shape: {dset.shape}")
            print(f"  Dtype: {dset.dtype}")
            print(f"  Source file: {f.attrs.get('vds_source_file', 'N/A')}")
            print(f"  Divisions: {f.attrs.get('vds_num_divisions', 'N/A')}")
        
        # Try reading first and last frame
        try:
            first = dset[0]
            last = dset[-1]
            if verbose:
                print(f"  ✓ Can read first frame: {first.shape}")
                print(f"  ✓ Can read last frame: {last.shape}")
                
            # Try reading middle
            middle = dset[dset.shape[0] // 2]
            if verbose:
                print(f"  ✓ Can read middle frame: {middle.shape}")
                
        except Exception as e:
            print(f"  ✗ Read failed: {e}")
            return False
    
    if verbose:
        print("  ✓ VDS verification passed")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create VDS by dividing existing HDF5 file into N virtual slices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Divide existing file into 8 virtual slices
  %(prog)s -i data.h5 -o data_vds.h5 -n 8
  
  # Specify dataset name
  %(prog)s -i data.h5 -o data_vds.h5 -d jungfrau1M_data -n 10
  
  # Custom virtual source naming
  %(prog)s -i data.h5 -o data_vds.h5 -n 8 -p "slice_{:02d}"
  
  # With verification
  %(prog)s -i data.h5 -o data_vds.h5 -n 8 --verify

How it works:
  - Takes ONE existing HDF5 file
  - Creates a VDS that logically divides it into N slices
  - VDS points back to the original file (no data copied)
  - Each "virtual slice" is a view of part of the original
  - Useful for parallel I/O testing or distributed processing
  
  Example: 1000-frame file divided into 8 slices:
    - Virtual slice 0: frames [0:125]
    - Virtual slice 1: frames [125:250]
    - ...
    - Virtual slice 7: frames [875:1000]
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input HDF5 file (existing file to divide)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output VDS file')
    parser.add_argument('-d', '--dataset', default='data',
                       help='Dataset name in input file (default: data)')
    parser.add_argument('-n', '--num-divisions', type=int, required=True,
                       help='Number of divisions (N)')
    parser.add_argument('-p', '--pattern',
                       help='Virtual source naming pattern (default: <vds_name>_virtual_{:03d})')
    parser.add_argument('--verify', action='store_true',
                       help='Verify VDS after creation')
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
            virtual_source_pattern=args.pattern,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify if requested
    if args.verify:
        if not verify_vds(args.output, args.dataset, verbose=not args.quiet):
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())