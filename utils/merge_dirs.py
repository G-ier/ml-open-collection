import os
import shutil
from pathlib import Path
from typing import List, Union, Optional
from enum import Enum


class ConflictResolution(Enum):
    """Enum for handling file conflicts during directory merging."""
    SKIP = "skip"           # Skip conflicting files
    OVERWRITE = "overwrite" # Overwrite existing files
    RENAME = "rename"       # Rename conflicting files with suffix
    RAISE = "raise"         # Raise exception on conflict


def merge_directories(
    source_dirs: List[Union[str, Path]], 
    target_dir: Union[str, Path],
    conflict_resolution: ConflictResolution = ConflictResolution.RENAME,
    create_target: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Merge multiple directories into a new target directory.
    
    Args:
        source_dirs: List of source directory paths to merge
        target_dir: Target directory path where merged content will be placed
        conflict_resolution: How to handle file conflicts (skip, overwrite, rename, raise)
        create_target: Whether to create target directory if it doesn't exist
        dry_run: If True, only simulate the operation and return what would be done
        
    Returns:
        dict: Summary of the merge operation including:
            - files_copied: Number of files successfully copied
            - files_skipped: Number of files skipped due to conflicts
            - files_renamed: Number of files renamed due to conflicts
            - directories_created: Number of directories created
            - errors: List of any errors encountered
            
    Raises:
        FileNotFoundError: If source directory doesn't exist
        FileExistsError: If conflict_resolution is RAISE and conflicts occur
        PermissionError: If insufficient permissions to perform operations
    """
    
    # Convert paths to Path objects
    source_dirs = [Path(d) for d in source_dirs]
    target_dir = Path(target_dir)
    
    # Initialize results tracking
    results = {
        'files_copied': 0,
        'files_skipped': 0,
        'files_renamed': 0,
        'directories_created': 0,
        'errors': []
    }
    
    # Validate source directories
    for source_dir in source_dirs:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
    
    # Create target directory if it doesn't exist
    if create_target and not target_dir.exists():
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        results['directories_created'] += 1
        print(f"{'[DRY RUN] ' if dry_run else ''}Created target directory: {target_dir}")
    
    # Merge each source directory
    for source_dir in source_dirs:
        print(f"{'[DRY RUN] ' if dry_run else ''}Merging: {source_dir} -> {target_dir}")
        _merge_single_directory(source_dir, target_dir, conflict_resolution, dry_run, results)
    
    # Print summary
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Merge Summary:")
    print(f"  Files copied: {results['files_copied']}")
    print(f"  Files skipped: {results['files_skipped']}")
    print(f"  Files renamed: {results['files_renamed']}")
    print(f"  Directories created: {results['directories_created']}")
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"    - {error}")
    
    return results


def _merge_single_directory(
    source_dir: Path, 
    target_dir: Path, 
    conflict_resolution: ConflictResolution,
    dry_run: bool,
    results: dict
) -> None:
    """Helper function to merge a single directory."""
    
    for item in source_dir.rglob('*'):
        if item.is_file():
            # Calculate relative path from source
            relative_path = item.relative_to(source_dir)
            target_file = target_dir / relative_path
            
            # Create parent directories if needed
            target_parent = target_file.parent
            if not target_parent.exists():
                if not dry_run:
                    target_parent.mkdir(parents=True, exist_ok=True)
                results['directories_created'] += 1
            
            # Handle file copying with conflict resolution
            _copy_file_with_conflict_resolution(
                item, target_file, conflict_resolution, dry_run, results
            )


def _copy_file_with_conflict_resolution(
    source_file: Path,
    target_file: Path,
    conflict_resolution: ConflictResolution,
    dry_run: bool,
    results: dict
) -> None:
    """Handle copying a single file with conflict resolution."""
    
    try:
        if target_file.exists():
            # Handle conflict
            if conflict_resolution == ConflictResolution.SKIP:
                print(f"{'[DRY RUN] ' if dry_run else ''}Skipping existing file: {target_file}")
                results['files_skipped'] += 1
                return
                
            elif conflict_resolution == ConflictResolution.OVERWRITE:
                print(f"{'[DRY RUN] ' if dry_run else ''}Overwriting: {target_file}")
                if not dry_run:
                    shutil.copy2(source_file, target_file)
                results['files_copied'] += 1
                
            elif conflict_resolution == ConflictResolution.RENAME:
                # Find a unique name
                counter = 1
                original_stem = target_file.stem
                original_suffix = target_file.suffix
                
                while target_file.exists():
                    new_name = f"{original_stem}_{counter}{original_suffix}"
                    target_file = target_file.parent / new_name
                    counter += 1
                
                print(f"{'[DRY RUN] ' if dry_run else ''}Renaming to avoid conflict: {target_file}")
                if not dry_run:
                    shutil.copy2(source_file, target_file)
                results['files_renamed'] += 1
                
            elif conflict_resolution == ConflictResolution.RAISE:
                raise FileExistsError(f"File already exists: {target_file}")
                
        else:
            # No conflict, copy normally
            if not dry_run:
                shutil.copy2(source_file, target_file)
            results['files_copied'] += 1
            
    except Exception as e:
        error_msg = f"Error copying {source_file} to {target_file}: {str(e)}"
        print(f"{'[DRY RUN] ' if dry_run else ''}ERROR: {error_msg}")
        results['errors'].append(error_msg)


def merge_directories_simple(source_dirs: List[Union[str, Path]], target_dir: Union[str, Path]) -> None:
    """
    Simple wrapper function for basic directory merging with default settings.
    
    Args:
        source_dirs: List of source directory paths to merge
        target_dir: Target directory path where merged content will be placed
    """
    merge_directories(
        source_dirs=source_dirs,
        target_dir=target_dir,
        conflict_resolution=ConflictResolution.RENAME,
        create_target=True,
        dry_run=False
    )


# Example usage
if __name__ == "__main__":
    # Example: Merge two directories into a new one
    source_directories = ["/Users/gier/Downloads/archive/KAGGLE/AUDIO/FAKE_test", "/Users/gier/Downloads/archive/KAGGLE/AUDIO/REAL_test"]
    target_directory = "/Users/gier/Downloads/archive/KAGGLE/test_data"
    
    merge_directories(
        source_dirs=source_directories,
        target_dir=target_directory,
        conflict_resolution=ConflictResolution.RENAME,
        dry_run=False
    )
