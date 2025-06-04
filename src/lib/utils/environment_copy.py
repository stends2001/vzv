import os
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from directories import dir_project

def copy_environment(source_dir=None, dest_dir=None, exclude_files=None, exclude_dirs = None):
    """
    Copies files from source_dir to dest_dir, excluding files with specified extensions.

    Parameters:
    - source_dir: str, path of the source directory.
    - dest_dir: str, path of the destination directory.
    - exclude_extensions: list, extensions of files to exclude (e.g., ['.csv', '.py', '.ipynb']).
    """
    if source_dir == dest_dir:
        raise ValueError('source and destination folders cannot be the same')

    if source_dir is None:
        raise ValueError('please insert a valid source directory')
    
    if dest_dir is None:
        raise ValueError('please insert a valid destination directory')

    if exclude_dirs is None:
        exclude_dirs = []  # Default to an empty list if no directories are specified

    if exclude_files is None:
        exclude_files = []  # Default to an empty list if no directories are specified
    
# Step 1: Calculate the total number of files to copy (excluding the ones to be skipped)
    total_files_to_copy = 0

    # Lists to track exclusions
    excluded_files = []
    excluded_dirs = set()

    for root, dirs, files in os.walk(source_dir):
        # Check if the directory should be skipped
        relative_path = os.path.relpath(root, source_dir)
        if any(exclude_dir in relative_path for exclude_dir in exclude_dirs):
            excluded_dirs.add(relative_path)
            continue
        
        # Count the files that will be copied
        total_files_to_copy += sum(1 for file in files if file not in exclude_files)

    print(f"Total files to be copied: {total_files_to_copy}")

    # Step 2: Start the actual copying process with a progress bar
    files_copied = 0  # Track the number of files copied
    
    # Create a progress bar using tqdm
    with tqdm(total=total_files_to_copy, desc="Copying files", unit="file") as pbar:
        for root, dirs, files in os.walk(source_dir):
            # Skip directories that are in the exclude list
            relative_path = os.path.relpath(root, source_dir)
            dest_root = os.path.join(dest_dir, relative_path)

            if any(exclude_dir in relative_path for exclude_dir in exclude_dirs):
                continue  # Skip this directory
            
            # Ensure destination folder exists
            if not os.path.exists(dest_root):
                os.makedirs(dest_root)

            # Process each file and update the progress bar
            for file in files:
                file_path = os.path.join(root, file)

                # Copy the file if it's not in the exclusion list
                if file not in exclude_files:
                    dest_file_path = os.path.join(dest_root, file)
                    shutil.copy(file_path, dest_file_path)
                    files_copied += 1
                    pbar.update(1)  # Update progress bar by 1 for each file copied
                else:
                    # If the file is excluded, just add it to the exclusion list
                    excluded_files.append(file)

    print(f"Finished copying {files_copied} files.")

    # Print the exclusion overview after copying is done
    print("\nExclusion Overview:")
    if excluded_files:
        print(f"Excluded files: {len(excluded_files)}")
        for file in excluded_files:
            print(f"  - {file}")

    if excluded_dirs:
        print(f"Excluded directories: {len(excluded_dirs)}")
        for dir in excluded_dirs:
            print(f"  - {dir}")



if __name__ == '__main__':
    dir_project_root        = dir_project
    dir_project_cleancoopy  = os.path.join(dir_project, 'cleancopy')

    files_excluded          = []
    dir_excluded            = []
    current_date            = "_"+str(datetime.today().date())
    dir_project_cleancoopy  = dir_project_cleancoopy+current_date

    copy_environment(source_dir= dir_project_root, dest_dir= dir_project_cleancoopy, exclude_files= files_excluded, exclude_dirs=dir_excluded)