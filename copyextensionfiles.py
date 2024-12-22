import os
import shutil
from pathlib import Path
import sys

def copy_files_by_extension(source_dir, destination_dir, extensions):
    # Convert extensions to lowercase for case-insensitive comparison
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Keep track of copied files
    copied_count = 0
    
    # Walk through directory tree
    for root, _, files in os.walk(source_dir):
        for filename in files:
            # Check if file has one of the desired extensions
            if any(filename.lower().endswith(ext) for ext in extensions):
                source_path = os.path.join(root, filename)
                
                # Create unique filename if file already exists in destination
                base_name = Path(filename).stem
                extension = Path(filename).suffix
                counter = 1
                dest_filename = filename
                
                while os.path.exists(os.path.join(destination_dir, dest_filename)):
                    dest_filename = f"{base_name}_{counter}{extension}"
                    counter += 1
                
                # Copy the file
                destination_path = os.path.join(destination_dir, dest_filename)
                try:
                    shutil.copy2(source_path, destination_path)
                    copied_count += 1
                    print(f"Copied: {source_path} -> {destination_path}")
                except Exception as e:
                    print(f"Error copying {source_path}: {str(e)}")
    
    print(f"\nOperation complete! Copied {copied_count} files.")

def main():
    # Get source directory, destination directory, and extensions from user
    source_dir = input("Enter the source directory path: ").strip()
    destination_dir = input("Enter the destination directory path: ").strip()
    extensions_input = input("Enter file extensions to copy (separate with spaces, e.g., pdf jpg): ").strip()
    
    # Convert extensions input to list
    extensions = extensions_input.split()
    
    # Validate inputs
    if not os.path.exists(source_dir):
        print("Error: Source directory does not exist!")
        sys.exit(1)
    
    # Run the copy operation
    try:
        copy_files_by_extension(source_dir, destination_dir, extensions)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()