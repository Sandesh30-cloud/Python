import os
import re
from datetime import datetime

def rename_files_us_to_eu_dates(directory):
    
    us_date_pattern = r'(\d{2})[-_](\d{2})[-_](\d{4})'
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Counter for renamed files
    renamed_count = 0
    
    for filename in files:
        # Search for date pattern in filename
        match = re.search(us_date_pattern, filename)
        
        if match:
            try:
                # Extract date components
                month, day, year = match.groups()
                
                # Validate date
                date_obj = datetime(int(year), int(month), int(day))
                
                # Create new filename with European date format
                new_filename = re.sub(
                    us_date_pattern,
                    f'{day}-{month}-{year}',
                    filename
                )
                
                # Construct full file paths
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)
                renamed_count += 1
                print(f'Renamed: {filename} → {new_filename}')
                
            except ValueError:
                # Skip if date is invalid
                print(f'Skipped {filename}: Invalid date')
                continue
    
    print(f'\nComplete! Renamed {renamed_count} files.')

# Example usage
if __name__ == "__main__":
    # Example directory path - modify this to your needs
    target_directory = r"C:\Users\Sandesh Yesane\Downloads"
    
    try:
        rename_files_us_to_eu_dates(target_directory)
    except Exception as e:
        print(f"An error occurred: {str(e)}")