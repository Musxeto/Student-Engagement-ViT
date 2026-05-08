import os
import re

def rename_to_sequential(directory, class_name):
    print(f"Renaming files in {directory}...")
    
    # Get all files and sort them naturally (version sort)
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    
    # Sort files to maintain relative order
    # Extract number for sorting if possible, otherwise use natural sort
    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
    files.sort(key=natural_key)
    
    for i, filename in enumerate(files, start=1):
        # New name with leading zero if desired? User said "01"
        new_name = f"{class_name}_{i:02d}.jpg"
        
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # Use a temporary name if the new name might conflict with an existing one
        # but since we are starting fresh and renaming all, we can just do it in a two-pass if needed
        # or just rename to a temp name first.
        
    # To avoid collisions, first rename everything to a temp name
    temp_files = []
    for i, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        temp_name = f"temp_{i}.jpg"
        temp_path = os.path.join(directory, temp_name)
        os.rename(old_path, temp_path)
        temp_files.append(temp_path)
    
    for i, temp_path in enumerate(temp_files, start=1):
        new_name = f"{class_name}_{i:02d}.jpg"
        new_path = os.path.join(directory, new_name)
        os.rename(temp_path, new_path)

    print(f"✅ Renamed {len(files)} files in {class_name}")

if __name__ == "__main__":
    base_path = "dataset_creation/engagement_dataset"
    classes = ["disengaged", "highly_engaged", "moderately_engaged"]
    
    for cls in classes:
        dir_path = os.path.join(base_path, cls)
        if os.path.exists(dir_path):
            rename_to_sequential(dir_path, cls)
