import os
import shutil

# Path to your main dataset folder which contains 32 subfolders
SOURCE_DIR = r"C:\Users\resha\Downloads\Plant Disease Prediction\data\train"

# Path to new folder where reduced dataset will be created
DESTINATION_DIR = r"C:\Users\resha\Downloads\Plant Disease Prediction\data\test"

# How many images you want to copy
N = 5

# Create destination folder if it doesn't exist
os.makedirs(DESTINATION_DIR, exist_ok=True)

# Iterate through each subfolder
for folder_name in os.listdir(SOURCE_DIR):
    source_subfolder = os.path.join(SOURCE_DIR, folder_name)

    # Only process directories
    if not os.path.isdir(source_subfolder):
        continue

    # Create matching folder in destination
    dest_subfolder = os.path.join(DESTINATION_DIR, folder_name)
    os.makedirs(dest_subfolder, exist_ok=True)

    # Get all image files (you can add more extensions if needed)
    images = [
        f for f in os.listdir(source_subfolder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Sort to ensure consistent ordering
    images.sort()

    # Copy first N images
    for img in images[:N]:
        src_path = os.path.join(source_subfolder, img)
        dst_path = os.path.join(dest_subfolder, img)

        shutil.copy(src_path, dst_path)

    print(f"Copied {min(N, len(images))} images → {folder_name}")

print("\n✔ Done! New dataset created successfully.")
