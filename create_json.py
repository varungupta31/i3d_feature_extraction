import os
import json

# Path to the folder containing video subfolders (each subfolder named after a video)
video_folder = '/ssd_scratch/cvit/varun/output_frames_cmd'

# Initialize a dictionary to store video names and frame counts
video_frame_counts = {}

# Iterate through the subfolders (video folders)
for video_name in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_name)

    # Check if the item is a directory
    if os.path.isdir(video_path):
        frame_count = len(os.listdir(video_path))
        video_frame_counts[video_name] = frame_count

# Path to the JSON file to be created
json_file_path = "video_frame_counts.json"

# Write the dictionary to the JSON file
with open(json_file_path, "w") as json_file:
    json.dump(video_frame_counts, json_file, indent=4)

print("JSON file created successfully!")