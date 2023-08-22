# Feature Extraction Using I3D Model

This repository is created by a complete reference from https://github.com/piergiaj/pytorch-i3d, for a custom video dataset feature extraction purposes.

Note the Following Essentials Steps

1. This repo expects the frames to be dumped at the desired FPS beforehand.

The following efficient `FFMPEG` script may be used

```
# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is not installed. Please install it before running this script."
    exit 1
fi

# Input directory containing .mkv videos
input_dir="path_to_input_videos"

# Output directory for extracted frames
output_dir="path_where_frames_are_to_be_stored"

# Iterate through each .mkv video
for video_path in "$input_dir"/*.mkv; do
    # Get the video filename without extension
    video_filename=$(basename "$video_path")
    video_name="${video_filename%.mkv}"

    # Create a directory with the video's name in the output directory
    video_output_dir="$output_dir/$video_name"
    mkdir -p "$video_output_dir"

    # Extract frames from the video at 1 FPS and save them in the output directory
    ffmpeg -i "$video_path" -vf "fps=1" "$video_output_dir/frame_%04d.jpg"
done

echo "Frame extraction complete."
```

2. Once frames extracted, run `python create_json.py` which will create a `JSON` mapping expected by the code as file `vid_info.json`.

3. The models may be downloaded and placed in `models` folder, from the original repo.

4. `python extract_features.py -mode rgb -load_model models/rgb_imagenet.pt -root /ssd_scratch/cvit/varun/output_frames_cmd/ -gpu 0 -save_dir /ssd_scratch/cvit/varun/i3d_feats`


