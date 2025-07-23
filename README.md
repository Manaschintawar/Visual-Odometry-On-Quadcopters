# Visual Inertial Odometry on Quadcopters

This project implements a simple visual odometry pipeline using optical flow for quadcopter videos. It estimates the camera's trajectory by analyzing the motion between video frames.

## Features

- Uses DIS optical flow for robust motion estimation.
- Supports preprocessing (contrast enhancement, denoising, sharpening).
- Allows region of interest (ROI) selection for more stable tracking.
- Visualizes both the optical flow and the estimated trajectory in real time.
- Saves the trajectory as a txt file.

## Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the script from the command line:

```bash
python Visual_inertial_Odometry.py --video <path_to_video> --calib <path_to_calibration_file> --resize WIDTH HEIGHT
```

### Arguments

- `--video`  
  Path to the input video file (required).

- `--calib`  
  Path to the camera calibration file (required).  
  The file should be a text file containing the camera intrinsic matrix (3x3).

- `--resize WIDTH HEIGHT`  
  (Optional) Resize the video frames to the given width and height.  
  Default: `300 300`

### Example

```bash
python Visual_inertial_Odometry.py --video data/quadcopter.mp4 --calib data/camera_calib.txt --resize 400 400
```

## Controls

- Press `q` or `Esc` to quit.
- Press `r` to reset the trajectory.

## Output

- The estimated trajectory is saved as a CSV file (`trajectory.txt` or `trajectory_roi_XX.txt`) after processing.

---

## Code Structure and Function Hierarchy

The main logic is encapsulated in the `VisualOdometry` class. Here is a brief overview of the functions and their roles:

- **`__init__`**: Initializes all hyperparameters, loads camera intrinsics, and sets up preprocessing and visualization options.
- **`extract_roi(frame, scale=None)`**: Extracts a central region of interest (ROI) from the frame for more stable flow estimation.
- **`preprocess(frame)`**: Applies preprocessing (padding, denoising, contrast enhancement, sharpening) to each frame.
- **`pixel_to_world(u, v, K_inv, height, offset)`**: Converts pixel coordinates to world coordinates using camera intrinsics and known camera height.
- **`robust_mean(vectors)`**: Computes a robust mean of 2D vectors by removing outliers.
- **`draw_trajectory(points, size=None)`**: Draws the estimated camera trajectory on a blank canvas for visualization.
- **`visualize(frame, flow, roi_info)`**: Overlays optical flow vectors and ROI rectangle on the frame for real-time visualization.
- **`run()`**: Main loop that:
  - Reads video frames
  - Preprocesses and extracts ROI
  - Computes optical flow between frames
  - Estimates camera motion and updates trajectory
  - Visualizes flow and trajectory
  - Handles user input for quitting or resetting
  - Saves the final trajectory to a txt file

### Code Flow

1. **Initialization**: The script parses command-line arguments and creates a `VisualOdometry` object.
2. **Main Loop (`run`)**:
   - Reads and preprocesses frames.
   - Extracts ROI and computes optical flow.
   - Estimates motion and updates trajectory.
   - Visualizes results in real time.
   
3. **Output**: After processing, the trajectory is saved as a txt file.

---
3. **Output**: After processing, the trajectory is saved as a txt file.

---
