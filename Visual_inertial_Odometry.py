import cv2
import numpy as np
import time
import argparse

class VisualOdometry:
    def __init__(self, video_path, calib_path, resize_dims=(300, 300)):
        """
        Initialize the VisualOdometry object with video path, camera calibration, and resize dimensions.
        Loads camera intrinsics and sets all hyperparameters for optical flow, preprocessing, and visualization.
        """
        self.VIDEO_PATH = video_path  # Path to the input video file
        self.CAMERA_HEIGHT_M = 0.75  # Height of the camera from the ground in meters (used for scale)
        self.RESIZE_DIMS = resize_dims  # Tuple for resizing frames (width, height)

        self.K = np.loadtxt(calib_path)  # Camera intrinsic matrix loaded from file

        # DIS optical flow settings (controls the accuracy and speed of flow estimation)
        self.DIS_SETTINGS = {
            'preset': cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,  # Preset for DIS optical flow (accuracy/speed tradeoff)
            'patch_size': 8,      # Size of the patch used for matching
            'patch_stride': 4,    # Stride between patches
            'gd_iters': 16,       # Number of gradient descent iterations
            'var_iters': 5,       # Number of variational refinement iterations
        }

        # Sampling & outlier rejection
        self.GRID_SPACING = 20  # Distance (in pixels) between sampled flow points
        self.OUTLIER_THRESHOLD = 3.0  # Threshold for outlier rejection in robust mean

        # Preprocessing options
        self.ENABLE_PREPROCESSING = True  # Whether to apply preprocessing to frames
        self.CLAHE_LIMIT = 2.0            # Contrast limit for CLAHE (adaptive histogram equalization)
        self.CLAHE_TILE = (8, 8)          # Tile grid size for CLAHE
        self.BILATERAL = {'d': 9, 'sigma_color': 75, 'sigma_space': 75}  # Bilateral filter params
        self.UNSHARP_AMOUNT = 1.5         # Amount for unsharp masking (sharpening)
        self.PAD = 20                     # Padding size for border reflection

        # Region of interest
        self.ENABLE_ROI = True    # Whether to use a central region of interest
        self.ROI_SCALE = 0.6      # Fraction of the frame to keep as ROI (centered)

        # Visualization constants
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for on-screen text
        self.FLOW_WINDOW = (800, 600)         # Window size for flow visualization
        self.TRAJ_WINDOW = (600, 600)         # Window size for trajectory plot

    def extract_roi(self, frame, scale=None):
        """
        Extracts a central region of interest (ROI) from the frame based on the scale.
        Returns the ROI and its position in the original frame.
        """
        if scale is None:
            scale = self.ROI_SCALE
            
        h, w = frame.shape[:2]
        if not self.ENABLE_ROI or scale >= 1.0:
            return frame.copy(), (0, 0, w, h)

        rh, rw = int(h * scale), int(w * scale)
        y_off = (h - rh) // 2
        x_off = (w - rw) // 2
        roi = frame[y_off:y_off+rh, x_off:x_off+rw].copy()
        return roi, (x_off, y_off, rw, rh)

    def preprocess(self, frame):
        """
        Applies preprocessing steps to the input frame:
        - Border reflection padding
        - Bilateral filtering for noise reduction
        - CLAHE for contrast enhancement
        - Unsharp masking for sharpening
        Returns the processed frame.
        """
        if not self.ENABLE_PREPROCESSING:
            return frame

        pad = self.PAD
        img = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        img = cv2.bilateralFilter(img, self.BILATERAL['d'], self.BILATERAL['sigma_color'], self.BILATERAL['sigma_space'])
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_LIMIT, tileGridSize=self.CLAHE_TILE)
        img = clahe.apply(img)

        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        img = cv2.addWeighted(img, 1 + self.UNSHARP_AMOUNT, blur, -self.UNSHARP_AMOUNT, 0)

        return img[pad:-pad, pad:-pad]

    def pixel_to_world(self, u, v, K_inv, height, offset):
        """
        Converts pixel coordinates (u, v) to world coordinates at a given camera height.
        Uses the inverse camera matrix and ROI offset.
        """
        u_full = u + offset[0]
        v_full = v + offset[1]
        uv1 = np.array([u_full, v_full, 1.0], dtype=np.float64)
        ray = K_inv @ uv1
        scale = height / ray[2]
        return ray * scale

    def robust_mean(self, vectors):
        """
        Computes a robust mean of a set of 2D vectors by removing outliers using the median absolute deviation.
        Returns the mean of inlier vectors.
        """
        if len(vectors) == 0:
            return np.zeros(2)

        med = np.median(vectors, axis=0)
        mad = np.median(np.abs(vectors - med), axis=0) + 1e-9
        mask = np.all(np.abs(vectors - med) / mad < self.OUTLIER_THRESHOLD, axis=1)
        return np.mean(vectors[mask], axis=0) if np.any(mask) else med

    def draw_trajectory(self, points, size=None):
        """
        Draws the trajectory of the camera on a blank canvas.
        Plots the path, start, and end points.
        Returns the visualization image.
        """
        if size is None:
            size = self.TRAJ_WINDOW
            
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        if len(points) < 2:
            return canvas

        arr = np.array(points)
        xs, ys = arr[:, 0], arr[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        pad = 50
        xrange = max(0.1, max_x - min_x)
        yrange = max(0.1, max_y - min_y)

        scale = min((size[0] - 2*pad)/xrange, (size[1] - 2*pad)/yrange)
        center_x = (min_x + max_x)/2
        center_y = (min_y + max_y)/2

        mapped = [(
            int(size[0]/2 + (x - center_x)*scale),
            int(size[1]/2 + (y - center_y)*scale)
        ) for x, y in points]

        for x in range(pad, size[0]-pad, 50):
            cv2.line(canvas, (x, pad), (x, size[1]-pad), (50,50,50), 1)
        for y in range(pad, size[1]-pad, 50):
            cv2.line(canvas, (pad, y), (size[0]-pad, y), (50,50,50), 1)

        for i in range(1, len(mapped)):
            cv2.line(canvas, mapped[i-1], mapped[i], (0,0,255), 2)
        cv2.circle(canvas, mapped[0], 8, (0,255,0), -1)
        cv2.circle(canvas, mapped[-1], 8, (255,0,0), -1)
        return canvas

    def visualize(self, frame, flow, roi_info):
        """
        Overlays the optical flow vectors on the frame for visualization.
        Draws arrows for sampled flow and the ROI rectangle.
        Returns the visualization image.
        """
        vis = frame.copy()
        h, w = flow.shape[:2]
        x_off, y_off = roi_info[:2]

        for y in range(0, h, self.GRID_SPACING):
            for x in range(0, w, self.GRID_SPACING):
                dx, dy = flow[y, x]
                if np.hypot(dx, dy) > 1.0:
                    pt1 = (x + x_off, y + y_off)
                    pt2 = (int(x + dx + x_off), int(y + dy + y_off))
                    cv2.arrowedLine(vis, pt1, pt2, (0,255,0), 1, tipLength=0.3)
        if self.ENABLE_ROI:
            x, y, rw, rh = roi_info
            cv2.rectangle(vis, (x, y), (x+rw, y+rh), (255,255,0), 2)
        return vis

    def run(self):
        """
        Main loop for visual odometry:
        - Reads video frames
        - Preprocesses and extracts ROI
        - Computes optical flow between frames
        - Estimates camera motion and updates trajectory
        - Visualizes flow and trajectory in real time
        - Handles user input for quitting or resetting
        - Saves the final trajectory to a file
        """
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.VIDEO_PATH}")

        # Prepare intrinsics, optionally scaled if resizing
        ret, first_frame = cap.read()
        if not ret:
            raise IOError("Cannot read first frame.")
        orig_h, orig_w = first_frame.shape[:2]

        if self.RESIZE_DIMS is not None:
            rw, rh = self.RESIZE_DIMS
            scale_x = rw / orig_w
            scale_y = rh / orig_h
            # adjust intrinsics
            K_scaled = self.K.copy()
            K_scaled[0,0] *= scale_x
            K_scaled[0,2] *= scale_x
            K_scaled[1,1] *= scale_y
            K_scaled[1,2] *= scale_y
            K_inv = np.linalg.inv(K_scaled)
            first_frame = cv2.resize(first_frame, (rw, rh))
        else:
            K_inv = np.linalg.inv(self.K)
            rw, rh = orig_w, orig_h

        flow_engine = cv2.DISOpticalFlow_create(self.DIS_SETTINGS['preset'])
        flow_engine.setPatchSize(self.DIS_SETTINGS['patch_size'])
        flow_engine.setPatchStride(self.DIS_SETTINGS['patch_stride'])
        flow_engine.setGradientDescentIterations(self.DIS_SETTINGS['gd_iters'])
        flow_engine.setVariationalRefinementIterations(self.DIS_SETTINGS['var_iters'])

        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_proc = self.preprocess(prev_gray)

        cum_xy = np.zeros(2)
        traj = [cum_xy.copy()]
        total_dist = 0.0
        fps_hist = []
        prev_time = time.time()

        cv2.namedWindow("Flow VO", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Trajectory", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Flow VO", *self.RESIZE_DIMS)  # Match window to resized frame
        cv2.resizeWindow("Trajectory", *self.TRAJ_WINDOW)


        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.RESIZE_DIMS is not None:
                frame = cv2.resize(frame, (rw, rh))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_proc = self.preprocess(gray)

            prev_roi, prev_info = self.extract_roi(prev_proc)
            curr_roi, curr_info = self.extract_roi(curr_proc)
            flow = flow_engine.calc(prev_roi, curr_roi, None)

            ys, xs = np.mgrid[
                self.GRID_SPACING//2:prev_roi.shape[0]:self.GRID_SPACING,
                self.GRID_SPACING//2:prev_roi.shape[1]:self.GRID_SPACING
            ].reshape(2, -1).astype(int)
            pts = np.stack([xs, ys], axis=-1)
            disp = []
            for (u, v), (du, dv) in zip(pts, flow[ys, xs]):
                if np.isnan(du) or np.isnan(dv):
                    continue
                p0 = self.pixel_to_world(u, v, K_inv, self.CAMERA_HEIGHT_M, prev_info[:2])
                p1 = self.pixel_to_world(u+du, v+dv, K_inv, self.CAMERA_HEIGHT_M, prev_info[:2])
                disp.append(p1[:2] - p0[:2])
            disp = np.array(disp)

           
            mean_disp = self.robust_mean(disp)
            move = -mean_disp
            step = np.linalg.norm(move)
            total_dist += step
            cum_xy += move
            traj.append(cum_xy.copy())

            now = time.time()
            fps = 1/(now - prev_time)
            prev_time = now
            fps_hist.append(fps)
            if len(fps_hist) > 10:
                fps_hist.pop(0)
            avg_fps = sum(fps_hist)/len(fps_hist)

            vis = self.visualize(frame, flow, curr_info)
            dist = np.linalg.norm(cum_xy)
            txt = [
                f"Frame: {idx}",
                f"FPS: {avg_fps:.2f}",
                f"Direct: {dist:.3f} m",   
                f"Total: {total_dist:.3f} m",
                f"Pos: ({cum_xy[0]:.3f},{cum_xy[1]:.3f}) m"
            ]
            for i, t in enumerate(txt):
                cv2.putText(vis, t, (20, 40 + i*35), self.FONT, 0.8, (0,255,255), 2)

            traj_img = self.draw_trajectory(traj)
            cv2.imshow("Flow VO", vis)
            cv2.imshow("Trajectory", traj_img)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('r'):
                cum_xy = np.zeros(2)
                traj = [cum_xy.copy()]
                total_dist = 0.0

            prev_proc = curr_proc
            idx += 1

        cap.release()
        cv2.destroyAllWindows()

        print(f"Processed {idx} frames")
        final_dist = np.linalg.norm(cum_xy)
        print(f"Direct distance: {final_dist:.3f} m, Total distance: {total_dist:.3f} m")

        arr = np.array(traj)
        fname = f"trajectory_roi_{int(self.ROI_SCALE*100)}.txt" if self.ENABLE_ROI else "trajectory.txt"
        np.savetxt(fname, arr, fmt='%.6f', delimiter=',', header='x,y')
        print(f"Saved trajectory: {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Odometry using DIS Optical Flow")
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--calib', type=str, required=True, help='Path to camera calibration .npy file')
    parser.add_argument('--resize', type=int, nargs=2, default=[300, 300], help='Resize dimensions (width height)')
    args = parser.parse_args()

    vo = VisualOdometry(args.video, args.calib, tuple(args.resize))
    vo.run()


