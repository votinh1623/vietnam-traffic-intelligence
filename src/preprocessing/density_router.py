import cv2
import numpy as np
from collections import deque


class DensityRouter:
    def __init__(
        self,
        roi_mask=None,
        density_threshold=0.85,
        grid_rows=8,
        grid_cols=10,
        cell_density_threshold=0.7,
        cell_ratio_threshold=0.8,
        confirm_time=3.0,
        normal_confirm_time=5.0,
    ):
        # MOG2 với history lớn, ngưỡng thấp để nhạy hơn với thay đổi nhỏ
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=1000, varThreshold=25, detectShadows=False
        )
        self.roi_mask = roi_mask
        self.density_threshold = density_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_density_threshold = cell_density_threshold
        self.cell_ratio_threshold = cell_ratio_threshold
        self.confirm_time = confirm_time
        self.normal_confirm_time = normal_confirm_time

        self.state_history = deque()
        self.current_mode = "NORMAL"
        self.prev_frame = None  # để tính frame differencing

    def compute_density(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. MOG2 mask
        mog_mask = self.mog2.apply(frame, learningRate=0.001)  # learning rate rất thấp

        # 2. Frame differencing (absdiff) để bắt thay đổi giữa các frame
        if self.prev_frame is not None:
            diff = cv2.absdiff(gray, self.prev_frame)
            _, diff_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        else:
            diff_mask = np.zeros_like(mog_mask)
        self.prev_frame = gray

        # 3. Kết hợp hai mask: foreground là vùng có trong MOG2 HOẶC có thay đổi gần đây
        combined = cv2.bitwise_or(mog_mask, diff_mask)

        if self.roi_mask is not None:
            road_pixels = np.count_nonzero(self.roi_mask)
            fg_pixels = np.count_nonzero(
                cv2.bitwise_and(combined, combined, mask=self.roi_mask)
            )
        else:
            road_pixels = combined.size
            fg_pixels = np.count_nonzero(combined)
        density = fg_pixels / road_pixels if road_pixels > 0 else 0
        return density, combined

    def analyze_density_grid(self, fg_mask):
        h, w = fg_mask.shape
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        high_density_cells = 0
        total_cells = 0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                cell_mask = fg_mask[y1:y2, x1:x2]
                if self.roi_mask is not None:
                    cell_roi = self.roi_mask[y1:y2, x1:x2]
                    road_pixels = np.count_nonzero(cell_roi)
                    if road_pixels == 0:
                        continue
                    fg_pixels = np.count_nonzero(
                        cv2.bitwise_and(cell_mask, cell_mask, mask=cell_roi)
                    )
                    cell_density = fg_pixels / road_pixels
                else:
                    cell_density = np.count_nonzero(cell_mask) / cell_mask.size
                total_cells += 1
                if cell_density > self.cell_density_threshold:
                    high_density_cells += 1

        cell_ratio = high_density_cells / total_cells if total_cells > 0 else 0
        return cell_ratio

    def route(self, frame, timestamp=None):
        if timestamp is None:
            timestamp = cv2.getTickCount() / cv2.getTickFrequency()

        density, fg_mask = self.compute_density(frame)
        cell_ratio = self.analyze_density_grid(fg_mask)

        is_jam_candidate = (density > self.density_threshold) and (
            cell_ratio > self.cell_ratio_threshold
        )

        self.state_history.append((timestamp, is_jam_candidate))
        while self.state_history and (
            timestamp - self.state_history[0][0] > self.confirm_time + 5
        ):
            self.state_history.popleft()

        current_state = is_jam_candidate
        duration = 0.0
        for t, state in reversed(self.state_history):
            if state == current_state:
                duration = timestamp - t
            else:
                break

        if (
            self.current_mode == "NORMAL"
            and current_state
            and duration >= self.confirm_time
        ):
            self.current_mode = "JAM"
        elif (
            self.current_mode == "JAM"
            and not current_state
            and duration >= self.normal_confirm_time
        ):
            self.current_mode = "NORMAL"

        return self.current_mode, density, cell_ratio, fg_mask
