#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from scipy.ndimage import gaussian_filter1d

class ReactiveFollowGap(Node):
    """ 
    Implement Gap Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.threshold_val = 0.4
        self.car_width = 0.13

        # Speed-related variables for dynamic inflation
        self.current_speed = 0.3   # Starting speed
        self.max_speed = 1.5      # Maximum speed for normal driving

        self.lidar_sub = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )
        

    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array:
            1. Smoothing the data (Gaussian).
            2. Clipping max distances to 3m.
        """
        proc_ranges = self.gaussian_smooth_lidar(ranges)
        proc_ranges = np.clip(proc_ranges, None, 3)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        return None
    
    def find_best_point(self, start_i, end_i, ranges):
        """ Naive approach: choose the furthest point within the gap.
        """
        return None

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped message.
        """
        # 1. Grab raw LiDAR data and preprocess
        ranges = data.ranges
        proc_ranges = self.preprocess_lidar(ranges)
        proc_ranges = self.apply_weighted_depths(proc_ranges)
        
        # 2. Find disparities (indices where difference is greater than threshold)
        disparities = np.where(np.abs(np.diff(proc_ranges)) > self.threshold_val)[0]
        readings_at_disp = [proc_ranges[idx] for idx in disparities.tolist()]
        reading_dict = dict(zip(disparities, readings_at_disp))

        # 3. Average out consecutive disparities to reduce noise
        disparities = self.average_consecutive_disparities(disparities, proc_ranges)
        # self.get_logger().info(f'Disparities at {reading_dict}')

        # 4. Extend disparities to account for vehicle width
        for disparity_idx in disparities:
            mask = self.disparity_extender(proc_ranges, disparity_idx)
            proc_ranges[mask] = 0

        # 5. Find the largest non-zero gap & compute steering angle
        gap_steering_angle, gap_idx = self.find_largest_nonzero_subsequence(proc_ranges)

        # FIX: Cast gap_idx to int so NumPy uses it without error
        gap_idx = int(gap_idx)

        gap_distance = proc_ranges[gap_idx] if 0 <= gap_idx < len(proc_ranges) else 0.0

        base_speed = 0.3
        distance_factor = min(gap_distance / 3.0, 1.0)
        angle_factor = 1.0 - (abs(gap_steering_angle) / 0.7)
        angle_factor = max(min(angle_factor, 1.0), 0.1)
        new_speed = base_speed + 0.6 * distance_factor + 0.6 * angle_factor
        new_speed = min(new_speed, self.max_speed)


        self.get_logger().info(f"""
            Current Speed: {str(self.current_speed)},
            Current Turn Angle: {str(gap_steering_angle)},
            Chosen Gap idx: {str(gap_idx)},
            Chosen Gap Distance: {str(gap_distance)}

        """)
        
        # Publish the drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = float(new_speed)
        drive_msg.drive.steering_angle = float(gap_steering_angle)
        self.drive_pub.publish(drive_msg)

        

    def gaussian_smooth_lidar(self, lidar_data, sigma=0.5):
        return gaussian_filter1d(lidar_data, sigma=sigma)

    def disparity_extender(self, lidar_data, disparity_idx):
        """
        Extends the disparity region in both directions by an amount
        proportional to car_width, factoring in a speed-based inflation.
        """
        speed_ratio = self.current_speed / self.max_speed
        dynamic_inflation = 1.352 + speed_ratio

        delta_theta = np.radians(0.25)
        arc_lengths = lidar_data * delta_theta
        mask = np.zeros(len(lidar_data), dtype=bool)

        # Extend towards the right (lower indices)
        if disparity_idx - 1 > 0:
            accumulated_arc_length = np.cumsum(arc_lengths[:disparity_idx - 1][::-1])
            right_indices = np.where(accumulated_arc_length < self.car_width * dynamic_inflation)[0]
            mask[disparity_idx - right_indices - 1] = True

        # Extend towards the left (higher indices)
        accumulated_arc_length = np.cumsum(arc_lengths[disparity_idx:])
        left_indices = np.where(accumulated_arc_length < self.car_width * dynamic_inflation)[0]
        mask[disparity_idx + left_indices] = True
        
        return mask

    def find_largest_nonzero_subsequence(self, lidar_data, alpha=0.5, beta=0.5, gamma=0.1, wheelbase=0.25, lookahead_scale=4.00):
        """
        Identifies the largest valid gap in LiDAR data and returns a steering angle
        and index for navigation. The logic remains the same except for dynamic
        inflation referencing self.current_speed and self.max_speed.
        """
        nonzero_indices = np.where(lidar_data > 0)[0]
        if len(nonzero_indices) == 0:
            return 0, 0

        breaks = np.where(np.diff(nonzero_indices) > 1)[0]
        segment_starts = np.insert(nonzero_indices[breaks + 1], 0, nonzero_indices[0])
        segment_ends = np.append(nonzero_indices[breaks], nonzero_indices[-1])
        segment_lengths = segment_ends - segment_starts
        segment_center_offsets = np.abs((segment_starts + segment_ends) // 2 - 540)

        arc_lengths = lidar_data * np.radians(0.25)
        segment_gap_widths = []
        for start, end, gap_length in zip(segment_starts, segment_ends, segment_lengths):
            if gap_length == 0:
                accumulated_arc_length = 0
            else:
                accumulated_arc_length = np.cumsum(arc_lengths[start:end])[gap_length - 1]
            segment_gap_widths.append(accumulated_arc_length)
        segment_gap_widths = np.array(segment_gap_widths)

        speed_ratio = self.current_speed / self.max_speed
        gap_inflation_factor = 2.0 + speed_ratio

        valid_mask = segment_gap_widths > (self.car_width * gap_inflation_factor)
        segment_starts = segment_starts[valid_mask]
        segment_ends = segment_ends[valid_mask]
        segment_gap_widths = segment_gap_widths[valid_mask]
        segment_center_offsets = segment_center_offsets[valid_mask]

        segment_depths = np.array([np.mean(lidar_data[start:end + 1]) for start, end in zip(segment_starts, segment_ends)])
        segment_deepest = np.array([
            np.max(lidar_data[start:end]) if lidar_data[start:end].size > 0 else 0
            for start, end in zip(segment_starts, segment_ends)
        ])
        segments_variability = np.array([np.std(lidar_data[start:end]) for start, end in zip(segment_starts, segment_ends)])

        if segment_gap_widths.size == 0:
            return 3, 540

        denominator = segment_gap_widths.max() - segment_gap_widths.min()
        segment_gap_widths_norm = (segment_gap_widths - segment_gap_widths.min()) / (denominator + 1e-6)
        segment_deepest_norm = (segment_deepest - segment_deepest.min()) / (segment_deepest.max() - segment_deepest.min())
        segment_depths_norm = (segment_depths - segment_depths.min()) / (segment_depths.max() - segment_depths.min())
        segments_variability_norm = (
            (segments_variability - segments_variability.min()) /
            (segments_variability.max() - segments_variability.min())
        )
        segment_lengths_norm = (segment_lengths - segment_lengths.min()) / (segment_lengths.max() - segment_lengths.min())
        segment_center_offsets_norm = (
            (segment_center_offsets - segment_center_offsets.min()) /
            (segment_center_offsets.max() - segment_center_offsets.min() + 1e-6)
        )

        weighted = (
            alpha * segment_depths_norm +
            (1 - alpha) * segment_deepest_norm +
            (1 - alpha - beta) * segment_gap_widths_norm -
            0.6 * segment_center_offsets_norm
        )

        best_idx = np.argmax(abs(weighted))

        safety_offset = int((self.car_width) / np.radians(0.25))
        safe_start = max(segment_starts[best_idx], segment_starts[best_idx] + safety_offset)
        safe_end = min(segment_ends[best_idx], segment_ends[best_idx] - safety_offset)
        lower_idx = min(safe_start, safe_end)
        upper_idx = max(safe_start, safe_end)
        deepest_slice = lidar_data[lower_idx:upper_idx]

        if deepest_slice.size > 0:
            deepest_reading = np.max(deepest_slice)
            percentile_val = np.percentile(deepest_slice, 95)
            largest_gap_in_slice = np.where(deepest_slice >= percentile_val)[0]

            subgap_breaks = np.where(np.diff(largest_gap_in_slice) > 1)[0]
            subgap_break_starts = np.insert(largest_gap_in_slice[subgap_breaks + 1], 0, largest_gap_in_slice[0])
            subgap_break_ends = np.append(largest_gap_in_slice[subgap_breaks], largest_gap_in_slice[-1])
            subgap_lengths = subgap_break_ends - subgap_break_starts
            largest_subgap_idx = np.argmax(subgap_lengths)

            max_start = subgap_break_starts[largest_subgap_idx]
            max_end = subgap_break_ends[largest_subgap_idx]

            middle_idx = max(segment_starts[best_idx], segment_starts[best_idx] + safety_offset) + (max_start + max_end) // 2

            angle = -135 + (middle_idx * 0.25)
            angle_rad = np.radians(angle)
            depth = lidar_data[middle_idx]

            if depth <= 0:
                return 0.0, 0.0

            x = depth * np.cos(angle_rad)
            y = depth * np.sin(angle_rad)
            lookahead = max(depth * lookahead_scale, wheelbase * 2)
            steering_angle = np.arctan2(2 * lookahead * y, x**2 + y**2)

            return steering_angle, middle_idx

        return 3, 540

    def lidar_index_to_angle(self, lidar_index):
        return (lidar_index * 0.25) - 135

    def average_consecutive_disparities(self, disparity_indices, lidar_data):
        if len(disparity_indices) == 0:
            return []
        breaks = np.where(np.diff(disparity_indices) > 1)[0]
        groups = np.split(disparity_indices, breaks + 1)

        def find_largest_disparity(group):
            if len(group) == 1:
                return group[0]
            depth_diffs = np.abs(np.diff(lidar_data[group]))
            max_diff_idx = np.argmax(depth_diffs)
            return group[max_diff_idx]

        averaged_disparities = [find_largest_disparity(g) for g in groups if len(g) > 0]
        return averaged_disparities

    def compute_center_weighting(self, lidar_data, center_idx=540, sigma=300):
        """
        Computes weights for LiDAR readings, favoring those near the center.
        """
        indices = np.arange(len(lidar_data))
        weights = np.exp(-((indices - center_idx) ** 2) / (2 * sigma ** 2))
        return weights

    def apply_weighted_depths(self, lidar_data, center_idx=540, sigma=350):
        """
        Applies a weighting function to LiDAR depths, emphasizing center readings.
        """
        weights = self.compute_center_weighting(lidar_data, center_idx, sigma)
        weighted_depths = lidar_data * weights
        return weighted_depths


def main(args=None):
    rclpy.init(args=args)
    print("Gap Follow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
