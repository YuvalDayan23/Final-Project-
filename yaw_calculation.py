# Description: This script reads IMU data from a CSV file and performs sensor fusion to calculate yaw angle.
#              It then performs linear regression on yaw and corrects the yaw values.
#              The corrected yaw values are plotted with annotations at selected timestamps.

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Fusion:
    """
    Madgwick-based sensor fusion for pitch, roll, and yaw.
    Uses the original code's numeric values and offsets for pitch & roll,
    plus a standard quaternion-to-yaw formula (no offset).
    """

    def __init__(self, gyro_error, sample_dur):
        """
        Initializes the fusion system.

        Parameters:
        gyro_error : float
            Estimated gyro error in deg/s.
        sample_dur : float
            Sample duration (time between measurements).
        """

        # Store sample duration
        self.sample_dur = sample_dur

        # Initial quaternion is identity (no rotation)
        self.q = [1.0, 0.0, 0.0, 0.0]

        # Madgwick beta calculation
        gyro_mean_error = np.radians(gyro_error)
        self.beta = np.sqrt(3.0 / 4.0) * gyro_mean_error

        # Initialize pitch, roll, yaw in degrees
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0

    def update(self, accel, gyro):
        """
        Update the orientation from a single sample of accel (g) and gyro (deg/s).

        Parameters:
        accel : array-like of shape (3,)
            Acceleration in (x, y, z) [g].
        gyro : array-like of shape (3,)
            Gyro measurements in (x, y, z) [deg/s].
        """

        # Unpack accelerometer and gyro data
        ax, ay, az = accel
        gx, gy, gz = (np.radians(val) for val in gyro)  # convert deg/s -> rad/s

        # Unpack current quaternion
        q1, q2, q3, q4 = self.q

        # Normalize accelerometer
        norm = np.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0:
            return
        norm = 1.0 / norm
        ax *= norm
        ay *= norm
        az *= norm

        # Gradient descent step
        _2q1, _2q2, _2q3, _2q4 = 2*q1, 2*q2, 2*q3, 2*q4
        q1q1, q2q2, q3q3, q4q4 = q1*q1, q2*q2, q3*q3, q4*q4

        s1 = 4*q1*q3q3 + 2*q3*ax + 4*q1*q2q2 - 2*q2*ay
        s2 = (4*q2*q4q4 - 2*q4*ax
              + 4*q1q1*q2 - 2*q1*ay - 4*q2
              + 8*q2*q2q2 + 8*q2*q3q3 + 4*q2*az)
        s3 = (4*q1q1*q3 + 2*q1*ax + 4*q3*q4q4 - 2*q4*ay
              - 4*q3 + 8*q3*q2q2 + 8*q3*q3q3 + 4*q3*az)
        s4 = 4*q2q2*q4 - 2*q2*ax + 4*q3q3*q4 - 2*q3*ay

        norm = 1.0 / np.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

        # Rate of change of quaternion from gyro + gradient
        q_dot1 = 0.5 * (-q2*gx - q3*gy - q4*gz) - self.beta*s1
        q_dot2 = 0.5 * ( q1*gx + q3*gz - q4*gy) - self.beta*s2
        q_dot3 = 0.5 * ( q1*gy - q2*gz + q4*gx) - self.beta*s3
        q_dot4 = 0.5 * ( q1*gz + q2*gy - q3*gx) - self.beta*s4

        # Integrate to get new quaternion
        q1 += q_dot1 * self.sample_dur
        q2 += q_dot2 * self.sample_dur
        q3 += q_dot3 * self.sample_dur
        q4 += q_dot4 * self.sample_dur

        # Normalize quaternion
        norm = 1.0 / np.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
        self.q = (q1*norm, q2*norm, q3*norm, q4*norm)

        # Update pitch, roll, yaw in degrees
        self.roll = (
            np.degrees(
                -np.arcsin(2.0 * (self.q[1]*self.q[3] - self.q[0]*self.q[2]))
            ) + 7
        )
        self.pitch = (
            np.degrees(
                np.arctan2(
                    2.0 * (self.q[0]*self.q[1] + self.q[2]*self.q[3]),
                    (self.q[0]*self.q[0] - self.q[1]*self.q[1]
                     - self.q[2]*self.q[2] + self.q[3]*self.q[3])
                )
            ) + 90
        )
        q0, q1_, q2_, q3_ = self.q
        self.yaw = np.degrees(
            np.arctan2(
                2.0*(q1_*q2_ + q0*q3_),
                (q0*q0 + q1_*q1_ - q2_*q2_ - q3_*q3_)
            )
        )

def read_imu_data(csv_path):
    """
    Reads a CSV file with columns:
      'timestamp [ns]', 'gyro x [deg/s]', 'gyro y [deg/s]', 'gyro z [deg/s]',
      'acceleration x [g]', 'acceleration y [g]', 'acceleration z [g]'.

    Returns:
      time_data (1D np.array in seconds)
      gyro_data (N×3) in deg/s
      accel_data (N×3) in g
    """
    t_list = []
    gyro_list = []
    accel_list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert nanoseconds -> seconds
            t_sec = float(row['timestamp [ns]']) / 1e9

            gx = float(row['gyro x [deg/s]'])
            gy = float(row['gyro y [deg/s]'])
            gz = float(row['gyro z [deg/s]'])

            ax = float(row['acceleration x [g]'])
            ay = float(row['acceleration y [g]'])
            az = float(row['acceleration z [g]'])

            t_list.append(t_sec)
            gyro_list.append([gx, gy, gz])
            accel_list.append([ax, ay, az])

    return np.array(t_list), np.array(gyro_list), np.array(accel_list)

def main():
    """
    Main function:
      1. Reads the CSV data.
      2. Shifts time so first sample is t=0.
      3. Creates a Madgwick fusion object.
      4. Calculates pitch, roll, and yaw for each sample.
      5. Performs linear regression on yaw.
      6. Plots original yaw and corrected yaw (with annotations).
    """

    # -- 1. Specify your CSV path --
    csv_file = 'add your csv file here'  # Add your CSV file here
    data = pd.read_csv(csv_file)

    # -- 2. Read IMU data from CSV --
    time_data, gyro_data, accel_data = read_imu_data(csv_file)

    # Shift time so first sample is at t=0
    time_data = time_data - time_data[0]

    # -- 3. Create Madgwick fusion object --
    gyro_error = 50  # deg/s, can be tuned
    sample_time = np.mean(np.diff(time_data))
    fuser = Fusion(gyro_error, sample_time)

    # Prepare lists to store fused angles
    pitch_vals = []
    roll_vals = []
    yaw_vals = []

    # -- 4. Run the fusion for each sample --
    for i in range(len(time_data)):
        fuser.update(accel_data[i], gyro_data[i])
        pitch_vals.append(fuser.pitch)
        roll_vals.append(fuser.roll)
        yaw_vals.append(fuser.yaw)

    # -- 5. Plot roll, pitch, and yaw (3 subplots) --
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 10))

    # Roll
    roll = data['roll [deg]']
    ax1.plot(time_data, roll, color="red", label="Roll")
    ax1.set_title("Roll vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Roll (degrees)")
    ax1.grid(True)
    ax1.legend()

    # Pitch
    pitch = data['pitch [deg]']
    ax2.plot(time_data, pitch, color="green", label="Pitch")
    ax2.set_title("Pitch vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Pitch (degrees)")
    ax2.grid(True)
    ax2.legend()

    # Yaw
    ax3.plot(time_data, yaw_vals, color="blue", label="Yaw")
    ax3.set_title("Yaw vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Yaw (degrees)")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # -- 6. Add linear regression to yaw --
    # Choose time range for linear regression
    t1, t2 = 'add your times here'  # Add your times here
    yaw1 = np.interp(t1, time_data, yaw_vals)
    yaw2 = np.interp(t2, time_data, yaw_vals)

    # Compute slope and intercept
    m = (yaw2 - yaw1) / (t2 - t1)
    b = yaw1 - m * t1
    regression_line = m * time_data + b

    # Plot original yaw and regression line
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, yaw_vals, label='Yaw', color='r')
    plt.plot(time_data, regression_line, label='Linear Regression', color='b', linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw Angle [deg]')
    plt.title('Yaw vs Time with Linear Regression')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- 7. Plot yaw (corrected by subtracting the regression) --
    yaw_vals_cor = yaw_vals - regression_line
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, yaw_vals, label='Yaw', color='r')
    plt.plot(time_data, yaw_vals_cor, label='Yaw corrected', color='g')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw Angle [deg]')
    plt.title('Yaw vs Time (Corrected)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -- 8. Plot corrected yaw with annotations --
    timestamps = ['add your timestamps here']  # Add your timestamps here
    # Interpolate corrected yaw at the given timestamps
    yaw_values = np.interp(timestamps, time_data, yaw_vals_cor)

    plt.figure(figsize=(10, 6))
    plt.plot(time_data, yaw_vals_cor, label='Yaw corrected', color='g')
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw Angle [deg]')
    plt.title('Yaw vs Time (Corrected) with Annotations')
    plt.grid(True)
    plt.legend()

    # Add numeric annotations at selected timestamps
    for t, yaw_v in zip(timestamps, yaw_values):
        plt.text(t, yaw_v, f'{yaw_v:.1f}°', color='blue', fontsize=9, ha='center')

    # Set x-axis ticks every 5 seconds for convenience
    plt.xticks(np.arange(0, max(time_data), 5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
