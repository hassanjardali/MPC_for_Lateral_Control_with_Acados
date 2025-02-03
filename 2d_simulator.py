import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time 
from car_models import CarKinematicModel, CarDynamicModel
from controllers import PIDController, PurePursuitController
from mpc_controller import MPCController


# Initialization
last_closest_idx = 0
max_data_length = 500 
target_velocity = 75 # Target velocity in m/s

# Function to read the raceline from CSV
def load_raceline(file_path):
    """
    Load raceline from a CSV file. CSV should have columns: x, y, desired_vel.
    """
    return pd.read_csv(file_path)

def animate_simulation(car, path, pid, mpc_controller, dt=0.1, max_time=20.0, search_window=10, num_points = 100, N_pred=45):
    
    # Create a figure with GridSpec for the layout
    fig = plt.figure(figsize=(18, 8))  # Adjust the figure size as needed
    gs = GridSpec(2, 3, height_ratios=[2, 1])  # 2 rows, 3 columns, with different height ratios

    # fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1 = fig.add_subplot(gs[0, :])

    path_points = path.to_numpy()

    # Plot raceline on the main subplot
    ax1.plot(path_points[:, 0], path_points[:, 1], 'r--', label='Raceline')
    car_path, = ax1.plot([], [], 'b-', label='Car Path')
    car_position, = ax1.plot([], [], 'bo', label='Car Position')
    orientation_quiver = ax1.quiver(car.x, car.y, np.cos(car.yaw), np.sin(car.yaw), scale=50, color='blue')
    # Initialize predicted orientation quiver with zeros
    x_pred_initial = np.zeros(N_pred)
    y_pred_initial = np.zeros(N_pred)
    u_pred_initial = np.zeros(N_pred)
    v_pred_initial = np.zeros(N_pred)
    predicted_orientation_quiver = ax1.quiver(
        x_pred_initial, y_pred_initial, u_pred_initial, v_pred_initial,
        scale=80, color='cyan', label='Predicted Orientation'
    )
    # Add a plot for predicted positions
    predicted_line, = ax1.plot([], [], 'c-', label='Predicted Path')


    # Plot for the path segment
    path_segment_line, = ax1.plot([], [], 'go-', label='Path Segment')  # 'go-' for green circles with lines

    # Set axis limits and labels for raceline plot
    ax1.set_xlim(min(path_points[:, 0]) - 10, max(path_points[:, 0]) + 10)
    ax1.set_ylim(min(path_points[:, 1]) - 10, max(path_points[:, 1]) + 10)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.legend()
    ax1.set_title("Real-Time Car Simulation with lateral MPC and PID Controllers")
    ax1.axis('equal')

    # Lower row for velocity, steering angle, and cross-track error (side-by-side plots)
    ax2 = fig.add_subplot(gs[1, 0])  # First column for velocity
    ax3 = fig.add_subplot(gs[1, 1])  # Second column for steering angle
    ax4 = fig.add_subplot(gs[1, 2])  # Third column for cross-track error

    # Data lists for velocity, steering angle, and cross-track error (CTE)
    time_data = []
    current_velocity_data = []
    target_velocity_data = []
    steering_angle_data = []
    cte_data = []

    # Velocity plot
    velocity_line, = ax2.plot([], [], 'b-', label='Current Velocity (m/s)')
    target_velocity_line, = ax2.plot([], [], 'r-', label='Target Velocity (m/s)')
    ax2.set_xlim(0, max_time)
    ax2.set_ylim(0, max(path_points[:, 2]) + 25)  # Adjust y-limit based on target velocity
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.set_title("Velocity Tracking")

    # Steering angle plot
    steering_angle_line, = ax3.plot([], [], 'g-', label='Steering Angle (rad)')
    ax3.set_xlim(0, max_time)
    ax3.set_ylim(-0.1, 0.1)  
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Steering Angle (rad)")
    ax3.legend()
    ax3.set_title("Steering Angle")

    # Cross-track error plot
    cte_line, = ax4.plot([], [], 'm-', label='Cross-Track Error (m)')
    ax4.set_xlim(0, max_time)
    ax4.set_ylim(-2, 2)  
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("CTE (m)")
    ax4.legend()
    ax4.set_title("Cross-Track Error")

    # History of car path
    history = []

    def compute_cte(car, path_points, closest_idx):
        # Car's current position
        car_position = np.array([car.x, car.y])
        
        # Set `point1` and `point2` based on `closest_idx` boundaries efficiently
        if closest_idx == 0:
            point1, point2 = path_points[0, :2], path_points[1, :2]
        elif closest_idx == len(path_points) - 1:
            point1, point2 = path_points[-2, :2], path_points[-1, :2]
        else:
            point1, point2 = path_points[closest_idx - 1, :2], path_points[closest_idx + 1, :2]
        
        # Calculate segment vector and car-to-point1 vector
        segment_vector = point2 - point1
        point_to_car = car_position - point1
        
        # Pre-compute dot products for projection
        segment_length_squared = np.dot(segment_vector, segment_vector)
        
        # Handle edge case where segment length is ze
        if segment_length_squared == 0:
            return np.linalg.norm(point_to_car)
        
        # Compute projection factor (clamped between 0 and 1)
        projection_factor = max(0, min(1, np.dot(point_to_car, segment_vector) / segment_length_squared))
        
        # Closest point on segment and CTE calculation
        closest_point_on_segment = point1 + projection_factor * segment_vector
        cte = np.linalg.norm(car_position - closest_point_on_segment)
        
        return cte



    def update(frame):
        global last_closest_idx  # Use the global variable to keep track of the previous closest index
        start_time = time.time()

        # Determine the search range based on the last closest index
        search_start = max(last_closest_idx - search_window, 0)
        search_end = min(last_closest_idx + search_window, len(path_points))

        # Calculate distances within the search range only
        distances = np.linalg.norm(path_points[search_start:search_end, :2] - np.array([car.x, car.y]), axis=1)
        closest_idx = search_start + np.argmin(distances)  # Get the index within the entire path

        if min(distances) > 10:
            closest_idx = np.argmin(np.linalg.norm(path_points[:, :2] - np.array([car.x, car.y]), axis=1))


        # Update last closest index for the next frame
        last_closest_idx = closest_idx

        # Extract a subset of path points to send to the controller
        # Wrap the start and end indices for a closed-loop path
        total_points = len(path_points)
        start_idx = (closest_idx - 1) % total_points
        end_idx = (closest_idx + num_points) % total_points

        # Handle the case where end_idx is less than start_idx due to wrapping
        if end_idx < start_idx:
            path_segment = np.vstack((path_points[start_idx:], path_points[:end_idx]))
        else:
            path_segment = path_points[start_idx:end_idx]

        # Update the path segment line plot
        path_segment_line.set_data(path_segment[:, 0], path_segment[:, 1])

        # Compute cross-track error (CTE)
        cte = compute_cte(car, path_points, closest_idx)
        cte_data.append(cte)

        # PID controller for velocity
        throttle = pid.compute(target_velocity, car.velocity, dt)
        steering_angle, predicted_positions = mpc_controller.get_lateral_control(car, dt, closest_idx) 
        print(f"Throttle: {throttle}, Steering Angle: {steering_angle}")
        
        steering_angle_data.append(steering_angle)
        

        # Update the car state
        car.update_state(throttle, steering_angle, dt)

        # Record history for visualization
        history.append((car.x, car.y))

        # Update car path and position on the plot
        car_path.set_data([p[0] for p in history], [p[1] for p in history])
        car_position.set_data([car.x], [car.y])
        predicted_line.set_data(predicted_positions[:, 0], predicted_positions[:, 1])

        # Extract predicted positions and yaw angles
        x_pred = predicted_positions[:, 0]
        y_pred = predicted_positions[:, 1]
        yaw_pred = predicted_positions[:, 2]

        # Compute the components for the quiver arrows
        u_pred = np.cos(yaw_pred)
        v_pred = np.sin(yaw_pred)

        # Update predicted orientation quiver
        predicted_orientation_quiver.set_offsets(np.c_[x_pred, y_pred])
        predicted_orientation_quiver.set_UVC(u_pred, v_pred) 

        # Update car orientation arrow
        orientation_quiver.set_offsets(np.array([[car.x, car.y]]))
        orientation_quiver.set_UVC(np.cos(car.yaw), np.sin(car.yaw))
        
        
        # Update time data
        time_data.append(frame * dt)
        current_velocity_data.append(car.velocity)
        target_velocity_data.append(target_velocity)

        # Update velocity plot
        velocity_line.set_data(time_data, current_velocity_data)
        target_velocity_line.set_data(time_data, target_velocity_data)

        # Update steering angle plot
        steering_angle_line.set_data(time_data, steering_angle_data)

        # Update cross-track error plot
        cte_line.set_data(time_data, cte_data)

        # Record end time
        end_time = time.time()
        
        # Calculate and print computation time
        computation_time = end_time - start_time
        print(f"Computation Time for Frame {frame}: {computation_time*1000:.1f} milliseconds")


        # Check if data lists have reached the maximum limit
        if len(time_data) > max_data_length:
            # Clear all data lists
            time_data.clear()
            current_velocity_data.clear()
            target_velocity_data.clear()
            steering_angle_data.clear()
            cte_data.clear()
            print(f"Data lists cleared at frame {frame} to prevent overflow.")

        return car_path, car_position, predicted_line, orientation_quiver, predicted_orientation_quiver, velocity_line, target_velocity_line, steering_angle_line, cte_line, path_segment_line

    ani = FuncAnimation(fig, update, frames=int(max_time / dt), interval=dt * 1000, blit=True)
    plt.tight_layout()
    plt.show()



# Main Function to run the simulation
def main():
    # Load raceline
    raceline = load_raceline('./racelines/raceline_KS.csv')

    # Initial car state
    kinematic_car = CarKinematicModel(x=-69.0, y=35.0, yaw=-2.2, velocity=0.0)
    
    # Controllers
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0)
    pure_pursuit = PurePursuitController(base_lookahead_distance=5.0, velocity_factor=0.1)
    # Create MPC controller instance
    mpc_controller = MPCController()
    n_pred = mpc_controller.N + 1

    # Run simulation
    animate_simulation(kinematic_car, raceline, pid, mpc_controller, dt=0.02, N_pred=n_pred)


# Run the main function
if __name__ == "__main__":
    main()
