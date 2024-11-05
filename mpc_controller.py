import numpy as np
import pandas as pd
import time 
from car_models import CarKinematicModel, CarDynamicModel
from mpc_solver import acados_settings
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


class MPCController:
    def __init__(self):
        track_file = "./racelines/raceline_KS.csv"
        self.s0, self.kapparef, self.x_ref, self.y_ref, self.v_ref = self.getTrack(track_file)
        self.pathlength = self.s0[-1]
        
        self.Tf = 2.0 # prediction horizon
        self.N = 50   # number of discretization steps
        self.T = 10.0  # maximum simulation time[s]
        self.sref_N = 3  # reference for final reference progress

        # Load model and initialize acados solver

        # self.constraint, self.model, self.acados_solver = acados_settings.acados_settings(
        #     self.Tf, self.N, self.s0, self.kapparef
        # )

        # self.constraint, self.model, self.acados_solver = acados_settings.acados_settings_kinematic(
        #     self.Tf, self.N, self.s0, self.kapparef
        # )

        self.constraint, self.model, self.acados_solver = acados_settings.acados_lateral_v2(
            self.Tf, self.N, self.s0, self.kapparef
        )

    

        self.spline_x = CubicSpline(self.s0, self.x_ref)
        self.spline_y = CubicSpline(self.s0, self.y_ref)

        # Dimensions
        self.nx = self.model.x.size()[0]
        self.nu = self.model.u.size()[0]
        self.ny = self.nx + self.nu
        self.Nsim = int(self.T * self.N / self.Tf)

        # Initialize data structs
        self.simX = np.zeros((self.Nsim, self.nx))
        self.simU = np.zeros((self.Nsim, self.nu))
        self.s0_car = self.model.x0[0]
        self.tcomp_sum = 0
        self.tcomp_max = 0

        self.current_D = 0.0
        self.current_delta = 0.0

        self.D_ref = 0.0
        self.delta_ref = 0.0

        self.throttle_kp = 1.0
        self.steering_kp = -1.0

        self.dt = self.Tf / self.N  # Time step for each prediction

    
    def getTrack(self, track_file):
        # Read the CSV file
        data = pd.read_csv(track_file)
        # x = data['x'].to_numpy()
        # y = data['y'].to_numpy()
        x = data.iloc[:, 0].to_numpy()
        y = data.iloc[:, 1].to_numpy()
        
        # Compute arc lengths (s0)
        s0 = [0]
        for i in range(1, len(x)):
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            ds = np.hypot(dx, dy)
            s0.append(s0[-1] + ds)
        s0 = np.array(s0)
        
        # Compute curvatures (kapparef)
        kapparef = []
        for i in range(1, len(x) - 1):
            x1, y1 = x[i - 1], y[i - 1]
            x2, y2 = x[i], y[i]
            x3, y3 = x[i + 1], y[i + 1]
            # Compute curvature using the formula for three points
            kappa = self.compute_curvature(x1, y1, x2, y2, x3, y3)
            kapparef.append(kappa)
        # Pad kapparef to match the length of s0
        kapparef = [kapparef[0]] + kapparef + [kapparef[-1]]
        kapparef = np.array(kapparef)
        
        return s0, kapparef, x, y, data.iloc[:, 2].to_numpy()

    def compute_curvature(self, x1, y1, x2, y2, x3, y3):
        # Calculate the curvature given three points
        a = np.hypot(x1 - x2, y1 - y2)
        b = np.hypot(x2 - x3, y2 - y3)
        c = np.hypot(x3 - x1, y3 - y1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        if area == 0:
            return 0
        curvature = 4 * area / (a * b * c)
        return curvature
    
    def cartesian_to_frenet(self, car, closest_idx):
        x = car.x
        y = car.y
        theta = car.yaw  # Use 'yaw' instead of 'theta' if that's your variable name

        # Efficient search for closest s by minimizing the distance to the spline curve
        s_closest = self.find_closest_s(x, y, closest_idx)
        closest_point = np.array([self.spline_x(s_closest), self.spline_y(s_closest)])
        
        # Compute lateral deviation 'n'
        dx = x - closest_point[0]
        dy = y - closest_point[1]
        
        # Compute path tangent angle
        path_tangent_x = self.spline_x(s_closest, 1)  # First derivative of x wrt s
        path_tangent_y = self.spline_y(s_closest, 1)  # First derivative of y wrt s
        path_angle = np.arctan2(path_tangent_y, path_tangent_x)

        # Lateral deviation 'n' is the perpendicular distance from the point to the path
        n = dx * np.sin(path_angle) - dy * np.cos(path_angle)

        # Compute heading error 'alpha'
        alpha = theta - path_angle
        # Normalize alpha to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        s_closest = s_closest % self.pathlength  # Wrap around after finding closest s

        return s_closest, n, alpha, closest_idx
    
    def find_closest_s(self, x, y, closest_idx):
        """
        Finds the closest arc-length parameter `s` on the track to the car's current position.
        Uses a refined local search around the last known closest index.
        """
        # Take a small window around closest_idx to minimize search space
        search_range = 10  # Can adjust based on how dense your points are
        idx_start = max(0, closest_idx - search_range)
        idx_end = min(len(self.s0) - 1, closest_idx + search_range)  # Ensures idx_end is within bounds

        # Define a function to minimize: distance from (x, y) to the spline at s
        def distance_to_s(s):
            x_s = self.spline_x(s)
            y_s = self.spline_y(s)
            return np.hypot(x - x_s, y - y_s)

        # Perform minimization to find the closest s
        result = minimize_scalar(distance_to_s, bounds=(self.s0[idx_start], self.s0[idx_end]), method='bounded')
        return result.x  # Return the closest arc-length `s`

    
    def frenet_to_cartesian(self, s_list, n_list, theta_list, car):
        """
        Converts lists of Frenet coordinates (s_list, n_list) to Cartesian coordinates (x_list, y_list).
        Returns arrays of x and y positions corresponding to the Frenet coordinates.
        """
        x_list = []
        y_list = []
        yaw_list = []

    
        for s, n, theta in zip(s_list, n_list, theta_list):
            # Position on the reference path at s
            x_s = self.spline_x(s)
            y_s = self.spline_y(s)
            
            # Path tangent angle at s
            dx_ds = self.spline_x(s, 1)
            dy_ds = self.spline_y(s, 1)
            path_angle = np.arctan2(dy_ds, dx_ds)
            
            # Normal vector (perpendicular to path tangent)
            normal_angle = path_angle + np.pi / 2  # Rotate by 90 degrees to get normal
            # Shift from (x_s, y_s) along the normal vector by distance n
            x = x_s - n * np.cos(normal_angle)
            y = y_s - n * np.sin(normal_angle)
            yaw = path_angle 
            
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(theta+yaw)
        
        return np.array(x_list), np.array(y_list), np.array(yaw_list)


    def get_control(self, car, dt, closest_idx):
        # Convert to Frenet frame
        s, n, alpha, min_idx = self.cartesian_to_frenet(car, closest_idx)
        s = s % self.pathlength
        # print(f"Car position: x={car.x}, y={car.y}, theta={car.yaw}")
        # print(f"Frenet coordinates: s={s}, n={n}, alpha={alpha}, min_idx={min_idx}")

        # Current control states
        v_car = car.velocity
        D = self.current_D
        delta = self.current_delta
        # print(f"Current states: v_car={v_car}, D={D}, delta={delta}")

        # Assemble x0 for OCP
        x0 = np.array([s, n, alpha, v_car, D, delta])
        # print(f"x0: {x0}")

        # Set initial condition in acados solver
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # Define sref based on target velocity and prediction horizon
        target_velocity = self.v_ref[min_idx]  # Target velocity at the closest point
        sref_N = target_velocity * self.Tf # Target progress over the horizon
        sref = s + sref_N
        sref = sref % self.pathlength  # Wrap around the track length


        for j in range(self.N):
            progress = s + target_velocity * dt * j
            progress = progress % self.pathlength  # Wrap around the track length
            yref = np.array([progress, 0, 0, target_velocity, 0,0, 0, 0])
            # yref = np.array([progress, 0, 0, target_velocity, self.D_ref, self.delta_ref, 0, 0])
            self.acados_solver.set(j, "yref", yref)
        # print(f"yref[0]: {yref}")
        # Terminal reference
        yref_N = np.array([sref, 0, 0, target_velocity, 0, 0])
        self.acados_solver.set(self.N, "yref", yref_N)
        # print(f"yref_N: {yref_N}")

        # Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            print("acados returned status {}.".format(status))

        # Extract control inputs
        u0 = self.acados_solver.get(0, "u")
        derD = u0[0]
        derDelta = u0[1]

        # Update D and delta
        D_new = D + derD * self.dt
        delta_new = delta + derDelta * self.dt

        # Save for next iteration
        self.current_D = D_new
        self.current_delta = delta_new * self.steering_kp

        # Map D_new to throttle command
        throttle = D_new * self.throttle_kp  # Adjust mapping if necessary

        # Steering angle is delta_new
        steering_angle = delta_new * self.steering_kp

        # **Extract predicted states**
        predicted_states = []
        for i in range(self.N + 1):  # N+1 because acados stores N+1 states
            xi = self.acados_solver.get(i, "x")
            predicted_states.append(xi)

        # Convert list to numpy array for easier processing
        predicted_states = np.array(predicted_states)
        # print("Predicted states in frenet", predicted_states[0])

        # **Extract s and n from predicted states**
        s_pred = predicted_states[:, 0]
        n_pred = predicted_states[:, 1]
        alpha_pred = predicted_states[:, 2]

        # **Convert predicted Frenet positions to Cartesian coordinates**
        x_pred, y_pred, yaw_pred = self.frenet_to_cartesian(s_pred, n_pred, alpha_pred, car)

        # Return throttle, steering_angle, and predicted positions
        predicted_poses = np.vstack((x_pred, y_pred, yaw_pred)).T  # Shape (N+1, 2)
        # print("Predicted positions in Cartesian", predicted_positions)
        return throttle, steering_angle, predicted_poses
    

    def get_control_kinematic(self, car, dt, closest_idx):
        # Convert to Frenet frame
        s, n, alpha, min_idx = self.cartesian_to_frenet(car, closest_idx)
        s = s % self.pathlength
        # print(f"Car position: x={car.x}, y={car.y}, theta={car.yaw}")
        # print(f"Frenet coordinates: s={s}, n={n}, alpha={alpha}, min_idx={min_idx}")

        # Current control states
        v_car = car.velocity
        D = self.current_D
        delta = self.current_delta
        # print(f"Current states: v_car={v_car}, D={D}, delta={delta}")

        # Assemble x0 for OCP
        x0 = np.array([s, n, alpha, v_car, delta])
        # print(f"x0: {x0}")

        # Set initial condition in acados solver
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # Define sref based on target velocity and prediction horizon
        target_velocity = self.v_ref[min_idx]  # Target velocity at the closest point
        sref_N = target_velocity * self.Tf # Target progress over the horizon
        sref = s + sref_N
        sref = sref % self.pathlength  # Wrap around the track length


        for j in range(self.N):
            progress = s + target_velocity * dt * j
            progress = progress % self.pathlength  # Wrap around the track length
            yref = np.array([progress, 0, 0, target_velocity, 0, 0, 0])
            # yref = np.array([progress, 0, 0, target_velocity, self.delta_ref, 0, 0])
            self.acados_solver.set(j, "yref", yref)
        # print(f"yref[0]: {yref}")
        # Terminal reference
        yref_N = np.array([sref, 0, 0, target_velocity, 0])
        self.acados_solver.set(self.N, "yref", yref_N)
        # print(f"yref_N: {yref_N}")

        # Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            print("acados returned status {}.".format(status))

        # Extract control inputs
        u0 = self.acados_solver.get(0, "u")
        D = u0[0]
        derDelta = u0[1]

        # Update D and delta
        D_new = D 
        delta_new = delta + derDelta * self.dt

        # Save for next iteration
        self.current_D = D_new
        self.current_delta = delta_new * self.steering_kp

        # Map D_new to throttle command
        throttle = D_new * self.throttle_kp  # Adjust mapping if necessary

        # Steering angle is delta_new
        steering_angle = delta_new * self.steering_kp

        # **Extract predicted states**
        predicted_states = []
        for i in range(self.N + 1):  # N+1 because acados stores N+1 states
            xi = self.acados_solver.get(i, "x")
            predicted_states.append(xi)

        # Convert list to numpy array for easier processing
        predicted_states = np.array(predicted_states)
        # print("Predicted states in frenet", predicted_states[0])

        # **Extract s and n from predicted states**
        s_pred = predicted_states[:, 0]
        n_pred = predicted_states[:, 1]
        alpha_pred = predicted_states[:, 2]

        # **Convert predicted Frenet positions to Cartesian coordinates**
        x_pred, y_pred, yaw_pred = self.frenet_to_cartesian(s_pred, n_pred, alpha_pred, car)

        # Return throttle, steering_angle, and predicted positions
        predicted_poses = np.vstack((x_pred, y_pred, yaw_pred)).T  # Shape (N+1, 2)
        # print("Predicted positions in Cartesian", predicted_positions)
        return throttle, steering_angle, predicted_poses
    

    def get_control_kinematic_v2(self, car, dt, closest_idx):
        # 1. Convert car position to Frenet frame
        s, n, alpha, min_idx = self.cartesian_to_frenet(car, closest_idx)
        s = s % self.pathlength  # Wrap-around if exceeds path length

        # 2. Retrieve current vehicle states
        v_car = car.velocity
        D = self.current_D
        delta = self.current_delta

        # 3. Assemble initial state vector for OCP
        x0 = np.array([s, n, alpha, v_car, delta])

        # 4. Set initial conditions in the Acados solver
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # 5. Define reference progress `sref` over the prediction horizon
        target_velocity = self.v_ref[min_idx]  # Target velocity at current closest point
        sref_N = target_velocity * self.Tf*2     # Reference progress over horizon
        sref = (s + sref_N) % self.pathlength  # Wrap-around if necessary

        # 6. Set reference for each prediction step
        for j in range(self.N):
            progress = (s + target_velocity * dt * j) % self.pathlength
            yref = np.array([progress, 0, 0, target_velocity, 0, 0, 0])
            self.acados_solver.set(j, "yref", yref)

        # 7. Set terminal reference (last prediction step)
        yref_N = np.array([sref, 0, 0, target_velocity, 0])
        self.acados_solver.set(self.N, "yref", yref_N)

        # 8. Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            print("acados returned status {}.".format(status))

        # 9. Extract control inputs
        u0 = self.acados_solver.get(0, "u")
        D = u0[0]           # Desired acceleration
        derDelta = u0[1]    # Desired rate of change of steering

        # 10. Update states for next control step
        D_new = D
        delta_new = delta + derDelta * dt
        self.current_D = D_new
        self.current_delta = delta_new * self.steering_kp

        # 11. Map control inputs to throttle and steering commands
        throttle = D_new * self.throttle_kp
        steering_angle = delta_new * self.steering_kp

        # 12. Extract predicted states from the solver
        predicted_states = []
        for i in range(self.N + 1):  # Retrieve N+1 states for full horizon
            xi = self.acados_solver.get(i, "x")
            predicted_states.append(xi)

        # Convert predicted states list to numpy array
        predicted_states = np.array(predicted_states)

        # 13. Extract Frenet frame predictions (s, n, alpha)
        s_pred = predicted_states[:, 0]
        n_pred = predicted_states[:, 1]
        alpha_pred = predicted_states[:, 2]

        # 14. Convert predicted Frenet coordinates to Cartesian positions
        x_pred, y_pred, yaw_pred = self.frenet_to_cartesian(s_pred, n_pred, alpha_pred, car)

        # 15. Return throttle, steering angle, and predicted Cartesian positions
        predicted_poses = np.vstack((x_pred, y_pred, yaw_pred)).T  # Shape (N+1, 3)
        return throttle, steering_angle, predicted_poses
    
    def get_lateral_control(self, car, dt, closest_idx):
        # 1. Convert car position to Frenet frame
        s, n, psi, min_idx = self.cartesian_to_frenet(car, closest_idx)  # Ensure 's' is returned
        psi = - psi
        
        # 2. Retrieve current vehicle state
        delta = self.current_delta
        v_car = car.velocity + 1e-2 # Actual current speed of the car

        # 3. Assemble initial state vector for OCP
        ndot = 0.0     # Assuming initial derivative of lateral deviation is zero
        psidot = 0.0   # Assuming initial derivative of heading error is zero
        x0 = np.array([n, ndot, psi, psidot])

        # 4. Set initial conditions in the Acados solver
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        # 5. Predict future states and set parameters
        N = self.N
        dt_horizon = dt  # Time step size
        s_pred = np.zeros(N+1)
        v_pred = np.zeros(N+1)
        kappa_pred = np.zeros(N+1)
        s_pred[0] = s
        v_pred[0] = v_car

        for j in range(N):
            # Predict future s positions
            s_pred[j+1] = s_pred[j] + v_pred[j] * dt_horizon
            s_pred[j+1] = np.mod(s_pred[j+1], self.pathlength)  # Wrap around track length

            # Get curvature at predicted s
            kappa_pred[j] = self.model.kapparef_s(s_pred[j]).full().flatten()[0]

            # Assume constant speed over horizon (or update v_pred[j+1] if speed changes)
            v_pred[j+1] = v_pred[j]

            # Set parameters v and kappa for each stage
            # p = np.array([v_pred[j], kappa_pred[j]])
            # self.acados_solver.set(j, "p", p)

            # Set reference to be zero deviation and zero heading error
            yref = np.array([0, 0, 0, 0, 0])  # [n, ndot, psi, psidot, delta]
            self.acados_solver.set(j, "yref", yref)

        # Set terminal stage parameters
        kappa_pred[N] = self.model.kapparef_s(s_pred[N]).full().flatten()[0]
        # p = np.array([v_pred[N], kappa_pred[N]])
        # self.acados_solver.set(N, "p", p)

        # Set terminal reference
        yref_N = np.array([0, 0, 0, 0])  # Terminal reference for states
        self.acados_solver.set(N, "yref", yref_N)

        # 6. Solve OCP
        status = self.acados_solver.solve()
        if status != 0:
            print(f"acados returned status {status}.")

        # 7. Extract control input
        delta_opt = self.acados_solver.get(0, "u")[0]  # Steering angle

        # 8. Update current steering angle
        self.current_delta = delta_opt

        # 9. Map control input to steering command
        steering_angle = delta_opt * self.steering_kp

        # 10. Extract predicted states from the solver
        predicted_states = []
        for i in range(N + 1):  # Retrieve N+1 states for full horizon
            xi = self.acados_solver.get(i, "x")
            predicted_states.append(np.array(xi))

        # Convert predicted states list to numpy array
        predicted_states = np.array(predicted_states)

        # 11. Extract Frenet frame predictions (n, psi)
        n_pred = predicted_states[:, 0]
        psi_pred = predicted_states[:, 2]

        # 12. Convert predicted Frenet coordinates to Cartesian positions
        x_pred, y_pred, yaw_pred = self.frenet_to_cartesian(s_pred, n_pred, psi_pred, car)

        # 13. Return steering angle and predicted Cartesian positions
        predicted_poses = np.vstack((x_pred, y_pred, yaw_pred)).T  # Shape (N+1, 3)
        # print("Curvature kappa_pred:", kappa_pred)

        return steering_angle, predicted_poses
