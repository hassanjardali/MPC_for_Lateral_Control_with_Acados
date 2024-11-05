import numpy as np

class CarDynamicModel:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, yaw_rate=0.0, mass=1500, lf=1.5, lr=1.5, Iz=2250, Cf=19000, Cr=33000):
        """
        Initialize the dynamic bicycle model.
        
        Parameters:
        x, y       : Initial position of the car (m)
        yaw        : Initial yaw angle (radians)
        vx, vy     : Initial longitudinal and lateral velocities (m/s)
        yaw_rate   : Initial yaw rate (rad/s)
        mass       : Mass of the vehicle (kg)
        lf, lr     : Distance from center of gravity to front and rear axle (m)
        Iz         : Yaw moment of inertia (kg*m^2)
        Cf, Cr     : Cornering stiffness of front and rear tires (N/rad)
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx  # Longitudinal velocity
        self.vy = vy  # Lateral velocity
        self.velocity = np.sqrt(vx**2 + vy**2)  # Total velocity
        self.yaw_rate = yaw_rate  # Yaw rate
        self.mass = mass  # Vehicle mass
        self.lf = lf  # Distance from CG to front axle
        self.lr = lr  # Distance from CG to rear axle
        self.wheel_base = lf + lr  # Wheel base
        self.Iz = Iz  # Yaw moment of inertia
        self.Cf = Cf  # Cornering stiffness front
        self.Cr = Cr  # Cornering stiffness rear
        self.roadWheelAngleFactor = 1.0  # Road wheel angle

    def update_state(self, throttle, steering_angle, dt):
        """
        Update the state of the car using the dynamic bicycle model.
        
        Parameters:
        throttle      : Longitudinal force or acceleration (N or m/s^2)
        steering_angle: Steering angle of the front wheels (radians)
        dt            : Time step (seconds)
        """
        # Wrap steering angle to ensure it stays within bounds
        steering_angle = self.wrap_angle(steering_angle*self.roadWheelAngleFactor)
        
        # Slip angles for front and rear tires
        epsilon = 0
        vx_eff = self.vx if abs(self.vx) > epsilon else epsilon
        alpha_f = np.arctan2(self.vy + self.lf * self.yaw_rate, vx_eff) - steering_angle
        alpha_r = np.arctan2(self.vy - self.lr * self.yaw_rate, vx_eff)

        # Wrap slip angles
        alpha_f = self.wrap_angle(alpha_f)
        alpha_r = self.wrap_angle(alpha_r)

        # Lateral forces on front and rear tires
        Fyf = -self.Cf * alpha_f
        Fyr = -self.Cr * alpha_r

        # Equations of motion
        self.vx += (throttle - Fyf * np.sin(steering_angle) / self.mass + self.vy * self.yaw_rate) * dt
        self.vy += (Fyf * np.cos(steering_angle) + Fyr) / self.mass - self.vx * self.yaw_rate * dt
        self.velocity = np.sqrt(self.vx**2 + self.vy**2)

        self.yaw_rate += (self.lf * Fyf * np.cos(steering_angle) - self.lr * Fyr) / self.Iz * dt

        # Update position and yaw angle
        self.x += (self.vx * np.cos(self.yaw) - self.vy * np.sin(self.yaw)) * dt
        self.y += (self.vx * np.sin(self.yaw) + self.vy * np.cos(self.yaw)) * dt
        self.yaw += self.yaw_rate * dt

        # Wrap the yaw angle between -pi and pi
        self.yaw = self.wrap_angle(self.yaw)


    def get_state(self):
        """
        Return the current state of the car as a tuple.
        (x, y, yaw, vx, vy, velocity, yaw_rate) 
        """
        return self.x, self.y, self.yaw, self.vx, self.vy, self.velocity, self.yaw_rate
    
    def wrap_angle(self, angle):
        """
        Wrap the angle between -pi and pi.
        
        Parameters:
        angle: The input angle in radians
        
        Returns:
        Wrapped angle between -pi and pi
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def reset_state(self, x, y, yaw, vx, vy, yaw_rate):
        """
        Reset the car's state to match ground truth data.
        :param x: X position
        :param y: Y position
        :param yaw: Yaw angle
        :param velocity: Velocity in m/s
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.velocity = np.sqrt(self.vx**2 + self.vy**2)




# Car Kinematic Model
class CarKinematicModel:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, velocity=0.0, wheel_base=2.96):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity
        self.wheel_base = wheel_base
    
    def update_state(self, throttle, steering_angle, dt):
        """
        Update the state of the car using a simple kinematic bicycle model.
        :param throttle: Acceleration (m/s^2)
        :param steering_angle: Steering angle (radians)
        :param dt: Time step (s)
        """
        # Update velocity (simplified with no drag or resistance)
        self.velocity += throttle * dt

        # Update position and heading
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += (self.velocity / self.wheel_base) * np.tan(steering_angle) * dt
        
        # Wrap yaw angle between -pi and pi
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi


# Car Kinematic Model
class CarKinematicModel_v2:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, velocity=0.0, wheel_base=2.96):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity
        self.wheel_base = wheel_base
    
    def update_state(self, velocity, steering_angle, dt):
        """
        Update the state of the car using a simple kinematic bicycle model.
        :param velocity: Velocity (m/s)
        :param steering_angle: Steering angle (radians)
        :param dt: Time step (s)
        """
        # Update velocity (simplified with no drag or resistance)
        self.velocity =  velocity

        # Update position and heading
        self.x += self.velocity * np.cos(self.yaw) * dt
        self.y += self.velocity * np.sin(self.yaw) * dt
        self.yaw += (self.velocity / self.wheel_base) * np.tan(steering_angle) * dt
        
        # Wrap yaw angle between -pi and pi
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi
    
    def reset_state(self, x, y, yaw, velocity):
        """
        Reset the car's state to match ground truth data.
        :param x: X position
        :param y: Y position
        :param yaw: Yaw angle
        :param velocity: Velocity in m/s
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity