# Project Dependencies and Instructions

## Dependencies

- **Casadi**: Version 3.6.7
- **acados**: [Installation guide](https://docs.acados.org/installation/)

## How to Run

The main Python script to execute the simulation is `2d_simulator.py`. Below is a brief description of the project structure and functionality.

### Project Structure

- **`2d_simulator.py`**: The primary script for running the simulation.
- **`car_models.py`**: Contains models used in the simulation. Currently, only the kinematic model is utilized by the simulator.
- **`controllers.py`**: Contains the following controllers:
  - **Pure Pursuit Controller**
  - **PID Controller**
- **`mpc_controller.py`**: Contains the class and functions for the MPC (Model Predictive Control) controller.
- **`mpc_solver/`**: Directory containing scripts used to generate the C code for MPC, the model used, and optimization options.

This structure provides a modular setup for simulation and controller management, with options to add more models and controllers as needed.


![Short video description](./media/python_sim_lat_mpc.gif)

--- 


The current model used is the model used now is the same used by KAIST and MIT, which is from "Rajamani, R. Vehicle Dynamics and Control; Springer Science & Business Media, 2011" which also I found some derivation of it here: https://www.researchgate.net/publication/341127675_Practical_Approach_for_Developing_Lateral_Motion_Control_of_Autonomous_Lane_Change_System (pages 4-5)


