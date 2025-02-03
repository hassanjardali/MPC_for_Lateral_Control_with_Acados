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
- **`mpc.py`**: It has the main MPC model and acados settings.

This structure provides a modular setup for simulation and controller management, with options to add more models and controllers as needed.


![Short video description](./media/python_sim_lat_mpc.gif)

--- 


## Install Casadi
Ensure you have the specified version of Casadi:
```bash
pip install casadi==3.6.7
```

## Install acados
**Important:** Clone `acados` in your home directory because the `CMakeLists.txt` of the controller uses an absolute path `~/acados`.

1. Clone the repository and update submodules:
   ```bash
   git clone https://github.com/acados/acados.git
   cd acados
   git submodule update --recursive --init
   ```

2. Build `acados`:
   ```bash
   mkdir -p build
   cd build
   cmake -DACADOS_WITH_QPOASES=ON ..
   # Add more optional arguments as needed, e.g.:
   # -DACADOS_WITH_OSQP=OFF/ON
   # -DACADOS_INSTALL_DIR=<path_to_acados_installation_folder>
   make install -j4
   ```

## Update Your `.bashrc`
Add the following lines to your `.bashrc` file:
```bash
export ACADOS_SOURCE_DIR=~/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
export PYTHONPATH=$PYTHONPATH:$ACADOS_SOURCE_DIR/interfaces/acados_template
```
After adding these, reload your `.bashrc`:
```bash
source ~/.bashrc
```

## Additional Dependency
Install `future-fstrings`:
```bash
pip install future-fstrings
```


