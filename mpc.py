import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import MX, vertcat, sin, atan2
from types import SimpleNamespace


mpc_params_dict = {
    "Tf": 1.6,  # Prediction horizon [s]
    "N": 45,  # Number of control intervals
    "s0": 0.0,  # Initial lateral deviation [m]
    "kapparef": 0.0,  # Initial curvature [1/m]
    "vref": 10.0,  # Initial speed [m/s]
    "Q" : [2.5, 1.5, 2.0, 4.0, 2.8],  # State weights - [n, ndot, psi, psidot, delta] 
    # "Q": [4.0, 0.0, 0.0, 0.0, 0.0],  # State weights - [n, ndot, psi, psidot, delta]
    "Q_beta": [1.0],
    "R" : [7.5],  # Weight for control input [delta]
    "Qe" : [2.5, 1.5, 2.0, 4.0, 2.8],  # Terminal cost weights - For now, same as Q 
    "unscale" : 20.0,  # Scaling factor for the cost function - For now set to N/Tf
    "cost_type" : "NONLINEAR_LS",  # Cost function type
    "cost_type_e" : "NONLINEAR_LS",  # Terminal cost function type
    "z_low" : 1e-3,  # Lower bound for slack variables
    "z_up" : 1e-3,  # Upper bound for slack variables
    "Z_low" : 0.1,  # Lower bound for slack variables
    "Z_up" : 0.1,  # Upper bound for slack variables
    "qp_solver" : "FULL_CONDENSING_HPIPM",  # QP solver
    "nlp_solver_type" : "SQP_RTI",  # NLP solver type
    "hessian_approx" : "GAUSS_NEWTON",  # Hessian approximation
    "integrator_type" : "ERK",  # Integrator type
    "sim_method_num_stages" : 3,  # Number of stages in the integrator
    "sim_method_num_steps" : 4,  # Number of steps in the integrator
    "nlp_solver_max_iter" : 200,  # Maximum number of iterations for the NLP solver
    "tol" : 1e-6,  # Tolerance for the NLP solver
}

car_params_dict = {
    "L": 2.9718,        # Wheelbase length [m]
    "m": 772.11,        # Vehicle mass [kg]
    "C_ar": 190000.00,    # Front cornering stiffness [N/rad] # Example value
    "C_af": 120000.00,   # Rear cornering stiffness [N/rad] # Example value
    "l_f": 1.6785,      # Distance from CG to front axle [m]
    "l_r": 1.2933,      # Distance from CG to rear axle [m]
    "I_z": 1200,         # Moment of inertia about z-axis [kg*m^2]
}

constraints_dict = {
    "n_min": -0.02, # Track width boundaries [m] - Not used in this model (NU)
    "n_max": 0.02, # Track width boundaries [m] - NU
    "delta_min": -0.15, # Steering angle limits [rad] 
    "delta_max": 0.15, # Steering angle limits [rad] 
    "psi_min": -0.03, # Heading angle limits [rad] - NU
    "psi_max": 0.03, # Heading angle limits [rad] - NU
    "ndot_min": -10.0, # Lateral acceleration limits [m/s^2] - NU
    "ndot_max": 10.0, # Lateral acceleration limits [m/s^2] - NU
    "psidot_min": -0.15, # Yaw rate limits [rad/s]  - NU
    "psidot_max": 0.15, # Yaw rate limits [rad/s] - NU
    "delta_dot_min": -0.05,  # Steering rate limits 
    "delta_dot_max": 0.05,
}


x_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Starting at centerline with no heading error
u_0 = np.array([0.0])

def LateralDynamicLinearTire(car_params_dict, constraints_dict, x_0):
    model_name = "LateralDynamicLinearTire"

    # Vehicle geometry and inertia parameters as fixed values
    L = car_params_dict["L"]
    l_f = car_params_dict["l_f"]
    l_r = car_params_dict["l_r"]
    m = car_params_dict["m"]
    I_z = car_params_dict["I_z"]

    # Now define them as parameters:
    C_af_param = MX.sym("C_af_param")
    C_ar_param = MX.sym("C_ar_param")

    # States
    n = MX.sym("n")        # Lateral deviation
    ndot = MX.sym("ndot")  # Derivative of lateral deviation
    psi = MX.sym("psi")    # Heading angle error
    psidot = MX.sym("psidot") # Heading Error rate and not Yaw rate!
    delta = MX.sym("delta") # Steering angle
    x = vertcat(n, ndot, psi, psidot, delta)

    # Controls
    delta_dot = MX.sym("delta_dot")
    u = vertcat(delta_dot)

    # Parameters
    v = MX.sym("v")       # Longitudinal speed
    kappa = MX.sym("kappa") # Curvature
    phi = MX.sym("phi") # banking angle
    p = vertcat(v, kappa, phi, C_af_param, C_ar_param)

    # System matrices depend on parameters:
    A = MX.zeros(4, 4)
    A[0, 1] = 1
    A[1, 1] = -((2 * C_af_param + 2 * C_ar_param) / (m * v))
    A[1, 2] = (2 * C_af_param + 2 * C_ar_param) / m
    A[1, 3] = (-2 * C_af_param * l_f + 2 * C_ar_param * l_r) / (m * v)
    A[2, 3] = 1
    A[3, 1] = -(2 * C_af_param * l_f - 2 * C_ar_param * l_r) / (I_z * v)
    A[3, 2] = (2 * C_af_param * l_f - 2 * C_ar_param * l_r) / I_z
    A[3, 3] = -((2 * C_af_param * l_f**2 + 2 * C_ar_param * l_r**2) / (I_z * v))

    B_delta = MX.zeros(4, 1)
    B_delta[1] = 2 * C_af_param / m
    B_delta[3] = 2 * C_af_param * l_f / I_z

    C_psi_des = MX.zeros(4, 1)
    C_psi_des[1] = (-(2 * C_af_param * l_f - 2 * C_ar_param * l_r) / (m * v) - v)
    C_psi_des[3] = -((2 * C_af_param * l_f**2 + 2 * C_ar_param * l_r**2) / (I_z * v))

    A_aug = MX.zeros(5,5)
    A_aug[0:4,0:4] = A
    A_aug[0:4,4] = B_delta

    B_aug = MX.zeros(5,1)
    B_aug[4] = 1

    C_aug = MX.zeros(5,1)
    C_aug[0:4] = C_psi_des

    psi_dot_des = v*kappa
    xdot_expr = A_aug@x + B_aug*delta_dot + C_aug*psi_dot_des

    n_dot = ndot
    ndot_dot_expr = xdot_expr[1]
    psidot_dot_expr = xdot_expr[3]

    # Gravity
    g = 9.81

    ndot_dot_expr = xdot_expr[1] + g * sin(phi)  # Lateral acceleration due to banking

    f_expl = vertcat(
        n_dot,
        ndot_dot_expr,
        psidot,
        psidot_dot_expr,
        delta_dot
    )

    xdot = MX.sym("xdot",5,1)

    model = SimpleNamespace()
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = vertcat([])
    model.p = p
    model.name = model_name

    # Constraints
    constraint = SimpleNamespace()
    constraint.expr = vertcat(delta)
    constraint.n_min = constraints_dict["n_min"]
    constraint.n_max = constraints_dict["n_max"]
    constraint.delta_min = constraints_dict["delta_min"]
    constraint.delta_max = constraints_dict["delta_max"]

    # Model bounds and initial condition
    model.n_min = constraints_dict["n_min"]
    model.n_max = constraints_dict["n_max"]
    model.delta_min = constraints_dict["delta_min"]
    model.delta_max = constraints_dict["delta_max"]
    model.delta_dot_min = constraints_dict["delta_dot_min"]
    model.delta_dot_max = constraints_dict["delta_dot_max"]
    model.x0 = x_0

    return model, constraint


# def acados_settings(mpc_params_dict, car_params_dict, constraints_dict, x_0):
def acados_settings():

    # Create render arguments
    ocp = AcadosOcp()

    # Export model
    model, constraint = LateralDynamicLinearTire(car_params_dict, constraints_dict, x_0)

    # Define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # Define constraint
    ocp.model.con_h_expr = constraint.expr  # Corrected attribute

    # Dimensions
    nx = model.x.size1()
    nu = model.u.size1()
    ny = nx + nu
    ny_e = nx

    nh = constraint.expr.size1()  # Update nh based on new constraints
    nsh = nh  # Number of soft constraints (if using soft constraints)
    ns = nsh

    # Discretization
    ocp.solver_options.N_horizon = mpc_params_dict["N"] # Number of control intervals

    # Set cost

    # Introduce Q and R as additional parameters:
    Q_param = MX.sym("Q_param", nx, 1)
    R_param = MX.sym("R_param", nu, 1)
    Q_beta_param = MX.sym("Q_beta_param", 1, 1)

    # Accessing parameters individually instead of unpacking
    v = ocp.model.p[0]
    kappa = ocp.model.p[1]
    phi = ocp.model.p[2]
    C_af_param = ocp.model.p[3]
    C_ar_param = ocp.model.p[4]

    # Include additional parameters for Q and R if necessary
    p_full = vertcat(v, kappa, phi, C_af_param, C_ar_param, Q_param, R_param, Q_beta_param)
    ocp.model.p = p_full

    ocp.cost.cost_type = mpc_params_dict["cost_type"]
    ocp.cost.cost_type_e = mpc_params_dict["cost_type_e"]

    W = np.diag(mpc_params_dict["Q"] + mpc_params_dict["R"] + mpc_params_dict["Q_beta"])  # Size (ny, ny)
    W_e = np.diag(mpc_params_dict["Qe"])  # Terminal cost weights (ny_e, ny_e)
    ocp.cost.W = W  # Weight matrix for intermediate stage
    # ocp.cost.W_e = W_e  # Weight matrix for terminal stage

    unscale = mpc_params_dict["N"] / mpc_params_dict["Tf"]


    x_ref = MX.sym("x_ref", nx, 1)
    u_ref = MX.sym("u_ref", nu, 1)

    ocp.model.p = vertcat(ocp.model.p, x_ref, u_ref)

    # NEW COST FUNCTION

    residual_list = []
    for i in range(nx):
        residual_list.append( (model.x[i] - x_ref[i]) * MX.sqrt(Q_param[i]) )
    for j in range(nu):
        residual_list.append( (model.u[j] - u_ref[j]) * MX.sqrt(R_param[j]) )

    # 1) Define side slip angle
    beta = atan2(model.x[1], v)  # ndot / v
    residual_list.append(beta**2 * Q_beta_param[0])

    cost_y_expr = vertcat(*residual_list)

    # Terminal cost residual:
    residual_list_e = []
    for i in range(nx):
        residual_list_e.append( (model.x[i] - x_ref[i]) * MX.sqrt(Q_param[i]) )
    cost_y_expr_e = vertcat(*residual_list_e)

    ocp.model.cost_y_expr = cost_y_expr
    # ocp.model.cost_y_expr_e = cost_y_expr_e

    # end of new cost function

    # Dimensions for yref
    ocp.cost.yref = np.zeros(ny+1)
    # ocp.cost.yref_e = np.zeros(ny_e)

    ocp.cost.zl = mpc_params_dict["z_low"] * np.ones((ns,))
    ocp.cost.zu = mpc_params_dict["z_up"] * np.ones((ns,))
    ocp.cost.Zl = mpc_params_dict["Z_low"] * np.ones((ns,))
    ocp.cost.Zu = mpc_params_dict["Z_up"] * np.ones((ns,))


    # Setting constraints
    # From the statesm only steering angle is constrained

    ocp.constraints.lbu = np.array([model.delta_dot_min])
    ocp.constraints.ubu = np.array([model.delta_dot_max])

    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lh = np.array([
        # constraint.n_min,
        # constraint.psi_min,
        # constraint.ndot_min,
        # constraint.psidot_min,
        constraint.delta_min
    ])
    ocp.constraints.uh = np.array([
        # constraint.n_max,
        # constraint.psi_max,
        # constraint.ndot_max,
        # constraint.psidot_max,
        constraint.delta_max
    ])


    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))

    # Set initial condition
    ocp.constraints.x0 = model.x0

    # Set up solver
    # ocp.dims.np = model.p.size1()
    
    # Set QP solver and integration
    ocp.solver_options.tf = mpc_params_dict["Tf"]
    ocp.solver_options.qp_solver =  mpc_params_dict["qp_solver"]
    ocp.solver_options.nlp_solver_type =  mpc_params_dict["nlp_solver_type"]
    ocp.solver_options.hessian_approx =  mpc_params_dict["hessian_approx"]
    ocp.solver_options.integrator_type =  mpc_params_dict["integrator_type"]
    ocp.solver_options.sim_method_num_stages =  mpc_params_dict["sim_method_num_stages"]
    ocp.solver_options.sim_method_num_steps =  mpc_params_dict["sim_method_num_steps"]
    ocp.solver_options.nlp_solver_max_iter =  mpc_params_dict["nlp_solver_max_iter"]
    ocp.solver_options.tol =  mpc_params_dict["tol"]
    # ocp.solver_options.qp_tol =  mpc_params_dict["tol"] / 10


    # Set initial parameter values (placeholders; will be updated during simulation)
    initial_v =  mpc_params_dict["vref"]  # Initial vehicle speed [m/s]
    initial_kappa = mpc_params_dict["kapparef"]  # Initial curvature [1/m]
    C_af_init = car_params_dict["C_af"]
    C_ar_init = car_params_dict["C_ar"]

    Q = mpc_params_dict["Q"]
    R = mpc_params_dict["R"]
    Q_beta = mpc_params_dict["Q_beta"]#


    x_ref_0 = x_0
    u_ref_0 = u_0

    # Combine all parameters into a single 1D array
    ocp.parameter_values = np.hstack([
        initial_v,
        initial_kappa,
        0, # phi
        C_af_init,
        C_ar_init,
        Q,
        R,
        Q_beta,
        x_ref_0,
        u_ref_0
    ])    

    # Create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return constraint, model, acados_solver


# constraint, model, acados_solver = acados_settings(mpc_params_dict, car_params_dict, constraints_dict, x_0)