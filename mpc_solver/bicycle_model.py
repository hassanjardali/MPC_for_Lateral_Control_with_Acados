#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

from casadi import *


def bicycle_model(s0, kapparef):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    # load track parameters
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    model.kapparef_s = kapparef_s  # Store in model for later use


    ## Race car parameters
    m = 0.043
    C1 = 0.5
    C2 = 15.5
    Cm1 = 0.28
    Cm2 = 0.05
    Cr0 = 0.011
    Cr2 = 0.006

    ## CasADi Model
    # set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    alpha = MX.sym("alpha")
    v = MX.sym("v")
    D = MX.sym("D")
    delta = MX.sym("delta")
    x = vertcat(s, n, alpha, v, D, delta)

    # controls
    derD = MX.sym("derD")
    derDelta = MX.sym("derDelta")
    u = vertcat(derD, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat([])

    # dynamics
    Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(v)
    sdota = (v * cos(alpha + C1 * delta)) / (1 - kapparef_s(s) * n)
    f_expl = vertcat(
        sdota,
        v * sin(alpha + C1 * delta),
        v * C2 * delta - kapparef_s(s) * sdota,
        Fxd / m * cos(C1 * delta),
        derD,
        derDelta,
    )

    # constraint on forces
    a_lat = C2 * v * v * delta + Fxd * sin(C1 * delta) / m
    a_long = Fxd / m

    # Model bounds
    model.n_min = -0.12  # width of the track [m]
    model.n_max = 0.12  # width of the track [m]

    # state bounds
    model.throttle_min = -0.0
    model.throttle_max = 15.0

    model.delta_min = -3.0  # minimum steering angle [rad]
    model.delta_max = 3.00  # maximum steering angle [rad

    # input bounds
    model.ddelta_min = -2.0  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 2.0  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -1  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 1  # 10.0  # maximum throttle change rate

    # nonlinear constraint
    constraint.alat_min = -10  # maximum lateral force [m/s^2]
    constraint.alat_max = 10  # maximum lateral force [m/s^1]

    constraint.along_min = -18  # maximum lateral force [m/s^2]
    constraint.along_max = 18  # maximum lateral force [m/s^2]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, a_lat, n, D, delta)

    # Define model struct
    params = types.SimpleNamespace()
    params.C1 = C1
    params.C2 = C2
    params.Cm1 = Cm1
    params.Cm2 = Cm2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    return model, constraint


def kinematic_bicycle_model(s0, kapparef):
    # Define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "KinematicBicycleModel"

    # Load track parameters
    length = len(s0)
    pathlength = s0[-1]

    # Extend the track for smooth interpolation
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    # Compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)
    model.kapparef_s = kapparef_s  # Store in model for later use

    ## Vehicle parameters
    L = 2.9  # Wheelbase length [m]

    ## CasADi Model
    # Set up states & controls
    s = MX.sym("s")
    n = MX.sym("n")
    psi = MX.sym("psi")      # Heading angle relative to the track
    v = MX.sym("v")
    delta = MX.sym("delta")
    x = vertcat(s, n, psi, v, delta)

    # Controls
    a = MX.sym("a")          # Acceleration
    derDelta = MX.sym("derDelta")  # Steering rate
    u = vertcat(a, derDelta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    psidot = MX.sym("psidot")
    vdot = MX.sym("vdot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, ndot, psidot, vdot, deltadot)

    # Algebraic variables
    z = vertcat([])

    # Parameters
    p = vertcat([])

    # Dynamics
    beta = atan((L / 2) * tan(delta) / L)
    sdota = (v * cos(psi + beta)) / (1 - kapparef_s(s) * n)
    f_expl = vertcat(
        sdota,
        v * sin(psi + beta),
        (v / L) * sin(beta) - kapparef_s(s) * sdota,
        a,
        derDelta,
    )

    # Constraints
    a_lat = (v ** 2) * tan(delta) / L
    a_long = a

    # Model bounds
    model.n_min = -0.12  # Width of the track [m]
    model.n_max = 0.12   # Width of the track [m]

    # State bounds
    model.v_min = 0.0     # Minimum speed [m/s]
    model.v_max = 99.0     # Maximum speed [m/s]

    model.delta_min = -1.5  # Minimum steering angle [rad]
    model.delta_max = 1.5   # Maximum steering angle [rad]

    # Input bounds
    model.a_min = -1.0    # Minimum acceleration [m/s^2]
    model.a_max = 3.0     # Maximum acceleration [m/s^2]
    model.ddelta_min = -2.0  # Minimum steering rate [rad/s]
    model.ddelta_max = 2.0   # Maximum steering rate [rad/s]

    # Nonlinear constraints
    constraint.alat_min = -40.0  # Maximum lateral acceleration [m/s^2]
    constraint.alat_max = 40.0   # Maximum lateral acceleration [m/s^2]

    constraint.along_min = -3.0  # Maximum longitudinal deceleration [m/s^2]
    constraint.along_max = 2.0   # Maximum longitudinal acceleration [m/s^2]

    # Define initial conditions
    model.x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0])  # Starting at s=0, on centerline, heading along track, at 1 m/s

    # Define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.pathlength = pathlength
    constraint.expr = vertcat(a_long, a_lat, n, delta)

    # Define model struct
    params = types.SimpleNamespace()
    params.L = L
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params

    return model, constraint

def LateralMPCModel(s0, kapparef):

    # Define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "LateralMPCModel"

    # Load track parameters
    length = len(s0)
    pathlength = s0[-1]
    model.pathlength = pathlength  # Store path length for later use

    # Extend the track for smooth interpolation
    s0_extended = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef_extended = np.append(kapparef, kapparef[1:length])
    s0_extended = np.append(
        -s0[length - 2] + s0[length - 81 : length - 2], s0_extended
    )
    kapparef_extended = np.append(kapparef[length - 80 : length - 1], kapparef_extended)

    # Compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0_extended], kapparef_extended)
    model.kapparef_s = kapparef_s  # Store in model for later use

    ## Vehicle parameters
    L = 2.96  # Wheelbase length [m]
    m = 720  # Vehicle mass [kg]
    # C_af = 2100  # Front cornering stiffness [N/rad]
    # C_ar = 3200  # Rear cornering stiffness [N/rad]
    C_af = 120000  # Front cornering stiffness [N/rad]
    C_ar = 110000  # Rear cornering stiffness [N/rad]
    l_f = L / 2  # Distance from CG to front axle [m]
    l_r = L / 2  # Distance from CG to rear axle [m]
    I_z = 900  # Moment of inertia about z-axis [kg*m^2]

    ## CasADi Model
    # Set up states & controls
    n = MX.sym("n")        # Lateral deviation from the centerline
    ndot = MX.sym("ndot")  # Derivative of lateral deviation
    psi = MX.sym("psi")    # Heading angle relative to the track
    psidot = MX.sym("psidot")  # Derivative of heading angle error
    x = vertcat(n, ndot, psi, psidot)

    # Controls
    delta = MX.sym("delta")    # Steering angle
    u = vertcat(delta)

    # xdot
    n_dot = ndot
    ndot_dot = MX.sym("ndot_dot")
    psi_dot = psidot
    psidot_dot = MX.sym("psidot_dot")
    xdot = vertcat(n_dot, ndot_dot, psi_dot, psidot_dot)

    # Parameters
    v = MX.sym("v")    # Longitudinal speed (input parameter)
    kappa = MX.sym("kappa")  # Curvature at current position (input parameter)
    p = vertcat(v, kappa)  # Include parameters in the parameter vector
    # p = vertcat([])

    # Compute psi_dot_des = v * kappa
    psi_dot_des = v * kappa

    # Define matrices A, B_delta, C_psi_des
    A = MX.zeros(4, 4)
    A[0, 1] = 1
    A[1, 1] = -((2 * C_af + 2 * C_ar) / (m * v))
    A[1, 2] = (2 * C_af + 2 * C_ar) / m
    A[1, 3] = (-2 * C_af * l_f + 2 * C_ar * l_r) / (m * v)
    A[2, 3] = 1
    A[3, 1] = (2 * C_af * l_f - 2 * C_ar * l_r) / (I_z * v)
    A[3, 2] = (2 * C_af * l_f - 2 * C_ar * l_r) / I_z
    A[3, 3] = -((2 * C_af * l_f**2 + 2 * C_ar * l_r**2) / (I_z * v))

    B_delta = MX.zeros(4, 1)
    B_delta[1] = 2 * C_af / m
    B_delta[3] = 2 * C_af * l_f / I_z

    C_psi_des = MX.zeros(4, 1)
    C_psi_des[1] = -((2 * C_af * l_f - 2 * C_ar * l_r) / (m * v) + v)
    C_psi_des[3] = -((2 * C_af * l_f**2 + 2 * C_ar * l_r**2) / (I_z * v))

    # Compute xdot_expr = A * x + B_delta * delta + C_psi_des * psi_dot_des
    xdot_expr = mtimes(A, x) + mtimes(B_delta, delta) + C_psi_des * psi_dot_des

    # Set the expressions for ndot_dot and psidot_dot from xdot_expr
    ndot_dot_expr = xdot_expr[1]
    psidot_dot_expr = xdot_expr[3]

    # Assemble the explicit dynamics
    f_expl = vertcat(
        ndot,            # n_dot = ndot
        ndot_dot_expr,   # ndot_dot
        psidot,          # psi_dot = psidot
        psidot_dot_expr  # psidot_dot
    )

    # Model bounds
    model.n_min = -0.15  # Track width boundaries [m]
    model.n_max = 0.15

    model.delta_min = -0.14  # Steering angle limits [rad]
    model.delta_max = 0.14

    # Define initial conditions
    model.x0 = np.array([0.0, 0.0, 0.0, 0.0])  # Starting at centerline with no heading error

    # Define model struct
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = vertcat([])
    model.p = p
    model.name = model_name

    # Constraints (if any)
    constraint.expr = vertcat(n)
    constraint.n_min = model.n_min
    constraint.n_max = model.n_max
    constraint.delta_min = model.delta_min
    constraint.delta_max = model.delta_max

    return model, constraint  # Removed the extra 'with'