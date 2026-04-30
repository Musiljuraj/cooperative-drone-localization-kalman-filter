"""
Cooperative Localization of Three Drones Using a Kalman Filter

This file contains the complete numerical part of the project:
1. model creation,
2. trajectory simulation,
3. measurement simulation,
4. Kalman filtering,
5. error evaluation,
6. plotting.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Global configuration
# ============================================================

# The random seed makes the simulation reproducible. With the same seed,
# the generated process noise, GPS noise and relative measurement noise are
# the same every time the program is run. This is useful for the project,
# because the figures and RMSE values remain stable while writing the report.
#
# Set RANDOM_SEED to an integer for reproducible results.
# Set RANDOM_SEED to None for a different random simulation each time.
# RANDOM_SEED = 1
RANDOM_SEED = None

# Simulation parameters
DT = 1.0                 # sampling period [s]
N_STEPS = 40            # number of simulation steps

# Noise parameters
#
# SIGMA_A describes the random acceleration used in the process noise model.
# It represents deviations from the ideal constant-velocity motion model.
#
# SIGMA_GPS describes the standard deviation of GPS position error.
# In this project it is intentionally chosen relatively large, because the
# goal is to study whether additional relative measurements can improve
# localization when GPS is noisy.
#
# SIGMA_REL describes the standard deviation of the relative position
# measurement error. It is chosen smaller than SIGMA_GPS, because the relative
# measurement is assumed to provide more accurate information about the
# geometry of the three-drone formation.
SIGMA_A = 0.15            # process acceleration noise [m/s^2]
SIGMA_GPS = 8.0          # GPS position measurement noise [m]
SIGMA_REL = 0.5          # relative position measurement noise [m]

# Output paths
#
# PROJECT_ROOT is the root folder of the project. Since this file is expected
# to be stored in the code/ directory, parents[1] points one level above code/.
# FIGURES_DIR is the folder where the plots used in the article are saved.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"


# ============================================================
# Model creation
# ============================================================

def create_model(dt, sigma_a, sigma_gps, sigma_rel):
    """
    Create all matrices required by the two Kalman filters.

    Returns
    -------
    A : ndarray, shape (12, 12)
        State transition matrix.
    Q : ndarray, shape (12, 12)
        Process noise covariance matrix.
    H_gps : ndarray, shape (6, 12)
        GPS-only measurement matrix.
    R_gps : ndarray, shape (6, 6)
        GPS-only measurement noise covariance matrix.
    H_comb : ndarray, shape (12, 12)
        Combined GPS + relative position measurement matrix.
    R_comb : ndarray, shape (12, 12)
        Combined measurement noise covariance matrix.
    """

    # This function implements the mathematical model from Chapter 3 of the
    # article, but extended from two drones to three drones. It creates the
    # matrices used in the linear Gaussian state-space model
    #
    #     X_k = A X_{k-1} + W_k,
    #     Y_k = H X_k + V_k.
    #
    # The same transition matrix A and process noise covariance Q are used
    # for both filters. The GPS-only filter and the combined filter differ only
    # in their measurement matrix H and measurement noise covariance R.

    # --------------------------------------------------------
    # State transition matrix for one drone
    # State ordering for one drone:
    # [x, y, vx, vy]
    # --------------------------------------------------------
    # The one-drone transition matrix F represents the constant-velocity model:
    #
    #     x_k  = x_{k-1} + dt * vx_{k-1}
    #     y_k  = y_{k-1} + dt * vy_{k-1}
    #     vx_k = vx_{k-1}
    #     vy_k = vy_{k-1}
    #
    # This is the deterministic part of the prediction step. The model assumes
    # that velocity is approximately constant during one time step. Any
    # deviation from this assumption is represented later by the process noise.
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # --------------------------------------------------------
    # Full state transition matrix for three drones
    # Full state ordering:
    # [x1, y1, vx1, vy1,
    #  x2, y2, vx2, vy2,
    #  x3, y3, vx3, vy3]
    # --------------------------------------------------------
    # The full hidden state contains three independent copies of the one-drone
    # state. Therefore, the full transition matrix A is block diagonal.
    # The zero blocks mean that the motion model itself does not directly
    # couple the drones. The coupling appears later only through the relative
    # position measurements in the combined measurement model.
    zero_4x4 = np.zeros((4, 4))

    A = np.block([
        [F,        zero_4x4, zero_4x4],
        [zero_4x4, F,        zero_4x4],
        [zero_4x4, zero_4x4, F       ],
    ])

    # --------------------------------------------------------
    # Process noise covariance for one drone
    # Based on unknown acceleration noise during one time step.
    # --------------------------------------------------------
    # Q_drone describes uncertainty in the motion model of one drone.
    # In Chapter 3, it is derived from the assumption that an unknown random
    # acceleration can act during one time step.
    #
    # For one coordinate:
    #
    #     Delta x  = 1/2 * dt^2 * a
    #     Delta vx = dt * a
    #
    # If a has variance sigma_a^2, then position and velocity errors have
    # covariance terms dt^4/4, dt^3/2 and dt^2. The same construction is used
    # independently for the x- and y-directions.
    Q_drone = sigma_a**2 * np.array([
        [dt**4 / 4.0, 0.0,          dt**3 / 2.0, 0.0         ],
        [0.0,         dt**4 / 4.0,  0.0,          dt**3 / 2.0],
        [dt**3 / 2.0, 0.0,          dt**2,        0.0         ],
        [0.0,         dt**3 / 2.0,  0.0,          dt**2       ],
    ])

    # Full process noise covariance for three drones
    # This again uses a block diagonal structure. The three drones are assumed
    # to have independent process noises, so the off-diagonal blocks are zero.
    Q = np.block([
        [Q_drone,  zero_4x4, zero_4x4],
        [zero_4x4, Q_drone,  zero_4x4],
        [zero_4x4, zero_4x4, Q_drone ],
    ])

    # --------------------------------------------------------
    # GPS-only measurement matrix
    # GPS measures:
    # [x1, y1, x2, y2, x3, y3]
    # --------------------------------------------------------
    # H_gps maps the hidden state X_k to the measurement vector Y_k^{GPS}.
    # It selects only the absolute position coordinates of all three drones.
    # Velocities are not measured directly by GPS, so the velocity columns are
    # zero. The Kalman filter estimates velocity indirectly from the sequence
    # of position measurements and the motion model.
    H_gps = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])

    # All six GPS components are assumed to have the same variance
    # sigma_gps^2. Therefore, the covariance matrix is simply sigma_gps^2
    # multiplied by the identity matrix.
    R_gps = sigma_gps**2 * np.eye(6)

    # --------------------------------------------------------
    # Combined measurement matrix
    # Combined measurement contains:
    # [x1_GPS, y1_GPS,
    #  x2_GPS, y2_GPS,
    #  x3_GPS, y3_GPS,
    #  x2 - x1, y2 - y1,
    #  x3 - x1, y3 - y1,
    #  x3 - x2, y3 - y2]
    # --------------------------------------------------------
    # H_comb is the measurement matrix for the GPS + relative position model.
    # The first six rows are identical to H_gps. The last six rows compute all
    # pairwise relative displacement vectors between the three drones:
    #
    #     Drone 2 with respect to Drone 1:  x2 - x1, y2 - y1
    #     Drone 3 with respect to Drone 1:  x3 - x1, y3 - y1
    #     Drone 3 with respect to Drone 2:  x3 - x2, y3 - y2
    #
    # These measurements describe the internal geometry of the three-drone
    # formation. They do not directly give the global position of the
    # formation, but they connect the drones inside one joint state estimate.
    H_comb = np.array([
        # GPS rows
        [1.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
        [0.0,  1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],

        [0.0,  0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],

        [0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0],

        # Relative position: Drone 2 - Drone 1
        [-1.0, 0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],

        # Relative position: Drone 3 - Drone 1
        [-1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0],

        # Relative position: Drone 3 - Drone 2
        [0.0,  0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0],
    ])

    # The combined measurement vector contains two different sensor types.
    # The first six components are GPS measurements with variance sigma_gps^2.
    # The last six components are relative position measurements with variance
    # sigma_rel^2. For that reason, R_comb cannot be written as one scalar
    # times an identity matrix.
    R_comb = np.diag([
        sigma_gps**2,
        sigma_gps**2,
        sigma_gps**2,
        sigma_gps**2,
        sigma_gps**2,
        sigma_gps**2,
        sigma_rel**2,
        sigma_rel**2,
        sigma_rel**2,
        sigma_rel**2,
        sigma_rel**2,
        sigma_rel**2,
    ])

    return A, Q, H_gps, R_gps, H_comb, R_comb

# ============================================================
# Simulation of true trajectory
# ============================================================

def simulate_true_trajectory(A, Q, x0, n_steps):
    """
    Simulate the true hidden states of the three drones.

    Returns
    -------
    x_true : ndarray, shape (n_steps, 12)
        True state trajectory.
    """

    # This function generates the true hidden state X_k used in the simulated
    # experiment. In a real drone localization problem, this true state would
    # not be known. It is available here only because the experiment is a
    # simulation and it is needed later for error evaluation.
    #
    # The state has twelve components:
    #
    #     [x1, y1, vx1, vy1,
    #      x2, y2, vx2, vy2,
    #      x3, y3, vx3, vy3].
    #
    # The evolution follows the model X_k = A X_{k-1} + W_k, where W_k is
    # Gaussian process noise with covariance Q.

    x_true = np.zeros((n_steps, 12))
    x_true[0] = x0

    for k in range(1, n_steps):
        # Draw a random process noise vector W_k. This represents unmodeled
        # acceleration and other imperfections of the constant-velocity model.
        process_noise = np.random.multivariate_normal(
            mean=np.zeros(12),
            cov=Q
        )

        # Propagate the previous true state through the transition model and
        # add process noise. The velocity would remain exactly constant if
        # Q were zero. Since Q is nonzero, the true trajectory can slowly
        # deviate from a straight-line motion.
        x_true[k] = A @ x_true[k - 1] + process_noise

    return x_true


# ============================================================
# Simulation of measurements
# ============================================================

def simulate_measurements(x_true, sigma_gps, sigma_rel):
    """
    Generate noisy GPS measurements and noisy relative position measurements.

    Returns
    -------
    y_gps : ndarray, shape (n_steps, 6)
        GPS-only measurements.
    y_comb : ndarray, shape (n_steps, 12)
        Combined GPS + relative position measurements.
    """

    # This function generates the observation sequences used by the filters.
    # The measurements correspond to the measurement models from Chapter 3:
    #
    #     Y_k^{GPS}  = H_gps  X_k + V_k^{GPS},
    #     Y_k^{comb} = H_comb X_k + V_k^{comb}.
    #
    # The first measurement sequence contains only noisy GPS positions.
    # The second sequence contains the same GPS positions and additionally all
    # pairwise relative position vectors between the three drones.

    n_steps = x_true.shape[0]

    y_gps = np.zeros((n_steps, 6))
    y_comb = np.zeros((n_steps, 12))

    for k in range(n_steps):
        # True absolute positions of all three drones
        # Only positions are used here, because the sensors in the project do
        # not measure velocities directly.
        x1 = x_true[k, 0]
        y1 = x_true[k, 1]
        x2 = x_true[k, 4]
        y2 = x_true[k, 5]
        x3 = x_true[k, 8]
        y3 = x_true[k, 9]

        true_gps_values = np.array([
            x1,
            y1,
            x2,
            y2,
            x3,
            y3
        ])

        # GPS measurement noise. The GPS sensor is modeled as an unbiased
        # sensor, because the noise has zero mean. The spread of the noise is
        # controlled by sigma_gps.
        gps_noise = np.random.normal(
            loc=0.0,
            scale=sigma_gps,
            size=6
        )

        # GPS measurement = true absolute positions + GPS noise.
        # These are the values used by the GPS-only Kalman filter.
        gps_measurement = true_gps_values + gps_noise

        # True pairwise relative position vectors.
        #
        # These are the ideal values of the additional relative measurements:
        #
        #     Drone 2 - Drone 1: [x2 - x1, y2 - y1]
        #     Drone 3 - Drone 1: [x3 - x1, y3 - y1]
        #     Drone 3 - Drone 2: [x3 - x2, y3 - y2]
        #
        # The opposite directions, for example Drone 1 - Drone 2, contain the
        # same information with the opposite sign. Therefore, each unordered
        # pair is included once.
        true_relative_values = np.array([
            x2 - x1,
            y2 - y1,
            x3 - x1,
            y3 - y1,
            x3 - x2,
            y3 - y2
        ])

        # Relative measurement noise. This noise is also zero-mean Gaussian,
        # but it has a smaller standard deviation than GPS noise in the
        # default experiment.
        relative_noise = np.random.normal(
            loc=0.0,
            scale=sigma_rel,
            size=6
        )

        # Relative measurement = true relative displacement + sensor noise.
        relative_measurement = true_relative_values + relative_noise

        # The GPS-only measurement vector contains only absolute GPS positions.
        y_gps[k] = gps_measurement

        # The combined measurement vector contains GPS positions followed by
        # all pairwise relative position measurements. Using the same GPS
        # values in both filters makes the comparison fair: the only additional
        # information in the combined filter is the relative measurement.
        y_comb[k] = np.concatenate([
            gps_measurement,
            relative_measurement
        ])

    return y_gps, y_comb


# ============================================================
# Kalman filter
# ============================================================

def run_kalman_filter(measurements, A, Q, H, R, mu0, P0):
    """
    Run a linear Kalman filter for a given measurement model.

    Returns
    -------
    estimates : ndarray, shape (n_steps, 12)
        Estimated state means.
    covariances : ndarray, shape (n_steps, 12, 12)
        Estimated state covariance matrices.
    """

    # This function implements the recursive Kalman filter equations from
    # Chapter 2. It is intentionally written as a general function: the same
    # function is used for the GPS-only filter and for the combined filter.
    # The difference between the two filters is only in the measurement vector,
    # measurement matrix H and measurement noise covariance R.
    #
    # In each time step, the filter first predicts the state using the motion
    # model and then updates this prediction using the current measurement.

    n_steps = measurements.shape[0]
    state_dim = mu0.shape[0]

    # estimates[k] stores the posterior mean mu_k after processing the
    # measurement at time step k. This is the main state estimate.
    estimates = np.zeros((n_steps, state_dim))

    # covariances[k] stores the posterior covariance P_k after processing the
    # measurement at time step k. This matrix represents remaining uncertainty
    # of the estimate.
    covariances = np.zeros((n_steps, state_dim, state_dim))

    # Current mean and covariance before the first filtering step.
    # mu0 and P0 represent the initial prior belief about the state.
    mu = mu0.copy()
    P = P0.copy()

    I = np.eye(state_dim)

    for k in range(n_steps):
        # ----------------------------------------------------
        # Prediction step
        # ----------------------------------------------------
        # Prediction of the mean:
        #
        #     mu_k^- = A mu_{k-1}
        #
        # This propagates the previous estimate through the deterministic
        # part of the motion model before the current measurement is used.
        mu_pred = A @ mu

        # Prediction of the covariance:
        #
        #     P_k^- = A P_{k-1} A^T + Q
        #
        # The term A P A^T propagates previous uncertainty through the motion
        # model. The term Q adds new uncertainty caused by process noise.
        P_pred = A @ P @ A.T + Q

        # ----------------------------------------------------
        # Update step
        # ----------------------------------------------------
        # The predicted measurement is H @ mu_pred. The innovation is the
        # difference between the actual measurement and this predicted
        # measurement:
        #
        #     innovation = y_k - H mu_k^-
        #
        # It represents the new information brought by the current sensor data.
        innovation = measurements[k] - H @ mu_pred

        # Innovation covariance:
        #
        #     S = H P_k^- H^T + R
        #
        # This describes the uncertainty of the predicted measurement. It
        # combines uncertainty coming from the predicted state and uncertainty
        # coming from the sensor noise.
        S = H @ P_pred @ H.T + R

        # Kalman gain:
        #
        #     K = P_k^- H^T S^{-1}
        #
        # The Kalman gain decides how strongly the estimate should react to
        # the innovation. If the predicted state is uncertain and the
        # measurement is accurate, the gain is larger. If the measurement noise
        # is large, the gain is smaller.
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Posterior mean update:
        #
        #     mu_k = mu_k^- + K (y_k - H mu_k^-)
        #
        # The predicted state is corrected in the direction suggested by the
        # innovation.
        mu = mu_pred + K @ innovation

        # Posterior covariance update:
        #
        #     P_k = (I - K H) P_k^-
        #
        # After using the measurement, uncertainty is usually reduced because
        # the measurement provides additional information about the state.
        P = (I - K @ H) @ P_pred

        # Store posterior estimate and covariance for later plotting and error
        # evaluation.
        estimates[k] = mu
        covariances[k] = P

    return estimates, covariances


# ============================================================
# Error evaluation
# ============================================================

def compute_errors(x_true, estimates):
    """
    Compute absolute position errors and relative position errors.

    Returns
    -------
    mean_abs_error : ndarray, shape (n_steps,)
        Mean absolute position error of the three drones.
    rel_error : ndarray, shape (n_steps,)
        Mean error of the estimated pairwise relative position vectors.
    """

    # This function compares the estimated posterior means with the simulated
    # true states. This is possible only because the experiment is simulated.
    # In a real application the true state would usually be unknown.
    #
    # Two types of error are computed:
    # 1. mean absolute position error of all three drones,
    # 2. mean relative position error of the three-drone formation.
    #
    # The first error evaluates global localization accuracy. The second error
    # evaluates how well the filter estimates the relative geometry that is
    # directly constrained by the additional relative measurements.

    # True positions of all three drones
    true_pos_1 = x_true[:, [0, 1]]
    true_pos_2 = x_true[:, [4, 5]]
    true_pos_3 = x_true[:, [8, 9]]

    # Estimated positions of all three drones
    est_pos_1 = estimates[:, [0, 1]]
    est_pos_2 = estimates[:, [4, 5]]
    est_pos_3 = estimates[:, [8, 9]]

    # Absolute position error of each drone
    # Each error is the Euclidean distance between the estimated position and
    # the true position at a given time step.
    error_drone_1 = np.linalg.norm(est_pos_1 - true_pos_1, axis=1)
    error_drone_2 = np.linalg.norm(est_pos_2 - true_pos_2, axis=1)
    error_drone_3 = np.linalg.norm(est_pos_3 - true_pos_3, axis=1)

    # Mean absolute position error of the three drones
    # This gives one global position-error curve for the whole three-drone
    # system.
    mean_abs_error = (error_drone_1 + error_drone_2 + error_drone_3) / 3.0

    # True and estimated pairwise relative position vectors
    # These vectors correspond to:
    #
    #     Drone 2 - Drone 1,
    #     Drone 3 - Drone 1,
    #     Drone 3 - Drone 2.
    true_rel_21 = true_pos_2 - true_pos_1
    true_rel_31 = true_pos_3 - true_pos_1
    true_rel_32 = true_pos_3 - true_pos_2

    est_rel_21 = est_pos_2 - est_pos_1
    est_rel_31 = est_pos_3 - est_pos_1
    est_rel_32 = est_pos_3 - est_pos_2

    # Error of each pairwise relative position vector
    error_rel_21 = np.linalg.norm(est_rel_21 - true_rel_21, axis=1)
    error_rel_31 = np.linalg.norm(est_rel_31 - true_rel_31, axis=1)
    error_rel_32 = np.linalg.norm(est_rel_32 - true_rel_32, axis=1)

    # Mean relative position error of the complete formation
    # This evaluates how accurately the filter estimates the shape of the
    # three-drone formation.
    rel_error = (error_rel_21 + error_rel_31 + error_rel_32) / 3.0

    return mean_abs_error, rel_error


def rmse(error):
    """
    Compute root mean square error.
    """

    # RMSE converts an error curve into one scalar value. It is used in the
    # article to compare the GPS-only filter and the combined filter in a
    # compact numerical form.
    return np.sqrt(np.mean(error**2))


# ============================================================
# Plotting functions
# ============================================================

def plot_trajectories(x_true, y_gps, est_gps, est_comb, output_path):
    """
    Plot true trajectories, GPS measurements, and Kalman estimates.
    """

    # This figure compares three objects from the Kalman filter model:
    #
    # 1. The true hidden state X_k, which is known only in simulation.
    # 2. The noisy GPS measurements Y_k, which are scattered around the true
    #    positions.
    # 3. The posterior means mu_k produced by the Kalman filters.
    #
    # The purpose of the plot is to show visually that the Kalman estimates
    # are smoother than raw GPS measurements and that the combined filter uses
    # the additional relative information to improve the estimated trajectories.

    plt.figure(figsize=(11, 8))

    # True trajectories
    # These curves represent the simulated ground truth of all three drones.
    plt.plot(
        x_true[:, 0], x_true[:, 1],
        linewidth=2,
        label="Drone 1 true trajectory"
    )
    plt.plot(
        x_true[:, 4], x_true[:, 5],
        linewidth=2,
        label="Drone 2 true trajectory"
    )
    plt.plot(
        x_true[:, 8], x_true[:, 9],
        linewidth=2,
        label="Drone 3 true trajectory"
    )

    # Noisy GPS measurements
    # These points are the raw observations of absolute positions. They are
    # visibly scattered because sigma_gps is relatively large.
    plt.scatter(
        y_gps[:, 0], y_gps[:, 1],
        s=14,
        alpha=0.30,
        marker="x",
        label="Drone 1 GPS measurements"
    )
    plt.scatter(
        y_gps[:, 2], y_gps[:, 3],
        s=14,
        alpha=0.30,
        marker="x",
        label="Drone 2 GPS measurements"
    )
    plt.scatter(
        y_gps[:, 4], y_gps[:, 5],
        s=14,
        alpha=0.30,
        marker="x",
        label="Drone 3 GPS measurements"
    )

    # GPS-only Kalman estimates
    # These trajectories are posterior mean estimates from the filter using
    # only GPS measurements. They demonstrate the smoothing effect of the
    # Kalman filter when only absolute measurements are available.
    plt.plot(
        est_gps[:, 0], est_gps[:, 1],
        linestyle="--",
        linewidth=2,
        label="Drone 1 GPS-only KF estimate"
    )
    plt.plot(
        est_gps[:, 4], est_gps[:, 5],
        linestyle="--",
        linewidth=2,
        label="Drone 2 GPS-only KF estimate"
    )
    plt.plot(
        est_gps[:, 8], est_gps[:, 9],
        linestyle="--",
        linewidth=2,
        label="Drone 3 GPS-only KF estimate"
    )

    # GPS + relative position Kalman estimates
    # These trajectories are posterior mean estimates from the filter using
    # both GPS and all pairwise relative position measurements. The additional
    # relative measurements constrain the formation geometry and can also
    # improve the absolute estimates through the joint Kalman update.
    plt.plot(
        est_comb[:, 0], est_comb[:, 1],
        linestyle="-.",
        linewidth=2,
        label="Drone 1 GPS + relative KF estimate"
    )
    plt.plot(
        est_comb[:, 4], est_comb[:, 5],
        linestyle="-.",
        linewidth=2,
        label="Drone 2 GPS + relative KF estimate"
    )
    plt.plot(
        est_comb[:, 8], est_comb[:, 9],
        linestyle="-.",
        linewidth=2,
        label="Drone 3 GPS + relative KF estimate"
    )

    plt.title("True trajectories, GPS measurements and Kalman estimates")
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_absolute_errors(error_gps, error_comb, output_path):
    """
    Plot absolute position error over time.
    """

    # This figure shows the global position estimation error over time.
    # For each time step, the error is the mean of the Euclidean position
    # errors of Drone 1, Drone 2 and Drone 3.
    #
    # The plot answers the main localization question:
    # how far are the estimated absolute positions from the true positions?

    time_steps = np.arange(len(error_gps))

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_steps,
        error_gps,
        linewidth=2,
        label="GPS-only Kalman filter"
    )
    plt.plot(
        time_steps,
        error_comb,
        linewidth=2,
        label="GPS + relative Kalman filter"
    )

    plt.title("Mean absolute position error")
    plt.xlabel("time step k")
    plt.ylabel("mean absolute position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_relative_errors(rel_error_gps, rel_error_comb, output_path):
    """
    Plot relative position error over time.
    """

    # This figure shows the mean error of the estimated pairwise relative
    # position vectors:
    #
    #     Drone 2 - Drone 1,
    #     Drone 3 - Drone 1,
    #     Drone 3 - Drone 2.
    #
    # This is the most direct evaluation of the additional relative
    # measurements. Since the combined filter explicitly observes these
    # pairwise vectors, theory predicts that the relative position error should
    # be significantly smaller for the GPS + relative Kalman filter than for
    # the GPS-only filter.

    time_steps = np.arange(len(rel_error_gps))

    plt.figure(figsize=(10, 5))

    plt.plot(
        time_steps,
        rel_error_gps,
        linewidth=2,
        label="GPS-only Kalman filter"
    )
    plt.plot(
        time_steps,
        rel_error_comb,
        linewidth=2,
        label="GPS + relative Kalman filter"
    )

    plt.title("Mean pairwise relative position error")
    plt.xlabel("time step k")
    plt.ylabel("mean relative position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rmse_comparison(rmse_values, output_path):
    """
    Plot RMSE comparison as a bar chart.
    """

    # The RMSE bar chart gives a compact summary of the experiment. The first
    # two bars compare absolute position accuracy. The last two bars compare
    # relative position accuracy. This figure is useful in the article because
    # it summarizes the complete time-dependent error curves in one visual
    # comparison.

    labels = list(rmse_values.keys())
    values = list(rmse_values.values())

    plt.figure(figsize=(8, 5))

    plt.bar(labels, values)

    plt.title("RMSE comparison")
    plt.ylabel("RMSE [m]")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Main experiment
# ============================================================

def main():
    """
    Run the complete experiment.
    """
    # The main function executes the complete computational experiment in the
    # same order as the project methodology:
    #
    # 1. create the mathematical model,
    # 2. simulate the hidden true trajectory,
    # 3. simulate noisy observations,
    # 4. run both Kalman filters,
    # 5. compute errors and RMSE values,
    # 6. save figures for the article.

    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Cooperative localization of three drones using a Kalman filter")
    print("-------------------------------------------------------------")
    print(f"Output figures directory: {FIGURES_DIR}")
    print()

    # --------------------------------------------------------
    # Step 1: Create model matrices
    # --------------------------------------------------------
    # This step constructs the matrices that define the three-drone state-space
    # model. They correspond directly to the equations in Chapter 3 of the
    # article, extended from two drones to three drones.
    A, Q, H_gps, R_gps, H_comb, R_comb = create_model(
        DT, SIGMA_A, SIGMA_GPS, SIGMA_REL
    )

    print("Step 1: Model matrices created")
    print("--------------------------------")
    print(f"A shape:       {A.shape}")
    print(f"Q shape:       {Q.shape}")
    print(f"H_gps shape:   {H_gps.shape}")
    print(f"R_gps shape:   {R_gps.shape}")
    print(f"H_comb shape:  {H_comb.shape}")
    print(f"R_comb shape:  {R_comb.shape}")
    print()

    # Simple test of measurement matrices
    # This artificial state verifies that H_gps selects absolute positions and
    # H_comb additionally computes all pairwise relative position vectors.
    test_state = np.array([
        10.0, 20.0, 1.0, 2.0,
        40.0, 70.0, 3.0, 4.0,
        60.0, 30.0, 5.0, 6.0
    ])

    print("Measurement matrix test")
    print("-----------------------")
    print(f"H_gps @ test_state:  {H_gps @ test_state}")
    print(f"H_comb @ test_state: {H_comb @ test_state}")
    print()

    # --------------------------------------------------------
    # Step 2: Simulate true trajectory
    # --------------------------------------------------------
    # This is the initial hidden state of the simulated system. Each group of
    # four values represents the position and velocity of one drone.
    x0_true = np.array([
        0.0, 0.0, 1.5, 0.8,        # Drone 1: x, y, vx, vy
        40.0, 20.0, 1.0, 1.1,      # Drone 2: x, y, vx, vy
        20.0, 55.0, 1.2, 0.6       # Drone 3: x, y, vx, vy
    ])

    # The true trajectory is generated using the same transition matrix A and
    # process noise covariance Q as defined by the model. This gives the
    # hidden state sequence against which the filter estimates are evaluated.
    x_true = simulate_true_trajectory(A, Q, x0_true, N_STEPS)

    print("Step 2: True trajectory simulated")
    print("---------------------------------")
    print(f"x_true shape: {x_true.shape}")
    print(f"Initial true state: {x_true[0]}")
    print(f"Final true state:   {x_true[-1]}")
    print()

    # --------------------------------------------------------
    # Step 3: Simulate measurements
    # --------------------------------------------------------
    # The simulated measurements are generated from the true trajectory by
    # adding zero-mean Gaussian noise. The GPS-only filter receives y_gps.
    # The combined filter receives y_comb.
    y_gps, y_comb = simulate_measurements(
        x_true, SIGMA_GPS, SIGMA_REL
    )

    print("Step 3: Measurements simulated")
    print("------------------------------")
    print(f"y_gps shape:  {y_gps.shape}")
    print(f"y_comb shape: {y_comb.shape}")
    print()

    print("First measurement test")
    print("----------------------")
    print(f"True GPS values:       {x_true[0, [0, 1, 4, 5, 8, 9]]}")
    print(f"GPS measurement:       {y_gps[0]}")
    print(f"True relative vectors: {np.array([x_true[0, 4] - x_true[0, 0], x_true[0, 5] - x_true[0, 1], x_true[0, 8] - x_true[0, 0], x_true[0, 9] - x_true[0, 1], x_true[0, 8] - x_true[0, 4], x_true[0, 9] - x_true[0, 5]])}")
    print(f"Relative measurements: {y_comb[0, 6:12]}")
    print()

    # --------------------------------------------------------
    # Step 4: Run both Kalman filters
    # --------------------------------------------------------
    # The initial estimate mu0 intentionally contains correct initial
    # positions but zero initial velocities. This means that the filter has to
    # infer velocities from the motion model and from the sequence of
    # measurements.
    mu0 = np.array([
        0.0, 0.0, 0.0, 0.0,
        40.0, 20.0, 0.0, 0.0,
        20.0, 55.0, 0.0, 0.0
    ])

    # P0 represents initial uncertainty. Position variances are larger than
    # velocity variances, but both are nonzero. This expresses that the initial
    # state estimate is not exact.
    P0 = np.diag([
        100.0, 100.0, 25.0, 25.0,
        100.0, 100.0, 25.0, 25.0,
        100.0, 100.0, 25.0, 25.0
    ])

    # GPS-only filter. It uses only absolute GPS position measurements.
    est_gps, cov_gps = run_kalman_filter(
        y_gps, A, Q, H_gps, R_gps, mu0, P0
    )

    # Combined filter. It uses the same GPS measurements and additionally all
    # pairwise relative position measurements. The filtering equations are the
    # same; only H and R are different.
    est_comb, cov_comb = run_kalman_filter(
        y_comb, A, Q, H_comb, R_comb, mu0, P0
    )

    print("Step 4: Kalman filters executed")
    print("-------------------------------")
    print(f"est_gps shape:   {est_gps.shape}")
    print(f"cov_gps shape:   {cov_gps.shape}")
    print(f"est_comb shape:  {est_comb.shape}")
    print(f"cov_comb shape:  {cov_comb.shape}")
    print()

    print("First estimate test")
    print("-------------------")
    print(f"Initial true state:      {x_true[0]}")
    print(f"First GPS-only estimate: {est_gps[0]}")
    print(f"First combined estimate: {est_comb[0]}")
    print()

    # --------------------------------------------------------
    # Step 5: Compute errors and RMSE
    # --------------------------------------------------------
    # Error curves are computed for both filters. They are later used in the
    # result plots and in the numerical RMSE comparison.
    abs_error_gps, rel_error_gps = compute_errors(x_true, est_gps)
    abs_error_comb, rel_error_comb = compute_errors(x_true, est_comb)

    # RMSE values summarize complete error curves into scalar metrics.
    rmse_abs_gps = rmse(abs_error_gps)
    rmse_abs_comb = rmse(abs_error_comb)
    rmse_rel_gps = rmse(rel_error_gps)
    rmse_rel_comb = rmse(rel_error_comb)

    print("Step 5: Errors computed")
    print("-----------------------")
    print(f"abs_error_gps shape:   {abs_error_gps.shape}")
    print(f"rel_error_gps shape:   {rel_error_gps.shape}")
    print(f"abs_error_comb shape:  {abs_error_comb.shape}")
    print(f"rel_error_comb shape:  {rel_error_comb.shape}")
    print()

    print("RMSE comparison")
    print("---------------")
    print(f"GPS-only absolute RMSE:        {rmse_abs_gps:.3f} m")
    print(f"Combined absolute RMSE:        {rmse_abs_comb:.3f} m")
    print(f"GPS-only relative RMSE:        {rmse_rel_gps:.3f} m")
    print(f"Combined relative RMSE:        {rmse_rel_comb:.3f} m")
    print()

    # --------------------------------------------------------
    # Step 6: Create plots
    # --------------------------------------------------------
    # The following figures are the main graphical outputs of the project.
    # They are saved to the figures/ directory and can be included directly in
    # the LaTeX article.
    plot_trajectories(
        x_true, y_gps, est_gps, est_comb,
        FIGURES_DIR / "trajectories_three_drones.png"
    )

    plot_absolute_errors(
        abs_error_gps, abs_error_comb,
        FIGURES_DIR / "absolute_error_three_drones.png"
    )

    plot_relative_errors(
        rel_error_gps, rel_error_comb,
        FIGURES_DIR / "relative_error_three_drones.png"
    )

    plot_rmse_comparison(
        {
            "GPS abs.": rmse_abs_gps,
            "Combined abs.": rmse_abs_comb,
            "GPS rel.": rmse_rel_gps,
            "Combined rel.": rmse_rel_comb,
        },
        FIGURES_DIR / "rmse_comparison_three_drones.png"
    )

    print("Step 6: Figures created")
    print("-----------------------")
    print(f"Trajectory plot:       {FIGURES_DIR / 'trajectories_three_drones.png'}")
    print(f"Absolute error plot:   {FIGURES_DIR / 'absolute_error_three_drones.png'}")
    print(f"Relative error plot:   {FIGURES_DIR / 'relative_error_three_drones.png'}")
    print(f"RMSE comparison plot:  {FIGURES_DIR / 'rmse_comparison_three_drones.png'}")
    print()

    print("Experiment completed.")


if __name__ == "__main__":
    main()
