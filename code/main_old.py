"""
Cooperative Localization of Two Drones Using a Kalman Filter

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

RANDOM_SEED = 7

# Simulation parameters
DT = 1.0                 # sampling period [s]
N_STEPS = 100            # number of simulation steps

# Noise parameters
SIGMA_A = 0.3            # process acceleration noise [m/s^2]
SIGMA_GPS = 8.0          # GPS position measurement noise [m]
SIGMA_REL = 1.0          # relative position measurement noise [m]

# Output paths
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
    A : ndarray, shape (8, 8)
        State transition matrix.
    Q : ndarray, shape (8, 8)
        Process noise covariance matrix.
    H_gps : ndarray, shape (4, 8)
        GPS-only measurement matrix.
    R_gps : ndarray, shape (4, 4)
        GPS-only measurement noise covariance matrix.
    H_comb : ndarray, shape (6, 8)
        Combined GPS + relative position measurement matrix.
    R_comb : ndarray, shape (6, 6)
        Combined measurement noise covariance matrix.
    """

    # --------------------------------------------------------
    # State transition matrix for one drone
    # State ordering for one drone:
    # [x, y, vx, vy]
    # --------------------------------------------------------
    F = np.array([
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # --------------------------------------------------------
    # Full state transition matrix for two drones
    # Full state ordering:
    # [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
    # --------------------------------------------------------
    zero_4x4 = np.zeros((4, 4))

    A = np.block([
        [F,        zero_4x4],
        [zero_4x4, F       ],
    ])

    # --------------------------------------------------------
    # Process noise covariance for one drone
    # Based on unknown acceleration noise during one time step.
    # --------------------------------------------------------
    Q_drone = sigma_a**2 * np.array([
        [dt**4 / 4.0, 0.0,          dt**3 / 2.0, 0.0         ],
        [0.0,         dt**4 / 4.0,  0.0,          dt**3 / 2.0],
        [dt**3 / 2.0, 0.0,          dt**2,        0.0         ],
        [0.0,         dt**3 / 2.0,  0.0,          dt**2       ],
    ])

    # Full process noise covariance for two drones
    Q = np.block([
        [Q_drone,  zero_4x4],
        [zero_4x4, Q_drone ],
    ])

    # --------------------------------------------------------
    # GPS-only measurement matrix
    # GPS measures:
    # [x1, y1, x2, y2]
    # --------------------------------------------------------
    H_gps = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])

    R_gps = sigma_gps**2 * np.eye(4)

    # --------------------------------------------------------
    # Combined measurement matrix
    # Combined measurement contains:
    # [x1_GPS, y1_GPS, x2_GPS, y2_GPS, x2 - x1, y2 - y1]
    # --------------------------------------------------------
    H_comb = np.array([
        [1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])

    R_comb = np.diag([
        sigma_gps**2,
        sigma_gps**2,
        sigma_gps**2,
        sigma_gps**2,
        sigma_rel**2,
        sigma_rel**2,
    ])

    return A, Q, H_gps, R_gps, H_comb, R_comb

# ============================================================
# Simulation of true trajectory
# ============================================================

def simulate_true_trajectory(A, Q, x0, n_steps):
    """
    Simulate the true hidden states of the two drones.

    Returns
    -------
    x_true : ndarray, shape (n_steps, 8)
        True state trajectory.
    """

    x_true = np.zeros((n_steps, 8))
    x_true[0] = x0

    for k in range(1, n_steps):
        process_noise = np.random.multivariate_normal(
            mean=np.zeros(8),
            cov=Q
        )

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
    y_gps : ndarray, shape (n_steps, 4)
        GPS-only measurements.
    y_comb : ndarray, shape (n_steps, 6)
        Combined GPS + relative position measurements.
    """

    n_steps = x_true.shape[0]

    y_gps = np.zeros((n_steps, 4))
    y_comb = np.zeros((n_steps, 6))

    for k in range(n_steps):
        # True absolute positions of both drones
        x1 = x_true[k, 0]
        y1 = x_true[k, 1]
        x2 = x_true[k, 4]
        y2 = x_true[k, 5]

        true_gps_values = np.array([
            x1,
            y1,
            x2,
            y2
        ])

        gps_noise = np.random.normal(
            loc=0.0,
            scale=sigma_gps,
            size=4
        )

        gps_measurement = true_gps_values + gps_noise

        # True relative position of Drone 2 with respect to Drone 1
        true_relative_values = np.array([
            x2 - x1,
            y2 - y1
        ])

        relative_noise = np.random.normal(
            loc=0.0,
            scale=sigma_rel,
            size=2
        )

        relative_measurement = true_relative_values + relative_noise

        y_gps[k] = gps_measurement
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
    estimates : ndarray, shape (n_steps, 8)
        Estimated state means.
    covariances : ndarray, shape (n_steps, 8, 8)
        Estimated state covariance matrices.
    """

    n_steps = measurements.shape[0]
    state_dim = mu0.shape[0]

    estimates = np.zeros((n_steps, state_dim))
    covariances = np.zeros((n_steps, state_dim, state_dim))

    mu = mu0.copy()
    P = P0.copy()

    I = np.eye(state_dim)

    for k in range(n_steps):
        # ----------------------------------------------------
        # Prediction step
        # ----------------------------------------------------
        mu_pred = A @ mu
        P_pred = A @ P @ A.T + Q

        # ----------------------------------------------------
        # Update step
        # ----------------------------------------------------
        innovation = measurements[k] - H @ mu_pred
        S = H @ P_pred @ H.T + R

        K = P_pred @ H.T @ np.linalg.inv(S)

        mu = mu_pred + K @ innovation
        P = (I - K @ H) @ P_pred

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
        Mean absolute position error of the two drones.
    rel_error : ndarray, shape (n_steps,)
        Error of the estimated relative position vector.
    """

    # True positions of both drones
    true_pos_1 = x_true[:, [0, 1]]
    true_pos_2 = x_true[:, [4, 5]]

    # Estimated positions of both drones
    est_pos_1 = estimates[:, [0, 1]]
    est_pos_2 = estimates[:, [4, 5]]

    # Absolute position error of each drone
    error_drone_1 = np.linalg.norm(est_pos_1 - true_pos_1, axis=1)
    error_drone_2 = np.linalg.norm(est_pos_2 - true_pos_2, axis=1)

    # Mean absolute position error of the two drones
    mean_abs_error = (error_drone_1 + error_drone_2) / 2.0

    # True and estimated relative position vectors
    true_relative_position = true_pos_2 - true_pos_1
    estimated_relative_position = est_pos_2 - est_pos_1

    # Error of the relative position vector
    rel_error = np.linalg.norm(
        estimated_relative_position - true_relative_position,
        axis=1
    )

    return mean_abs_error, rel_error


def rmse(error):
    """
    Compute root mean square error.
    """

    return np.sqrt(np.mean(error**2))


# ============================================================
# Plotting functions
# ============================================================

def plot_trajectories(x_true, y_gps, est_gps, est_comb, output_path):
    """
    Plot true trajectories, GPS measurements, and Kalman estimates.
    """

    plt.figure(figsize=(10, 7))

    # True trajectories
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

    # Noisy GPS measurements
    plt.scatter(
        y_gps[:, 0], y_gps[:, 1],
        s=15,
        alpha=0.35,
        marker="x",
        label="Drone 1 GPS measurements"
    )
    plt.scatter(
        y_gps[:, 2], y_gps[:, 3],
        s=15,
        alpha=0.35,
        marker="x",
        label="Drone 2 GPS measurements"
    )

    # GPS-only Kalman estimates
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

    # GPS + relative position Kalman estimates
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

    plt.title("True trajectories, GPS measurements and Kalman estimates")
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_absolute_errors(error_gps, error_comb, output_path):
    """
    Plot absolute position error over time.
    """

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

    plt.title("Relative position error")
    plt.xlabel("time step k")
    plt.ylabel("relative position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_rmse_comparison(rmse_values, output_path):
    """
    Plot RMSE comparison as a bar chart.
    """

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

    This function will be gradually activated as individual functions
    are implemented.
    """
    np.random.seed(RANDOM_SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Cooperative localization of two drones using a Kalman filter")
    print("----------------------------------------------------------")
    print(f"Output figures directory: {FIGURES_DIR}")
    print()

    # --------------------------------------------------------
    # Step 1: Create model matrices
    # --------------------------------------------------------
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
    test_state = np.array([
        10.0, 20.0, 1.0, 2.0,
        40.0, 70.0, 3.0, 4.0
    ])

    print("Measurement matrix test")
    print("-----------------------")
    print(f"H_gps @ test_state:  {H_gps @ test_state}")
    print(f"H_comb @ test_state: {H_comb @ test_state}")
    print()

    # --------------------------------------------------------
    # Step 2: Simulate true trajectory
    # --------------------------------------------------------
    x0_true = np.array([
        0.0, 0.0, 1.5, 0.8,       # Drone 1: x, y, vx, vy
        40.0, 20.0, 1.0, 1.1      # Drone 2: x, y, vx, vy
    ])

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
    print(f"True GPS values:      {x_true[0, [0, 1, 4, 5]]}")
    print(f"GPS measurement:      {y_gps[0]}")
    print(f"True relative vector: {x_true[0, [4, 5]] - x_true[0, [0, 1]]}")
    print(f"Relative measurement: {y_comb[0, 4:6]}")
    print()

    # --------------------------------------------------------
    # Step 4: Run both Kalman filters
    # --------------------------------------------------------
    mu0 = np.array([
        0.0, 0.0, 0.0, 0.0,
        40.0, 20.0, 0.0, 0.0
    ])

    P0 = np.diag([
        100.0, 100.0, 25.0, 25.0,
        100.0, 100.0, 25.0, 25.0
    ])

    est_gps, cov_gps = run_kalman_filter(
        y_gps, A, Q, H_gps, R_gps, mu0, P0
    )

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
    abs_error_gps, rel_error_gps = compute_errors(x_true, est_gps)
    abs_error_comb, rel_error_comb = compute_errors(x_true, est_comb)

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
    plot_trajectories(
        x_true, y_gps, est_gps, est_comb,
        FIGURES_DIR / "trajectories.png"
    )

    plot_absolute_errors(
        abs_error_gps, abs_error_comb,
        FIGURES_DIR / "absolute_error.png"
    )

    plot_relative_errors(
        rel_error_gps, rel_error_comb,
        FIGURES_DIR / "relative_error.png"
    )

    plot_rmse_comparison(
        {
            "GPS abs.": rmse_abs_gps,
            "Combined abs.": rmse_abs_comb,
            "GPS rel.": rmse_rel_gps,
            "Combined rel.": rmse_rel_comb,
        },
        FIGURES_DIR / "rmse_comparison.png"
    )

    print("Step 6: Figures created")
    print("-----------------------")
    print(f"Trajectory plot:       {FIGURES_DIR / 'trajectories.png'}")
    print(f"Absolute error plot:   {FIGURES_DIR / 'absolute_error.png'}")
    print(f"Relative error plot:   {FIGURES_DIR / 'relative_error.png'}")
    print(f"RMSE comparison plot:  {FIGURES_DIR / 'rmse_comparison.png'}")
    print()

    print("Experiment completed.")


if __name__ == "__main__":
    main()