#!/usr/bin/env python3
"""
Structured Instability in Learning Dynamics: Empirical Simulation Suite
========================================================================
Companion code for: "Instability as Insight: Reinterpreting Learning Dynamics
in Repeated Games through Bounded Rationality" — Pratyush Mahadevaiah

This module implements simulations across three game classes (zero-sum,
potential, and general-sum) using Q-learning and replicator dynamics,
then extracts dynamical signatures of structured instability using:
  - Lyapunov exponent estimation
  - Entropy rate time-series
  - Transfer entropy (opponent modeling)
  - Phase portraits & Poincaré sections
  - PCA of joint strategy evolution
  - Frequency spectra (FFT)
  - Eigenvalue analysis of local Jacobians
  - KL-divergence rationality tracking
  - Surrogate-data significance testing

Author: Pratyush Mahadevaiah
"""

import numpy as np
from scipy import signal, stats
from scipy.linalg import eigvals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# SECTION 1: GAME DEFINITIONS
# ============================================================================

GAMES = {
    # --- Zero-Sum Games ---
    'RPS': {
        'class': 'zero-sum',
        'name': 'Rock-Paper-Scissors',
        'payoff_A': np.array([[0, -1, 1],
                              [1,  0, -1],
                              [-1, 1,  0]], dtype=float),
        'payoff_B': np.array([[0,  1, -1],
                              [-1, 0,  1],
                              [1, -1,  0]], dtype=float),
        'description': 'Canonical zero-sum cycling game'
    },
    'Shapley': {
        'class': 'zero-sum',
        'name': "Shapley's Game (3x3)",
        'payoff_A': np.array([[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]], dtype=float),
        'payoff_B': np.array([[0, 1, 0],
                              [0, 0, 1],
                              [1, 0, 0]], dtype=float),
        'description': 'Classical example of persistent cycling'
    },
    'MatchingPennies': {
        'class': 'zero-sum',
        'name': 'Matching Pennies (perturbed)',
        'payoff_A': np.array([[1.0, -1.0],
                              [-1.0, 1.0]]),
        'payoff_B': np.array([[-1.0, 1.0],
                              [1.0, -1.0]]),
        'description': 'Perturbed matching pennies with Q-learning'
    },
    # --- Potential Games ---
    'StagHunt': {
        'class': 'potential',
        'name': 'Stag Hunt (with exploration)',
        'payoff_A': np.array([[4.0, 0.0],
                              [3.0, 2.0]]),
        'payoff_B': np.array([[4.0, 3.0],
                              [0.0, 2.0]]),
        'description': 'Coordination game revealing partial convergence'
    },
    'Coordination3x3': {
        'class': 'potential',
        'name': '3x3 Coordination Game',
        'payoff_A': np.array([[3.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0],
                              [0.0, 0.0, 1.0]]),
        'payoff_B': np.array([[3.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0],
                              [0.0, 0.0, 1.0]]),
        'description': 'Multi-equilibrium coordination with exploration'
    },
    # --- General-Sum Games ---
    'BattleOfSexes': {
        'class': 'general-sum',
        'name': 'Battle of the Sexes',
        'payoff_A': np.array([[3.0, 0.0],
                              [0.0, 2.0]]),
        'payoff_B': np.array([[2.0, 0.0],
                              [0.0, 3.0]]),
        'description': 'Asymmetric coordination / general-sum'
    },
    'MinorityGame': {
        'class': 'general-sum',
        'name': 'Minority Game (3-action)',
        'payoff_A': np.array([[0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0]]),
        'payoff_B': np.array([[0.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0]]),
        'description': 'Adaptive market / minority game — structured chaos'
    },
    'PrisonersDilemma': {
        'class': 'general-sum',
        'name': "Prisoner's Dilemma",
        'payoff_A': np.array([[3.0, 0.0],
                              [5.0, 1.0]]),
        'payoff_B': np.array([[3.0, 5.0],
                              [0.0, 1.0]]),
        'description': 'Classic general-sum social dilemma'
    },
}

# ============================================================================
# SECTION 2: LEARNING ALGORITHMS
# ============================================================================

def softmax(q_values, temperature):
    """Boltzmann softmax policy with numerical stability."""
    q = q_values - np.max(q_values)
    exp_q = np.exp(q / max(temperature, 1e-10))
    return exp_q / np.sum(exp_q)


def run_q_learning(payoff_A, payoff_B, T=10000, alpha=0.1, gamma=0.95,
                   tau_init=1.0, tau_decay=0.9995, seed=None):
    """
    Multi-agent Q-learning with Boltzmann exploration.

    Returns strategy trajectories for both agents (T x n_actions each).
    """
    if seed is not None:
        np.random.seed(seed)

    n_A = payoff_A.shape[0]
    n_B = payoff_A.shape[1]

    Q_A = np.zeros(n_A)
    Q_B = np.zeros(n_B)

    traj_A = np.zeros((T, n_A))
    traj_B = np.zeros((T, n_B))
    actions_A = np.zeros(T, dtype=int)
    actions_B = np.zeros(T, dtype=int)

    tau = tau_init

    for t in range(T):
        # Compute policies
        pi_A = softmax(Q_A, tau)
        pi_B = softmax(Q_B, tau)

        traj_A[t] = pi_A
        traj_B[t] = pi_B

        # Sample actions
        a = np.random.choice(n_A, p=pi_A)
        b = np.random.choice(n_B, p=pi_B)
        actions_A[t] = a
        actions_B[t] = b

        # Get rewards
        r_A = payoff_A[a, b]
        r_B = payoff_B[a, b]

        # Q-value updates
        Q_A[a] = Q_A[a] + alpha * (r_A + gamma * np.max(Q_A) - Q_A[a])
        Q_B[b] = Q_B[b] + alpha * (r_B + gamma * np.max(Q_B) - Q_B[b])

        tau *= tau_decay

    return traj_A, traj_B, actions_A, actions_B


def run_replicator_dynamics(payoff_A, payoff_B, T=10000, dt=0.01,
                            noise_scale=0.005, seed=None):
    """
    Continuous-time replicator dynamics with small stochastic perturbation.

    Returns strategy trajectories for both agents (T x n_actions each).
    """
    if seed is not None:
        np.random.seed(seed)

    n_A = payoff_A.shape[0]
    n_B = payoff_A.shape[1]

    # Initialize near uniform
    x = np.ones(n_A) / n_A + np.random.randn(n_A) * 0.01
    x = np.clip(x, 0.01, None)
    x /= x.sum()

    y = np.ones(n_B) / n_B + np.random.randn(n_B) * 0.01
    y = np.clip(y, 0.01, None)
    y /= y.sum()

    traj_A = np.zeros((T, n_A))
    traj_B = np.zeros((T, n_B))

    for t in range(T):
        traj_A[t] = x.copy()
        traj_B[t] = y.copy()

        # Fitness for player A
        fitness_A = payoff_A @ y
        avg_A = x @ fitness_A
        dx = x * (fitness_A - avg_A) * dt + noise_scale * np.random.randn(n_A) * dt

        # Fitness for player B
        fitness_B = payoff_B.T @ x
        avg_B = y @ fitness_B
        dy = y * (fitness_B - avg_B) * dt + noise_scale * np.random.randn(n_B) * dt

        x = x + dx
        x = np.clip(x, 1e-8, None)
        x /= x.sum()

        y = y + dy
        y = np.clip(y, 1e-8, None)
        y /= y.sum()

    return traj_A, traj_B


# ============================================================================
# SECTION 3: ANALYTICAL TOOLS — Extracting Structured Instability
# ============================================================================

def compute_entropy(p):
    """Shannon entropy of a probability distribution."""
    p = p[p > 1e-15]
    return -np.sum(p * np.log2(p))


def entropy_timeseries(traj, window=100):
    """Sliding-window entropy of strategy distributions."""
    T = traj.shape[0]
    ent = np.zeros(T)
    for t in range(T):
        ent[t] = compute_entropy(traj[t])
    # Smooth with rolling average
    kernel = np.ones(window) / window
    ent_smooth = np.convolve(ent, kernel, mode='same')
    return ent, ent_smooth


def estimate_lyapunov_exponent(traj, dt=1.0, embed_dim=3, tau=1):
    """
    Estimate the maximal Lyapunov exponent from a 1D time-series
    using the Rosenstein et al. (1993) method (nearest-neighbor divergence).
    """
    # Use first component
    if traj.ndim > 1:
        x = traj[:, 0]
    else:
        x = traj

    N = len(x)
    # Construct delay embedding
    M = N - (embed_dim - 1) * tau
    if M < 50:
        return 0.0, np.array([])

    X = np.zeros((M, embed_dim))
    for i in range(embed_dim):
        X[:, i] = x[i * tau: i * tau + M]

    # Find nearest neighbors (excluding temporal neighbors)
    min_sep = embed_dim * tau + 1
    divergence = []

    for i in range(M):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[max(0, i - min_sep):min(M, i + min_sep)] = np.inf
        j = np.argmin(dists)
        if dists[j] < np.inf:
            divergence.append((i, j, dists[j]))

    if len(divergence) < 10:
        return 0.0, np.array([])

    # Track divergence over time
    max_steps = min(M // 4, 200)
    avg_div = np.zeros(max_steps)
    counts = np.zeros(max_steps)

    for i, j, d0 in divergence:
        for k in range(max_steps):
            if i + k < M and j + k < M:
                d = np.linalg.norm(X[i + k] - X[j + k])
                if d > 1e-15:
                    avg_div[k] += np.log(d)
                    counts[k] += 1

    mask = counts > 0
    avg_div[mask] /= counts[mask]

    # Linear fit to the divergence curve
    valid = np.where(mask)[0]
    if len(valid) < 10:
        return 0.0, avg_div

    # Use first quarter for fit
    fit_range = valid[:len(valid) // 4]
    if len(fit_range) < 5:
        fit_range = valid[:max(5, len(valid))]

    slope, _, _, _, _ = stats.linregress(fit_range * dt, avg_div[fit_range])
    return slope, avg_div


def compute_transfer_entropy(source_actions, target_actions, n_actions_src,
                             n_actions_tgt, lag=1):
    """
    Estimate transfer entropy from source to target action sequences.
    TE(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
    """
    T = len(source_actions)
    if T < lag + 2:
        return 0.0

    # Build joint counts
    # P(Y_t, Y_{t-1}, X_{t-lag})
    joint_counts = {}
    marginal_counts = {}  # P(Y_t, Y_{t-1})
    cond_counts = {}      # P(Y_{t-1}, X_{t-lag})

    for t in range(max(1, lag), T):
        yt = target_actions[t]
        yt1 = target_actions[t - 1]
        xt_lag = source_actions[t - lag]

        key3 = (yt, yt1, xt_lag)
        joint_counts[key3] = joint_counts.get(key3, 0) + 1

        key2 = (yt, yt1)
        marginal_counts[key2] = marginal_counts.get(key2, 0) + 1

        key_c = (yt1, xt_lag)
        cond_counts[key_c] = cond_counts.get(key_c, 0) + 1

    # Also need P(Y_{t-1})
    y_prev_counts = {}
    for t in range(max(1, lag), T):
        yt1 = target_actions[t - 1]
        y_prev_counts[yt1] = y_prev_counts.get(yt1, 0) + 1

    total = T - max(1, lag)
    if total == 0:
        return 0.0

    te = 0.0
    for (yt, yt1, xt_lag), count in joint_counts.items():
        p_joint = count / total
        p_cond_full = count / max(cond_counts.get((yt1, xt_lag), 1), 1)

        marginal_key = (yt, yt1)
        p_cond_marginal = marginal_counts.get(marginal_key, 1) / max(
            y_prev_counts.get(yt1, 1), 1)

        if p_cond_full > 1e-15 and p_cond_marginal > 1e-15:
            te += p_joint * np.log2(p_cond_full / p_cond_marginal)

    return max(te, 0.0)


def compute_kl_divergence(p, q):
    """KL(p || q) — KL divergence from q to p."""
    p = np.clip(p, 1e-15, 1.0)
    q = np.clip(q, 1e-15, 1.0)
    return np.sum(p * np.log2(p / q))


def kl_from_uniform_timeseries(traj):
    """Track KL divergence from uniform distribution over time."""
    T, n = traj.shape
    uniform = np.ones(n) / n
    kl = np.zeros(T)
    for t in range(T):
        kl[t] = compute_kl_divergence(traj[t], uniform)
    return kl


def compute_pca(traj_A, traj_B):
    """PCA of joint strategy evolution."""
    joint = np.hstack([traj_A, traj_B])
    joint_centered = joint - joint.mean(axis=0)
    cov = np.cov(joint_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    projected = joint_centered @ eigenvectors[:, :3]
    return projected, eigenvalues


def compute_frequency_spectrum(traj, component=0):
    """FFT-based frequency spectrum of a strategy component."""
    x = traj[:, component] if traj.ndim > 1 else traj
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(len(x))
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    return freqs, spectrum


def compute_jacobian_eigenvalues(payoff_A, payoff_B, x, y):
    """
    Compute eigenvalues of the Jacobian of the replicator dynamics
    at a given strategy profile (x, y).
    """
    n_A = len(x)
    n_B = len(y)

    # For 2-strategy case, work in reduced coordinates
    # For general case, use finite differences
    eps = 1e-6
    dim = n_A + n_B

    def dynamics(state):
        xa = state[:n_A]
        ya = state[n_A:]
        xa = np.clip(xa, 1e-10, None)
        ya = np.clip(ya, 1e-10, None)
        xa /= xa.sum()
        ya /= ya.sum()

        f_A = payoff_A @ ya
        avg_A = xa @ f_A
        dx = xa * (f_A - avg_A)

        f_B = payoff_B.T @ xa
        avg_B = ya @ f_B
        dy = ya * (f_B - avg_B)

        return np.concatenate([dx, dy])

    state0 = np.concatenate([x, y])
    J = np.zeros((dim, dim))
    f0 = dynamics(state0)

    for i in range(dim):
        state_pert = state0.copy()
        state_pert[i] += eps
        f1 = dynamics(state_pert)
        J[:, i] = (f1 - f0) / eps

    return eigvals(J)


def surrogate_test(traj, metric_func, n_surrogates=100):
    """
    Surrogate data test: shuffle temporal order, recompute metric,
    compare to original.
    """
    original_val = metric_func(traj)
    surrogate_vals = np.zeros(n_surrogates)

    for i in range(n_surrogates):
        shuffled = traj.copy()
        np.random.shuffle(shuffled)
        surrogate_vals[i] = metric_func(shuffled)

    p_value = np.mean(np.abs(surrogate_vals) >= np.abs(original_val))
    return original_val, surrogate_vals, p_value


# ============================================================================
# SECTION 4: COMPREHENSIVE SIMULATION & ANALYSIS
# ============================================================================

def analyze_game(game_key, game_info, output_dir, T=10000):
    """Run full analysis pipeline for one game."""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {game_info['name']} ({game_info['class']})")
    print(f"  {game_info['description']}")
    print(f"{'='*70}")

    payoff_A = game_info['payoff_A']
    payoff_B = game_info['payoff_B']
    n_actions = payoff_A.shape[0]

    results = {'game': game_key, 'class': game_info['class'],
               'name': game_info['name']}

    # --- Run Q-Learning ---
    print("  Running Q-learning...")
    traj_A_q, traj_B_q, act_A_q, act_B_q = run_q_learning(
        payoff_A, payoff_B, T=T, alpha=0.1, gamma=0.95,
        tau_init=1.0, tau_decay=0.9998, seed=42
    )

    # --- Run Replicator Dynamics ---
    print("  Running replicator dynamics...")
    traj_A_r, traj_B_r = run_replicator_dynamics(
        payoff_A, payoff_B, T=T, dt=0.01, noise_scale=0.005, seed=42
    )

    # --- Entropy Time-Series ---
    print("  Computing entropy time-series...")
    ent_A_q, ent_A_q_smooth = entropy_timeseries(traj_A_q, window=200)
    ent_B_q, ent_B_q_smooth = entropy_timeseries(traj_B_q, window=200)
    ent_A_r, ent_A_r_smooth = entropy_timeseries(traj_A_r, window=200)

    results['entropy_q_mean'] = np.mean(ent_A_q[T//2:])
    results['entropy_q_std'] = np.std(ent_A_q[T//2:])
    results['entropy_r_mean'] = np.mean(ent_A_r[T//2:])

    print(f"    Q-learning entropy (last half): {results['entropy_q_mean']:.4f} "
          f"± {results['entropy_q_std']:.4f} bits")
    print(f"    Replicator entropy (last half): {results['entropy_r_mean']:.4f} bits")

    # --- Lyapunov Exponents ---
    print("  Estimating Lyapunov exponents...")
    lya_q, div_q = estimate_lyapunov_exponent(traj_A_q, dt=1.0)
    lya_r, div_r = estimate_lyapunov_exponent(traj_A_r, dt=0.01)
    results['lyapunov_q'] = lya_q
    results['lyapunov_r'] = lya_r
    print(f"    Q-learning λ_max ≈ {lya_q:.4f}")
    print(f"    Replicator λ_max ≈ {lya_r:.4f}")

    # --- Transfer Entropy ---
    print("  Computing transfer entropy...")
    te_AB = compute_transfer_entropy(act_A_q, act_B_q, n_actions, n_actions)
    te_BA = compute_transfer_entropy(act_B_q, act_A_q, n_actions, n_actions)
    results['transfer_entropy_AB'] = te_AB
    results['transfer_entropy_BA'] = te_BA
    print(f"    TE(A→B) = {te_AB:.4f} bits,  TE(B→A) = {te_BA:.4f} bits")

    # --- KL Divergence from Uniform ---
    print("  Computing KL divergence (rationality tracking)...")
    kl_A = kl_from_uniform_timeseries(traj_A_q)
    kl_B = kl_from_uniform_timeseries(traj_B_q)
    results['kl_mean'] = np.mean(kl_A[T//2:])

    # --- PCA ---
    print("  Running PCA on joint strategy evolution...")
    pca_proj_q, pca_eigs_q = compute_pca(traj_A_q, traj_B_q)
    pca_proj_r, pca_eigs_r = compute_pca(traj_A_r, traj_B_r)
    var_explained_q = pca_eigs_q[:3] / pca_eigs_q.sum() * 100
    results['pca_var_explained'] = var_explained_q

    # --- Frequency Spectrum ---
    print("  Computing frequency spectra...")
    freqs_q, spec_q = compute_frequency_spectrum(traj_A_q, 0)
    freqs_r, spec_r = compute_frequency_spectrum(traj_A_r, 0)

    # --- Jacobian Eigenvalues at midpoint ---
    print("  Analyzing Jacobian eigenvalues...")
    mid_x = traj_A_r[T // 2]
    mid_y = traj_B_r[T // 2]
    jac_eigs = compute_jacobian_eigenvalues(payoff_A, payoff_B, mid_x, mid_y)
    results['jacobian_eigenvalues'] = jac_eigs
    print(f"    Jacobian eigenvalues: {np.round(jac_eigs, 4)}")

    # --- Surrogate Test ---
    print("  Running surrogate significance test...")
    def autocorr_metric(tr):
        """Sum of first 10 autocorrelation lags — detects temporal structure."""
        if tr.ndim > 1:
            x = tr[:, 0]
        else:
            x = tr
        x = x - np.mean(x)
        norm = np.sum(x ** 2)
        if norm < 1e-15:
            return 0.0
        acf = np.correlate(x, x, 'full')
        acf = acf[len(acf)//2:]
        acf = acf / (norm + 1e-15)
        return np.sum(acf[1:min(11, len(acf))])

    orig_ent, surr_ents, p_val = surrogate_test(
        traj_A_q[T//2:], autocorr_metric, n_surrogates=50
    )
    results['surrogate_p_value'] = p_val
    print(f"    Original entropy: {orig_ent:.4f}, "
          f"Surrogate mean: {surr_ents.mean():.4f}, p = {p_val:.4f}")

    # ===================================================================
    # PLOTTING
    # ===================================================================
    print("  Generating figures...")

    fig = plt.figure(figsize=(24, 28))
    fig.suptitle(f"{game_info['name']} — Structured Instability Analysis\n"
                 f"Game class: {game_info['class']}",
                 fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Row 1: Strategy Trajectories ----
    # Q-learning trajectories
    ax1 = fig.add_subplot(gs[0, 0:2])
    colors = plt.cm.Set1(np.linspace(0, 1, n_actions))
    for i in range(n_actions):
        ax1.plot(traj_A_q[:, i], alpha=0.7, color=colors[i],
                 label=f'Action {i}', linewidth=0.5)
    ax1.set_title('Q-Learning: Player A Strategy Trajectories', fontsize=10)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Probability')
    ax1.legend(fontsize=7)
    ax1.set_ylim(-0.05, 1.05)

    # Replicator trajectories
    ax2 = fig.add_subplot(gs[0, 2:4])
    for i in range(n_actions):
        ax2.plot(traj_A_r[:, i], alpha=0.7, color=colors[i],
                 label=f'Action {i}', linewidth=0.5)
    ax2.set_title('Replicator Dynamics: Player A Strategy Trajectories', fontsize=10)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Probability')
    ax2.legend(fontsize=7)
    ax2.set_ylim(-0.05, 1.05)

    # ---- Row 2: Entropy + Lyapunov ----
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.plot(ent_A_q, alpha=0.15, color='steelblue', linewidth=0.3)
    ax3.plot(ent_A_q_smooth, color='navy', linewidth=1.5, label='Player A (smoothed)')
    ax3.plot(ent_B_q_smooth, color='firebrick', linewidth=1.5, label='Player B (smoothed)')
    max_ent = np.log2(n_actions)
    ax3.axhline(max_ent, color='gray', linestyle='--', alpha=0.5, label=f'Max entropy = {max_ent:.2f}')
    ax3.set_title('Q-Learning: Entropy Time-Series', fontsize=10)
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Entropy (bits)')
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[1, 2])
    if len(div_q) > 0:
        ax4.plot(div_q[:min(200, len(div_q))], color='darkgreen', linewidth=1)
        ax4.set_title(f'Lyapunov Divergence (Q)\nλ ≈ {lya_q:.4f}', fontsize=9)
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('log(divergence)')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    ax5 = fig.add_subplot(gs[1, 3])
    if len(div_r) > 0:
        ax5.plot(div_r[:min(200, len(div_r))], color='purple', linewidth=1)
        ax5.set_title(f'Lyapunov Divergence (Rep)\nλ ≈ {lya_r:.4f}', fontsize=9)
        ax5.set_xlabel('Steps')
        ax5.set_ylabel('log(divergence)')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')

    # ---- Row 3: Phase Portrait + Poincaré + Transfer Entropy ----
    if n_actions >= 3:
        # Simplex phase portrait (Q-learning)
        ax6 = fig.add_subplot(gs[2, 0])
        # Project onto 2D simplex
        s = traj_A_q
        x_simp = s[:, 1] + 0.5 * s[:, 2]
        y_simp = (np.sqrt(3) / 2) * s[:, 2]
        scatter = ax6.scatter(x_simp[::5], y_simp[::5], c=np.arange(0, T, 5),
                              cmap='viridis', s=0.5, alpha=0.5)
        # Draw simplex boundary
        ax6.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', linewidth=0.5)
        ax6.set_title('Q-Learning Phase Portrait\n(Simplex)', fontsize=9)
        ax6.set_aspect('equal')
        ax6.set_xlim(-0.1, 1.1)
        ax6.set_ylim(-0.1, 1.0)

        # Simplex phase portrait (Replicator)
        ax7 = fig.add_subplot(gs[2, 1])
        s = traj_A_r
        x_simp = s[:, 1] + 0.5 * s[:, 2]
        y_simp = (np.sqrt(3) / 2) * s[:, 2]
        ax7.scatter(x_simp[::5], y_simp[::5], c=np.arange(0, T, 5),
                    cmap='magma', s=0.5, alpha=0.5)
        ax7.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', linewidth=0.5)
        ax7.set_title('Replicator Phase Portrait\n(Simplex)', fontsize=9)
        ax7.set_aspect('equal')
        ax7.set_xlim(-0.1, 1.1)
        ax7.set_ylim(-0.1, 1.0)
    else:
        # 2-action: plot p(action 0) for A vs B
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.scatter(traj_A_q[::5, 0], traj_B_q[::5, 0], c=np.arange(0, T, 5),
                    cmap='viridis', s=0.5, alpha=0.5)
        ax6.set_title('Q-Learning Phase Portrait\n(p_A vs p_B)', fontsize=9)
        ax6.set_xlabel('Player A: P(action 0)')
        ax6.set_ylabel('Player B: P(action 0)')

        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(traj_A_r[::5, 0], traj_B_r[::5, 0], c=np.arange(0, T, 5),
                    cmap='magma', s=0.5, alpha=0.5)
        ax7.set_title('Replicator Phase Portrait\n(p_A vs p_B)', fontsize=9)
        ax7.set_xlabel('Player A: P(action 0)')
        ax7.set_ylabel('Player B: P(action 0)')

    # Poincaré section (crossing of first component through 1/n_actions)
    ax8 = fig.add_subplot(gs[2, 2])
    threshold = 1.0 / n_actions
    crossings_t = []
    crossings_val = []
    for t in range(1, T):
        if traj_A_q[t-1, 0] < threshold <= traj_A_q[t, 0]:
            crossings_t.append(t)
            if n_actions >= 3:
                crossings_val.append(traj_A_q[t, 1])
            else:
                crossings_val.append(traj_B_q[t, 0])
    if len(crossings_t) > 2:
        ax8.scatter(crossings_val[:-1], crossings_val[1:], s=3, alpha=0.7, c='darkred')
        ax8.set_title(f'Poincaré Return Map\n({len(crossings_t)} crossings)', fontsize=9)
        ax8.set_xlabel('Value at crossing n')
        ax8.set_ylabel('Value at crossing n+1')
    else:
        ax8.text(0.5, 0.5, 'Too few crossings', ha='center', va='center',
                 transform=ax8.transAxes)
        ax8.set_title('Poincaré Return Map', fontsize=9)

    # Transfer entropy over time windows
    ax9 = fig.add_subplot(gs[2, 3])
    window_te = 500
    te_series_AB = []
    te_series_BA = []
    te_times = []
    for start in range(0, T - window_te, window_te // 2):
        end = start + window_te
        te_ab = compute_transfer_entropy(
            act_A_q[start:end], act_B_q[start:end], n_actions, n_actions)
        te_ba = compute_transfer_entropy(
            act_B_q[start:end], act_A_q[start:end], n_actions, n_actions)
        te_series_AB.append(te_ab)
        te_series_BA.append(te_ba)
        te_times.append((start + end) / 2)
    ax9.plot(te_times, te_series_AB, label='TE(A→B)', color='navy', linewidth=1.5)
    ax9.plot(te_times, te_series_BA, label='TE(B→A)', color='firebrick', linewidth=1.5)
    ax9.set_title('Transfer Entropy Over Time\n(Opponent Modeling)', fontsize=9)
    ax9.set_xlabel('Time step')
    ax9.set_ylabel('Transfer entropy (bits)')
    ax9.legend(fontsize=7)

    # ---- Row 4: Frequency Spectrum + PCA + KL Divergence ----
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.semilogy(freqs_q[1:len(freqs_q)//4], spec_q[1:len(spec_q)//4],
                  color='steelblue', linewidth=0.8)
    ax10.set_title('Frequency Spectrum (Q-Learning)', fontsize=9)
    ax10.set_xlabel('Frequency')
    ax10.set_ylabel('Power')

    ax11 = fig.add_subplot(gs[3, 1])
    ax11.semilogy(freqs_r[1:len(freqs_r)//4], spec_r[1:len(spec_r)//4],
                  color='purple', linewidth=0.8)
    ax11.set_title('Frequency Spectrum (Replicator)', fontsize=9)
    ax11.set_xlabel('Frequency')
    ax11.set_ylabel('Power')

    # PCA projection
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.scatter(pca_proj_q[::5, 0], pca_proj_q[::5, 1],
                 c=np.arange(0, T, 5), cmap='viridis', s=0.5, alpha=0.5)
    ax12.set_title(f'PCA: Joint Strategy (Q-Learn)\n'
                   f'PC1={var_explained_q[0]:.1f}%, PC2={var_explained_q[1]:.1f}%',
                   fontsize=9)
    ax12.set_xlabel('PC1')
    ax12.set_ylabel('PC2')

    # KL divergence tracking
    ax13 = fig.add_subplot(gs[3, 3])
    kernel = np.ones(200) / 200
    kl_smooth = np.convolve(kl_A, kernel, mode='same')
    ax13.plot(kl_A, alpha=0.15, color='darkorange', linewidth=0.3)
    ax13.plot(kl_smooth, color='darkorange', linewidth=1.5,
              label='Player A')
    kl_B_smooth = np.convolve(kl_B, kernel, mode='same')
    ax13.plot(kl_B_smooth, color='teal', linewidth=1.5, label='Player B')
    ax13.set_title('KL Divergence from Uniform\n(Adaptive Rationality)', fontsize=9)
    ax13.set_xlabel('Time step')
    ax13.set_ylabel('KL(policy || uniform) bits')
    ax13.legend(fontsize=7)

    # ---- Row 5: Jacobian Eigenvalues + Surrogate Test + Summary ----
    ax14 = fig.add_subplot(gs[4, 0])
    eigs = results['jacobian_eigenvalues']
    ax14.scatter(np.real(eigs), np.imag(eigs), s=50, c='crimson',
                 edgecolors='black', zorder=5)
    theta = np.linspace(0, 2 * np.pi, 100)
    max_abs = max(np.max(np.abs(eigs)), 0.1)
    ax14.plot(max_abs * np.cos(theta), max_abs * np.sin(theta),
              'gray', linewidth=0.5, alpha=0.5)
    ax14.axhline(0, color='gray', linewidth=0.5)
    ax14.axvline(0, color='gray', linewidth=0.5)
    ax14.set_title('Jacobian Eigenvalues\n(Replicator at t=T/2)', fontsize=9)
    ax14.set_xlabel('Real part')
    ax14.set_ylabel('Imaginary part')
    ax14.set_aspect('equal')

    # Surrogate test histogram
    ax15 = fig.add_subplot(gs[4, 1])
    surr_range = surr_ents.max() - surr_ents.min()
    if surr_range < 1e-10:
        # All surrogates identical — plot bar chart instead
        ax15.bar([0], [len(surr_ents)], color='lightgray', edgecolor='gray', label='Surrogates')
        ax15.axvline(0, color='crimson', linewidth=2)
        ax15.text(0.5, 0.7, f'All surrogates = {surr_ents[0]:.3f}\nOriginal = {orig_ent:.3f}',
                  transform=ax15.transAxes, ha='center', fontsize=8)
    else:
        ax15.hist(surr_ents, bins=min(15, max(3, int(np.sqrt(len(surr_ents))))),
                  color='lightgray', edgecolor='gray', label='Surrogates')
        ax15.axvline(orig_ent, color='crimson', linewidth=2, label=f'Original = {orig_ent:.3f}')
    ax15.set_title(f'Surrogate Significance Test\np = {p_val:.4f}', fontsize=9)
    ax15.set_xlabel('Metric value')
    ax15.legend(fontsize=7)

    # Entropy rate stabilization detail
    ax16 = fig.add_subplot(gs[4, 2])
    # Compute entropy rate in sliding windows
    window_er = 200
    ent_rate = []
    for start in range(0, T - window_er, window_er // 4):
        end = start + window_er
        # Approximate entropy rate via conditional entropy
        transitions = {}
        counts = {}
        for t in range(start + 1, end):
            prev = act_A_q[t - 1]
            curr = act_A_q[t]
            counts[prev] = counts.get(prev, 0) + 1
            transitions[(prev, curr)] = transitions.get((prev, curr), 0) + 1

        h_rate = 0.0
        total = sum(counts.values())
        for (prev, curr), cnt in transitions.items():
            p_joint = cnt / total
            p_prev = counts[prev] / total
            p_cond = cnt / counts[prev]
            if p_cond > 1e-15:
                h_rate -= p_joint * np.log2(p_cond)
        ent_rate.append(h_rate)

    ax16.plot(np.linspace(0, T, len(ent_rate)), ent_rate, color='darkgreen', linewidth=1)
    if len(ent_rate) > 0:
        final_er = np.mean(ent_rate[-max(1, len(ent_rate)//4):])
        ax16.axhline(final_er, color='red', linestyle='--', alpha=0.7,
                     label=f'Stabilized ≈ {final_er:.3f} bits')
    ax16.set_title('Entropy Rate (Q-Learning)\n"Order within chaos"', fontsize=9)
    ax16.set_xlabel('Time step')
    ax16.set_ylabel('Entropy rate (bits/step)')
    ax16.legend(fontsize=7)

    # Text summary
    ax17 = fig.add_subplot(gs[4, 3])
    ax17.axis('off')
    summary_text = (
        f"Summary: {game_info['name']}\n"
        f"{'─' * 35}\n"
        f"Class: {game_info['class']}\n\n"
        f"Q-Learning Diagnostics:\n"
        f"  Entropy (stable): {results['entropy_q_mean']:.3f} ± {results['entropy_q_std']:.3f}\n"
        f"  Lyapunov λ_max: {lya_q:.4f}\n"
        f"  TE(A→B): {te_AB:.4f} bits\n"
        f"  TE(B→A): {te_BA:.4f} bits\n"
        f"  KL(π||uniform): {results['kl_mean']:.3f}\n\n"
        f"Replicator Diagnostics:\n"
        f"  Lyapunov λ_max: {lya_r:.4f}\n"
        f"  Entropy (stable): {results['entropy_r_mean']:.3f}\n\n"
        f"Jacobian: {'Imaginary parts present → cycling' if any(np.abs(np.imag(eigs)) > 1e-6) else 'Real eigenvalues → no cycling'}\n"
        f"Surrogate test: p = {p_val:.4f}\n"
        f"  {'Structure IS significant' if p_val < 0.05 else 'Cannot reject noise'}"
    )
    ax17.text(0.05, 0.95, summary_text, transform=ax17.transAxes,
              fontsize=8, fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(output_dir, f'{game_key}_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    results['traj_A_q'] = traj_A_q
    results['traj_B_q'] = traj_B_q
    results['traj_A_r'] = traj_A_r
    results['traj_B_r'] = traj_B_r
    results['ent_rate_final'] = final_er if len(ent_rate) > 0 else None

    return results


def create_cross_game_comparison(all_results, output_dir):
    """Generate comparison figures across game classes."""
    print("\n" + "=" * 70)
    print("  Cross-Game Comparison Analysis")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Cross-Game Comparison: Structured Instability Signatures',
                 fontsize=14, fontweight='bold')

    game_names = [r['name'] for r in all_results]
    game_classes = [r['class'] for r in all_results]
    class_colors = {'zero-sum': 'crimson', 'potential': 'forestgreen',
                    'general-sum': 'royalblue'}
    colors = [class_colors[c] for c in game_classes]

    # 1. Lyapunov exponents comparison
    ax = axes[0, 0]
    lya_q = [r['lyapunov_q'] for r in all_results]
    lya_r = [r['lyapunov_r'] for r in all_results]
    x_pos = np.arange(len(all_results))
    ax.bar(x_pos - 0.2, lya_q, 0.35, color=colors, alpha=0.7, label='Q-Learning',
           edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + 0.2, lya_r, 0.35, color=colors, alpha=0.4, label='Replicator',
           edgecolor='black', linewidth=0.5, hatch='//')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip()[:12] for n in game_names],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Lyapunov exponent')
    ax.set_title('Maximal Lyapunov Exponents')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=7)

    # 2. Entropy comparison
    ax = axes[0, 1]
    ent_means = [r['entropy_q_mean'] for r in all_results]
    ent_stds = [r['entropy_q_std'] for r in all_results]
    ax.bar(x_pos, ent_means, yerr=ent_stds, color=colors, alpha=0.7,
           edgecolor='black', linewidth=0.5, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip()[:12] for n in game_names],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean entropy (bits)')
    ax.set_title('Strategy Entropy (Q-Learning, 2nd half)')

    # 3. Transfer entropy comparison
    ax = axes[0, 2]
    te_ab = [r['transfer_entropy_AB'] for r in all_results]
    te_ba = [r['transfer_entropy_BA'] for r in all_results]
    ax.bar(x_pos - 0.2, te_ab, 0.35, color='navy', alpha=0.7, label='TE(A→B)')
    ax.bar(x_pos + 0.2, te_ba, 0.35, color='firebrick', alpha=0.7, label='TE(B→A)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip()[:12] for n in game_names],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Transfer entropy (bits)')
    ax.set_title('Transfer Entropy (Opponent Modeling)')
    ax.legend(fontsize=7)

    # 4. KL divergence comparison
    ax = axes[1, 0]
    kl_vals = [r['kl_mean'] for r in all_results]
    ax.bar(x_pos, kl_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip()[:12] for n in game_names],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('KL(policy || uniform) bits')
    ax.set_title('Adaptive Rationality (KL from uniform)')

    # 5. Entropy rate
    ax = axes[1, 1]
    er_vals = [r.get('ent_rate_final', 0) or 0 for r in all_results]
    ax.bar(x_pos, er_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.split('(')[0].strip()[:12] for n in game_names],
                       rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Entropy rate (bits/step)')
    ax.set_title('Stabilized Entropy Rate')

    # 6. Classification summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = "Classification of Dynamical Signatures\n" + "─" * 40 + "\n\n"
    for r in all_results:
        lya = r['lyapunov_q']
        ent = r['entropy_q_mean']
        te = max(r['transfer_entropy_AB'], r['transfer_entropy_BA'])
        n_act = r['traj_A_q'].shape[1]
        max_ent = np.log2(n_act)
        ent_ratio = ent / max_ent if max_ent > 0 else 0

        if lya > 0.01:
            dyn_type = "CHAOTIC (structured)"
        elif lya > -0.005:
            dyn_type = "LIMIT CYCLE / neutral"
        else:
            dyn_type = "CONVERGENT"

        if ent_ratio > 0.8:
            ent_type = "high (exploratory)"
        elif ent_ratio > 0.3:
            ent_type = "intermediate (adaptive)"
        else:
            ent_type = "low (exploitative)"

        summary += f"  {r['name'][:25]:25s}\n"
        summary += f"    Dynamics: {dyn_type}\n"
        summary += f"    Entropy:  {ent_type} ({ent:.2f}/{max_ent:.2f})\n"
        summary += f"    Coupling: TE={te:.3f}\n\n"

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=7,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(os.path.join(output_dir, 'cross_game_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Legend figure for game classes ---
    fig_leg, ax_leg = plt.subplots(figsize=(8, 1))
    ax_leg.axis('off')
    for i, (cls, col) in enumerate(class_colors.items()):
        ax_leg.add_patch(plt.Rectangle((0.1 + i * 0.3, 0.3), 0.05, 0.4,
                                        color=col, alpha=0.7))
        ax_leg.text(0.17 + i * 0.3, 0.5, cls, fontsize=10, va='center')
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    fig_leg.savefig(os.path.join(output_dir, 'legend.png'), dpi=100, bbox_inches='tight')
    plt.close()


def create_formal_definitions_table(output_dir):
    """Generate a figure with formal definitions required by reviewer."""
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis('off')

    text = r"""
FORMAL DEFINITIONS AND ASSUMPTIONS
═══════════════════════════════════════════════════════════════════════════

Definition 1 (Structured Instability / Structured Chaos)
─────────────────────────────────────────────────────────
A learning dynamics {x_t} in a repeated game G exhibits structured instability if:
  (i)   Non-convergence: lim inf ||x_{t+1} - x_t|| > 0  (the trajectory does not settle)
  (ii)  Boundedness: ∃ compact set K ⊂ Δ^n such that x_t ∈ K for all t ≥ t_0
  (iii) Positive finite complexity: 0 < h_μ < H_max, where h_μ is the entropy rate
        of the discretized trajectory, and H_max = log_2(|A|) is the maximum entropy.
  (iv)  Sensitivity: The maximal Lyapunov exponent satisfies λ_max > 0 (bounded).
  (v)   Coupling: Time-averaged mutual information I(X;Y) > 0 between agents.

Conditions (i)-(ii) distinguish structured instability from convergence AND from divergence.
Condition (iii) distinguishes it from pure noise (h_μ ≈ H_max) and periodic orbits (h_μ ≈ 0).
Conditions (iv)-(v) ensure the dynamics are both sensitive and inter-agent coupled.

Definition 2 (Conditions for Boundedness)
─────────────────────────────────────────
The learning dynamics remain bounded if the following assumptions hold:

  A1 (Bounded payoffs): ∃ M > 0 such that |u_i(a)| ≤ M for all players i and action profiles a.
  A2 (Simplex constraint): Strategy updates preserve the probability simplex:
      x_i(t) ∈ Δ^{|A_i|} for all t, where Δ^k = {p ∈ R^k : p_j ≥ 0, Σ p_j = 1}.
  A3 (Decaying or bounded step-sizes): For Q-learning, α_t ∈ (0,1] with Σ α_t = ∞,
      Σ α_t^2 < ∞ (Robbins-Monro conditions) OR α_t = α ∈ (0,1) (constant step-size
      with bounded Q-values: |Q(s,a)| ≤ M/(1-γ) by the contraction argument).
  A4 (Boltzmann exploration): Temperature τ > 0 ensures policies are interior to the simplex,
      preventing boundary singularities. If τ → 0, boundedness still holds on int(Δ).
  A5 (Lipschitz dynamics): The replicator map F(x) = x_i[(Ax)_i - x^T Ax] is Lipschitz
      on Δ, since payoffs are bounded and x ∈ Δ is compact. This ensures short-time existence
      and uniqueness of solutions (Picard-Lindelöf theorem), with the simplex as an invariant set.

Proposition (Existence of Compact Attractor)
────────────────────────────────────────────
Under A1-A5, the joint strategy trajectory {(x_t, y_t)} remains in Δ^{|A|} × Δ^{|B|} for all t,
which is compact. By the Birkhoff theorem, the ω-limit set ω({x_t}) is nonempty, compact,
and invariant. When ω({x_t}) is not a fixed point, it is either a limit cycle, quasi-periodic
orbit, or a strange attractor — i.e., structured instability.

Justification: Bounded payoffs (A1) bound the drift; simplex constraints (A2) confine trajectories;
step-size conditions (A3) prevent Q-value explosion; Boltzmann policies (A4) keep strategies
interior; Lipschitz continuity (A5) prevents finite-time blowup. Together these guarantee that
non-convergent trajectories remain bounded, making structured instability the only alternative
to convergence.
"""

    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8.5,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='ivory', alpha=0.9))

    plt.savefig(os.path.join(output_dir, 'formal_definitions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# SECTION 5: MAIN EXECUTION
# ============================================================================

def main():
    output_dir = '/home/claude/simulation_results'
    os.makedirs(output_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Structured Instability Simulation Suite                        ║")
    print("║  Companion to: Instability as Insight (Mahadevaiah)             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Generate formal definitions figure
    print("\nGenerating formal definitions and assumptions...")
    create_formal_definitions_table(output_dir)

    # Run all game analyses
    all_results = []
    for game_key, game_info in GAMES.items():
        result = analyze_game(game_key, game_info, output_dir, T=10000)
        all_results.append(result)

    # Cross-game comparison
    create_cross_game_comparison(all_results, output_dir)

    # Print final summary table
    print("\n" + "=" * 90)
    print("FINAL SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Game':<25} {'Class':<14} {'λ_Q':>8} {'λ_R':>8} {'H(Q)':>8} {'TE(A→B)':>8} {'TE(B→A)':>8} {'p-val':>7}"
    print(header)
    print("-" * 90)
    for r in all_results:
        line = (f"{r['name'][:24]:<25} {r['class']:<14} "
                f"{r['lyapunov_q']:>8.4f} {r['lyapunov_r']:>8.4f} "
                f"{r['entropy_q_mean']:>8.3f} "
                f"{r['transfer_entropy_AB']:>8.4f} {r['transfer_entropy_BA']:>8.4f} "
                f"{r['surrogate_p_value']:>7.4f}")
        print(line)
    print("=" * 90)

    print(f"\nAll figures saved to: {output_dir}/")
    print("Files generated:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")

    return all_results


if __name__ == '__main__':
    results = main()
