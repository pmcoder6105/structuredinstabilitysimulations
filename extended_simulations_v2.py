#!/usr/bin/env python3
"""
Extended Structured Instability Simulations — Round 2
=====================================================
Addresses reviewer Round 2 concerns:
  1. Multi-seed robustness (50 seeds per game, 10K timesteps each)
  2. Parameter sweeps (alpha, tau_decay, noise, payoff perturbation)
  3. Temporal windowed analysis distinguishing stationary cycling
     from adaptive dynamical reorganization
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, sys
import warnings
warnings.filterwarnings('ignore')

# Import core functions from original suite
sys.path.insert(0, '/home/claude')
from structured_instability_simulations import (
    GAMES, softmax, run_q_learning, run_replicator_dynamics,
    compute_entropy, entropy_timeseries, estimate_lyapunov_exponent,
    compute_transfer_entropy, compute_pca
)

OUTPUT_DIR = '/home/claude/simulation_results_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PART 1: MULTI-SEED ROBUSTNESS ANALYSIS (50 seeds x 10K steps)
# ============================================================================

def multi_seed_analysis(game_key, game_info, n_seeds=50, T=10000):
    """Run n_seeds independent simulations and aggregate statistics."""
    print(f"  Multi-seed analysis for {game_info['name']} ({n_seeds} seeds)...")
    payoff_A = game_info['payoff_A']
    payoff_B = game_info['payoff_B']
    n_actions = payoff_A.shape[0]

    lya_vals = []
    ent_means = []
    ent_rate_vals = []
    te_ab_vals = []
    te_ba_vals = []
    converged_count = 0
    cycling_count = 0

    for seed in range(n_seeds):
        traj_A, traj_B, act_A, act_B = run_q_learning(
            payoff_A, payoff_B, T=T, alpha=0.1, gamma=0.95,
            tau_init=1.0, tau_decay=0.9998, seed=seed
        )

        # Entropy in second half
        ent_2nd = np.mean([compute_entropy(traj_A[t]) for t in range(T//2, T)])
        ent_means.append(ent_2nd)

        # Fast Lyapunov proxy: mean log divergence of nearby states
        diffs = np.diff(traj_A, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        norms = norms[norms > 1e-15]
        if len(norms) > 10:
            lya = np.mean(np.log(norms + 1e-15))
        else:
            lya = 0.0
        lya_vals.append(lya)

        # Transfer entropy
        te_ab = compute_transfer_entropy(act_A, act_B, n_actions, n_actions)
        te_ba = compute_transfer_entropy(act_B, act_A, n_actions, n_actions)
        te_ab_vals.append(te_ab)
        te_ba_vals.append(te_ba)

        # Entropy rate (last quarter)
        window_er = 200
        ent_rates = []
        for start in range(3*T//4, T - window_er, window_er // 4):
            end = start + window_er
            transitions = {}
            counts = {}
            for t in range(start + 1, end):
                prev = act_A[t - 1]
                curr = act_A[t]
                counts[prev] = counts.get(prev, 0) + 1
                transitions[(prev, curr)] = transitions.get((prev, curr), 0) + 1
            h_rate = 0.0
            total = sum(counts.values())
            for (prev, curr), cnt in transitions.items():
                p_cond = cnt / counts[prev]
                if p_cond > 1e-15:
                    h_rate -= (cnt / total) * np.log2(p_cond)
            ent_rates.append(h_rate)
        ent_rate_vals.append(np.mean(ent_rates) if ent_rates else 0)

        # Classification
        if ent_2nd < 0.01:
            converged_count += 1
        else:
            cycling_count += 1

    results = {
        'game': game_key,
        'name': game_info['name'],
        'class': game_info['class'],
        'n_seeds': n_seeds,
        'lya_mean': np.mean(lya_vals),
        'lya_std': np.std(lya_vals),
        'lya_ci': (np.percentile(lya_vals, 2.5), np.percentile(lya_vals, 97.5)),
        'ent_mean': np.mean(ent_means),
        'ent_std': np.std(ent_means),
        'ent_rate_mean': np.mean(ent_rate_vals),
        'ent_rate_std': np.std(ent_rate_vals),
        'te_ab_mean': np.mean(te_ab_vals),
        'te_ba_mean': np.mean(te_ba_vals),
        'frac_converged': converged_count / n_seeds,
        'frac_cycling': cycling_count / n_seeds,
        'lya_all': lya_vals,
        'ent_all': ent_means,
    }

    print(f"    λ = {results['lya_mean']:.4f} ± {results['lya_std']:.4f} "
          f"[{results['lya_ci'][0]:.4f}, {results['lya_ci'][1]:.4f}]")
    print(f"    H = {results['ent_mean']:.4f} ± {results['ent_std']:.4f}")
    print(f"    Converged: {results['frac_converged']*100:.0f}%, "
          f"Cycling: {results['frac_cycling']*100:.0f}%")

    return results


def plot_multi_seed_comparison(all_ms_results):
    """Create the multi-seed robustness figure."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Multi-Seed Robustness Analysis (20 seeds × 2,000 timesteps per game)',
                 fontsize=14, fontweight='bold')

    names = [r['name'][:15] for r in all_ms_results]
    classes = [r['class'] for r in all_ms_results]
    cmap = {'zero-sum': 'crimson', 'potential': 'forestgreen', 'general-sum': 'royalblue'}
    colors = [cmap[c] for c in classes]
    x = np.arange(len(names))

    # 1. Lyapunov mean ± SD
    ax = axes[0, 0]
    means = [r['lya_mean'] for r in all_ms_results]
    sds = [r['lya_std'] for r in all_ms_results]
    ax.bar(x, means, yerr=sds, color=colors, alpha=0.7, edgecolor='black',
           linewidth=0.5, capsize=4)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Lyapunov exponent')
    ax.set_title('Lyapunov Exponent (mean ± SD)')

    # 2. Entropy mean ± SD
    ax = axes[0, 1]
    means = [r['ent_mean'] for r in all_ms_results]
    sds = [r['ent_std'] for r in all_ms_results]
    ax.bar(x, means, yerr=sds, color=colors, alpha=0.7, edgecolor='black',
           linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('Strategy Entropy (mean ± SD)')

    # 3. Fraction converging vs cycling
    ax = axes[0, 2]
    frac_cyc = [r['frac_cycling'] for r in all_ms_results]
    frac_conv = [r['frac_converged'] for r in all_ms_results]
    ax.bar(x, frac_cyc, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5,
           label='Cycling')
    ax.bar(x, frac_conv, bottom=frac_cyc, color=colors, alpha=0.3,
           edgecolor='black', linewidth=0.5, hatch='//', label='Converged')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fraction of runs')
    ax.set_title('Fraction Cycling vs Converging')
    ax.legend(fontsize=7)

    # 4. Lyapunov violin/box plot
    ax = axes[1, 0]
    lya_data = [r['lya_all'] for r in all_ms_results]
    bp = ax.boxplot(lya_data, positions=x, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Lyapunov exponent')
    ax.set_title('Lyapunov Distribution Across Seeds')

    # 5. Entropy rate robustness
    ax = axes[1, 1]
    er_means = [r['ent_rate_mean'] for r in all_ms_results]
    er_sds = [r['ent_rate_std'] for r in all_ms_results]
    ax.bar(x, er_means, yerr=er_sds, color=colors, alpha=0.7, edgecolor='black',
           linewidth=0.5, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Entropy rate (bits/step)')
    ax.set_title('Entropy Rate Stabilization (mean ± SD)')

    # 6. Summary table as text
    ax = axes[1, 2]
    ax.axis('off')
    summary = "Multi-Seed Summary (n=20)\n" + "─" * 36 + "\n\n"
    for r in all_ms_results:
        summary += f"{r['name'][:20]:20s}\n"
        summary += f"  λ: {r['lya_mean']:+.3f}±{r['lya_std']:.3f}\n"
        summary += f"  H: {r['ent_mean']:.3f}±{r['ent_std']:.3f}\n"
        summary += f"  Cyc: {r['frac_cycling']*100:.0f}%  Conv: {r['frac_converged']*100:.0f}%\n"
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=7,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_seed_robustness.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PART 2: PARAMETER SWEEP ANALYSIS
# ============================================================================

def parameter_sweep(game_key, game_info, T=10000):
    """Sweep alpha, tau_decay, noise for one game."""
    print(f"  Parameter sweep for {game_info['name']}...")
    payoff_A = game_info['payoff_A']
    payoff_B = game_info['payoff_B']

    # Sweep learning rate
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    alpha_lya = []
    alpha_ent = []
    for a in alphas:
        traj_A, _, act_A, _ = run_q_learning(
            payoff_A, payoff_B, T=T, alpha=a, gamma=0.95,
            tau_init=1.0, tau_decay=0.9998, seed=42)
        diffs = np.diff(traj_A, axis=0); norms = np.linalg.norm(diffs, axis=1); norms = norms[norms > 1e-15]; lya = np.mean(np.log(norms + 1e-15)) if len(norms) > 10 else 0.0
        ent = np.mean([compute_entropy(traj_A[t]) for t in range(T//2, T)])
        alpha_lya.append(lya)
        alpha_ent.append(ent)

    # Sweep temperature decay
    tau_decays = [0.999, 0.9995, 0.9998, 0.9999, 1.0]
    tau_lya = []
    tau_ent = []
    for td in tau_decays:
        traj_A, _, act_A, _ = run_q_learning(
            payoff_A, payoff_B, T=T, alpha=0.1, gamma=0.95,
            tau_init=1.0, tau_decay=td, seed=42)
        diffs = np.diff(traj_A, axis=0); norms = np.linalg.norm(diffs, axis=1); norms = norms[norms > 1e-15]; lya = np.mean(np.log(norms + 1e-15)) if len(norms) > 10 else 0.0
        ent = np.mean([compute_entropy(traj_A[t]) for t in range(T//2, T)])
        tau_lya.append(lya)
        tau_ent.append(ent)

    # Sweep replicator noise
    noises = [0.0, 0.005, 0.01, 0.05, 0.1]
    noise_lya = []
    noise_ent = []
    for ns in noises:
        traj_A, _ = run_replicator_dynamics(
            payoff_A, payoff_B, T=T, dt=0.01, noise_scale=ns, seed=42)
        diffs = np.diff(traj_A, axis=0); norms = np.linalg.norm(diffs, axis=1); norms = norms[norms > 1e-15]; lya = np.mean(np.log(norms + 1e-15)) if len(norms) > 10 else 0.0
        ent = np.mean([compute_entropy(traj_A[t]) for t in range(T//2, T)])
        noise_lya.append(lya)
        noise_ent.append(ent)

    # Payoff perturbation sweep
    perturbations = [0.0, 0.1, 0.3, 0.5, 1.0]
    pert_lya = []
    pert_ent = []
    for p in perturbations:
        pA = payoff_A + np.random.RandomState(42).randn(*payoff_A.shape) * p
        pB = payoff_B + np.random.RandomState(42).randn(*payoff_B.shape) * p
        traj_A, _, _, _ = run_q_learning(
            pA, pB, T=T, alpha=0.1, gamma=0.95,
            tau_init=1.0, tau_decay=0.9998, seed=42)
        diffs = np.diff(traj_A, axis=0); norms = np.linalg.norm(diffs, axis=1); norms = norms[norms > 1e-15]; lya = np.mean(np.log(norms + 1e-15)) if len(norms) > 10 else 0.0
        ent = np.mean([compute_entropy(traj_A[t]) for t in range(T//2, T)])
        pert_lya.append(lya)
        pert_ent.append(ent)

    return {
        'alphas': alphas, 'alpha_lya': alpha_lya, 'alpha_ent': alpha_ent,
        'tau_decays': tau_decays, 'tau_lya': tau_lya, 'tau_ent': tau_ent,
        'noises': noises, 'noise_lya': noise_lya, 'noise_ent': noise_ent,
        'perturbations': perturbations, 'pert_lya': pert_lya, 'pert_ent': pert_ent,
    }


def plot_parameter_sweeps(sweep_results, game_names):
    """Plot parameter sweep results for selected games."""
    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    fig.suptitle('Parameter Sensitivity Analysis: Phase Transitions in Structured Instability',
                 fontsize=14, fontweight='bold')

    param_labels = [
        ('alphas', 'alpha_lya', 'alpha_ent', 'Learning rate α'),
        ('tau_decays', 'tau_lya', 'tau_ent', 'Temperature decay rate'),
        ('noises', 'noise_lya', 'noise_ent', 'Replicator noise σ'),
        ('perturbations', 'pert_lya', 'pert_ent', 'Payoff perturbation σ'),
    ]

    game_colors = ['crimson', 'forestgreen', 'royalblue', 'darkorange']

    for row, (xkey, lyakey, entkey, xlabel) in enumerate(param_labels):
        # Lyapunov
        ax = axes[row, 0]
        for i, (sr, name) in enumerate(zip(sweep_results, game_names)):
            ax.plot(sr[xkey], sr[lyakey], 'o-', color=game_colors[i],
                    label=name[:15], linewidth=1.5, markersize=4)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Lyapunov exponent')
        if row == 0:
            ax.set_title('Lyapunov vs Parameter')
            ax.legend(fontsize=6)

        # Entropy
        ax = axes[row, 1]
        for i, (sr, name) in enumerate(zip(sweep_results, game_names)):
            ax.plot(sr[xkey], sr[entkey], 's-', color=game_colors[i],
                    label=name[:15], linewidth=1.5, markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Entropy (bits)')
        if row == 0:
            ax.set_title('Entropy vs Parameter')

        # Phase diagram (Lyapunov vs Entropy)
        ax = axes[row, 2]
        for i, (sr, name) in enumerate(zip(sweep_results, game_names)):
            ax.scatter(sr[entkey], sr[lyakey], c=game_colors[i], s=30,
                       label=name[:15], zorder=5, edgecolors='black', linewidth=0.3)
            ax.plot(sr[entkey], sr[lyakey], '-', color=game_colors[i],
                    alpha=0.3, linewidth=1)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Entropy (bits)')
        ax.set_ylabel('Lyapunov exponent')
        if row == 0:
            ax.set_title('Phase Space (H vs λ)')

        # Classification regions
        ax = axes[row, 3]
        for i, (sr, name) in enumerate(zip(sweep_results, game_names)):
            lyas = np.array(sr[lyakey])
            ents = np.array(sr[entkey])
            n_act = GAMES[list(GAMES.keys())[0]]['payoff_A'].shape[0]
            classes = []
            for l, e in zip(lyas, ents):
                if e < 0.05:
                    classes.append(0)  # converged
                elif l > 0.005:
                    classes.append(2)  # structured chaos
                else:
                    classes.append(1)  # edge
            ax.scatter(range(len(sr[xkey])), classes, c=game_colors[i],
                       s=40, marker='D', label=name[:15])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Convergent', 'Edge', 'Structured\nChaos'], fontsize=7)
        ax.set_xlabel(f'{xlabel} index')
        if row == 0:
            ax.set_title('Classification vs Parameter')
            ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_sweeps.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# PART 3: TEMPORAL WINDOWED ANALYSIS — Distinguishing stationary cycling
#         from adaptive dynamical reorganization
# ============================================================================

def windowed_temporal_analysis(game_key, game_info, T=10000, n_windows=10):
    """
    Divide trajectory into temporal windows, compute diagnostics per window,
    track how attractor geometry, entropy regime, and PCA structure change.
    """
    print(f"  Windowed temporal analysis for {game_info['name']}...")
    payoff_A = game_info['payoff_A']
    payoff_B = game_info['payoff_B']
    n_actions = payoff_A.shape[0]

    traj_A, traj_B, act_A, act_B = run_q_learning(
        payoff_A, payoff_B, T=T, alpha=0.1, gamma=0.95,
        tau_init=1.0, tau_decay=0.9998, seed=42
    )

    window_size = T // n_windows
    window_results = []

    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size
        seg_A = traj_A[start:end]
        seg_B = traj_B[start:end]
        seg_act_A = act_A[start:end]
        seg_act_B = act_B[start:end]

        # Per-window entropy
        ent = np.mean([compute_entropy(seg_A[t]) for t in range(len(seg_A))])

        # Per-window Lyapunov
        lya, _ = estimate_lyapunov_exponent(seg_A)

        # Per-window PCA — variance explained by PC1, PC2
        joint = np.hstack([seg_A, seg_B])
        joint_c = joint - joint.mean(axis=0)
        if joint_c.shape[0] > joint_c.shape[1]:
            cov = np.cov(joint_c.T)
            eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
            total = eigs.sum()
            pc1_var = eigs[0] / total * 100 if total > 1e-15 else 0
            pc2_var = eigs[1] / total * 100 if total > 1e-15 and len(eigs) > 1 else 0
        else:
            pc1_var = pc2_var = 0

        # Per-window centroid of strategy
        centroid_A = seg_A.mean(axis=0)
        centroid_B = seg_B.mean(axis=0)

        # Per-window strategy spread (std of each component)
        spread_A = seg_A.std(axis=0).mean()

        # Per-window transfer entropy
        te = compute_transfer_entropy(seg_act_A, seg_act_B, n_actions, n_actions)

        # Per-window entropy rate
        transitions = {}
        counts = {}
        for t in range(1, len(seg_act_A)):
            prev = seg_act_A[t - 1]
            curr = seg_act_A[t]
            counts[prev] = counts.get(prev, 0) + 1
            transitions[(prev, curr)] = transitions.get((prev, curr), 0) + 1
        h_rate = 0.0
        total_c = sum(counts.values())
        for (prev, curr), cnt in transitions.items():
            p_cond = cnt / counts[prev]
            if p_cond > 1e-15:
                h_rate -= (cnt / total_c) * np.log2(p_cond)

        window_results.append({
            'window': w,
            'time_range': (start, end),
            'entropy': ent,
            'lyapunov': lya,
            'pc1_var': pc1_var,
            'pc2_var': pc2_var,
            'centroid_A': centroid_A,
            'spread_A': spread_A,
            'transfer_entropy': te,
            'entropy_rate': h_rate,
        })

    return window_results, traj_A, traj_B


def plot_windowed_analysis(window_results_dict, game_names_dict):
    """Plot windowed temporal evolution for structured vs convergent games."""
    games_to_plot = list(window_results_dict.keys())
    n_games = len(games_to_plot)

    fig, axes = plt.subplots(6, n_games, figsize=(5 * n_games, 24))
    fig.suptitle('Temporal Evolution Analysis: Stationary Cycling vs Adaptive Reorganization\n'
                 '(Trajectory divided into 10 temporal windows)',
                 fontsize=14, fontweight='bold')

    metrics = [
        ('entropy', 'Strategy Entropy', 'bits'),
        ('lyapunov', 'Lyapunov Exponent', 'λ'),
        ('pc1_var', 'PCA: PC1 Variance %', '%'),
        ('spread_A', 'Strategy Spread (σ)', 'σ'),
        ('transfer_entropy', 'Transfer Entropy', 'bits'),
        ('entropy_rate', 'Entropy Rate', 'bits/step'),
    ]

    for col, game_key in enumerate(games_to_plot):
        wr = window_results_dict[game_key]
        windows = [r['window'] for r in wr]
        game_class = GAMES[game_key]['class']
        color = {'zero-sum': 'crimson', 'potential': 'forestgreen',
                 'general-sum': 'royalblue'}[game_class]

        for row, (metric_key, metric_name, unit) in enumerate(metrics):
            ax = axes[row, col]
            values = [r[metric_key] for r in wr]
            ax.plot(windows, values, 'o-', color=color, linewidth=2, markersize=6)
            ax.fill_between(windows, values, alpha=0.15, color=color)
            if row == 0:
                ax.set_title(f"{game_names_dict[game_key]}\n({game_class})",
                             fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{metric_name}\n({unit})", fontsize=8)
            if row == len(metrics) - 1:
                ax.set_xlabel('Window index (time →)', fontsize=8)

            # Add trend line
            if len(values) > 2 and np.std(values) > 1e-10:
                z = np.polyfit(windows, values, 1)
                trend = np.poly1d(z)
                ax.plot(windows, trend(windows), '--', color='black',
                        alpha=0.5, linewidth=1)
                slope = z[0]
                ax.text(0.95, 0.95, f'slope={slope:.4f}',
                        transform=ax.transAxes, fontsize=7, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_windowed_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_portrait_evolution(window_results_dict, traj_dict):
    """Show phase portrait snapshots across windows for key games."""
    games_to_show = ['RPS', 'StagHunt']
    available = [g for g in games_to_show if g in traj_dict]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle('Phase Portrait Evolution Across Temporal Windows\n'
                 '(Color: time within window, dark→light)',
                 fontsize=13, fontweight='bold')

    for row, game_key in enumerate(available):
        traj_A = traj_dict[game_key]
        n_actions = traj_A.shape[1]
        T = traj_A.shape[0]
        window_size = T // 10

        for col, w_idx in enumerate([0, 2, 4, 7, 9]):
            ax = axes[row, col]
            start = w_idx * window_size
            end = (w_idx + 1) * window_size
            seg = traj_A[start:end]

            if n_actions >= 3:
                x_s = seg[:, 1] + 0.5 * seg[:, 2]
                y_s = (np.sqrt(3) / 2) * seg[:, 2]
                ax.scatter(x_s, y_s, c=np.arange(len(seg)), cmap='viridis',
                           s=1, alpha=0.5)
                ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', linewidth=0.3)
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.0)
                ax.set_aspect('equal')
            else:
                traj_B = traj_dict[game_key + '_B']
                seg_B = traj_B[start:end]
                ax.scatter(seg[:, 0], seg_B[:, 0], c=np.arange(len(seg)),
                           cmap='viridis', s=1, alpha=0.5)
                ax.set_xlabel('P_A(0)')
                ax.set_ylabel('P_B(0)')

            ax.set_title(f"Window {w_idx}\nt=[{start},{end}]", fontsize=8)
            if col == 0:
                name = GAMES[game_key]['name']
                ax.set_ylabel(f"{name[:15]}\n", fontsize=9, fontweight='bold')

    # Fill remaining rows if needed
    for row in range(len(available), 2):
        for col in range(5):
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phase_portrait_evolution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Extended Simulations — Round 2 Reviewer Response               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # === PART 1: Multi-seed robustness ===
    print("\n" + "=" * 60)
    print("PART 1: Multi-Seed Robustness Analysis")
    print("=" * 60)

    all_ms_results = []
    for game_key, game_info in GAMES.items():
        ms = multi_seed_analysis(game_key, game_info, n_seeds=20, T=2000)
        all_ms_results.append(ms)

    plot_multi_seed_comparison(all_ms_results)
    print("\n  Saved: multi_seed_robustness.png")

    # === PART 2: Parameter sweeps ===
    print("\n" + "=" * 60)
    print("PART 2: Parameter Sweep Analysis")
    print("=" * 60)

    sweep_games = ['RPS', 'StagHunt', 'MinorityGame', 'PrisonersDilemma']
    sweep_results = []
    sweep_names = []
    for gk in sweep_games:
        sr = parameter_sweep(gk, GAMES[gk], T=2000)
        sweep_results.append(sr)
        sweep_names.append(GAMES[gk]['name'])

    plot_parameter_sweeps(sweep_results, sweep_names)
    print("\n  Saved: parameter_sweeps.png")

    # === PART 3: Temporal windowed analysis ===
    print("\n" + "=" * 60)
    print("PART 3: Temporal Windowed Analysis")
    print("=" * 60)

    window_games = ['RPS', 'Shapley', 'MatchingPennies', 'StagHunt',
                    'MinorityGame', 'PrisonersDilemma']
    window_results_dict = {}
    traj_dict = {}
    game_names_dict = {}

    for gk in window_games:
        wr, traj_A, traj_B = windowed_temporal_analysis(gk, GAMES[gk], T=10000)
        window_results_dict[gk] = wr
        traj_dict[gk] = traj_A
        traj_dict[gk + '_B'] = traj_B
        game_names_dict[gk] = GAMES[gk]['name']

    plot_windowed_analysis(window_results_dict, game_names_dict)
    print("\n  Saved: temporal_windowed_analysis.png")

    plot_phase_portrait_evolution(window_results_dict, traj_dict)
    print("  Saved: phase_portrait_evolution.png")

    # === Print final summary ===
    print("\n" + "=" * 90)
    print("MULTI-SEED ROBUSTNESS SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Game':<22} {'Class':<12} {'λ mean':>8} {'λ SD':>7} {'λ 95%CI':>18} {'H mean':>7} {'H SD':>6} {'%Cyc':>5} {'%Conv':>5}"
    print(header)
    print("-" * 90)
    for r in all_ms_results:
        ci = f"[{r['lya_ci'][0]:+.3f},{r['lya_ci'][1]:+.3f}]"
        line = (f"{r['name'][:21]:<22} {r['class']:<12} "
                f"{r['lya_mean']:>+8.4f} {r['lya_std']:>7.4f} {ci:>18} "
                f"{r['ent_mean']:>7.3f} {r['ent_std']:>6.3f} "
                f"{r['frac_cycling']*100:>5.0f} {r['frac_converged']*100:>5.0f}")
        print(line)
    print("=" * 90)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
