#!/usr/bin/env python3
"""
Cross-Game Transfer Experiments — Round 3
==========================================
Tests whether adversarial topology leaves reusable dynamical structure.

Four experiments:
  Case A: RPS (adversarial) → Prisoner's Dilemma (convergent)
  Case B: Prisoner's Dilemma → RPS
  Case C: RPS → Stag Hunt (coordination)
  Case D: Stag Hunt → RPS

Each compared against fresh (naive) initialization as control.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/claude')
from structured_instability_simulations import GAMES, softmax, compute_entropy

OUTPUT_DIR = '/home/claude/simulation_results_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_q_learning_with_state(payoff_A, payoff_B, T, alpha=0.1, gamma=0.95,
                               tau_init=1.0, tau_decay=0.9998,
                               Q_A_init=None, Q_B_init=None, tau_start=None,
                               seed=None):
    """
    Q-learning that accepts and returns internal state (Q-values, tau).
    If Q_A_init/Q_B_init provided, resumes from that state.
    """
    if seed is not None:
        np.random.seed(seed)

    n_A = payoff_A.shape[0]
    n_B = payoff_A.shape[1]

    # Initialize or resume
    if Q_A_init is not None:
        # Handle dimension mismatch when switching games
        if len(Q_A_init) == n_A:
            Q_A = Q_A_init.copy()
        else:
            # Pad or truncate, preserving what we can
            Q_A = np.zeros(n_A)
            shared = min(len(Q_A_init), n_A)
            Q_A[:shared] = Q_A_init[:shared]
    else:
        Q_A = np.zeros(n_A)

    if Q_B_init is not None:
        if len(Q_B_init) == n_B:
            Q_B = Q_B_init.copy()
        else:
            Q_B = np.zeros(n_B)
            shared = min(len(Q_B_init), n_B)
            Q_B[:shared] = Q_B_init[:shared]
    else:
        Q_B = np.zeros(n_B)

    tau = tau_start if tau_start is not None else tau_init

    traj_A = np.zeros((T, n_A))
    traj_B = np.zeros((T, n_B))
    actions_A = np.zeros(T, dtype=int)
    actions_B = np.zeros(T, dtype=int)

    for t in range(T):
        pi_A = softmax(Q_A, tau)
        pi_B = softmax(Q_B, tau)
        traj_A[t] = pi_A
        traj_B[t] = pi_B

        a = np.random.choice(n_A, p=pi_A)
        b = np.random.choice(n_B, p=pi_B)
        actions_A[t] = a
        actions_B[t] = b

        r_A = payoff_A[a, b]
        r_B = payoff_B[a, b]

        Q_A[a] = Q_A[a] + alpha * (r_A + gamma * np.max(Q_A) - Q_A[a])
        Q_B[b] = Q_B[b] + alpha * (r_B + gamma * np.max(Q_B) - Q_B[b])

        tau *= tau_decay

    return traj_A, traj_B, actions_A, actions_B, Q_A, Q_B, tau


def compute_windowed_metrics(traj_A, traj_B, actions_A, actions_B, n_windows=20):
    """Compute per-window entropy, spread, PCA PC1%."""
    T = len(traj_A)
    ws = T // n_windows
    results = []
    for w in range(n_windows):
        s, e = w * ws, (w + 1) * ws
        seg_A = traj_A[s:e]
        seg_B = traj_B[s:e]

        ent = np.mean([compute_entropy(seg_A[t]) for t in range(len(seg_A))])
        spread = seg_A.std(axis=0).mean()

        joint = np.hstack([seg_A, seg_B])
        joint_c = joint - joint.mean(axis=0)
        if joint_c.shape[0] > joint_c.shape[1] and joint_c.shape[1] > 0:
            cov = np.cov(joint_c.T)
            eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
            total = eigs.sum()
            pc1 = eigs[0] / total * 100 if total > 1e-15 else 0
        else:
            pc1 = 0

        # Transfer entropy (simplified)
        seg_act_A = actions_A[s:e]
        seg_act_B = actions_B[s:e]
        n_act = traj_A.shape[1]
        te = 0.0
        if len(seg_act_A) > 10:
            from structured_instability_simulations import compute_transfer_entropy
            te = compute_transfer_entropy(seg_act_A, seg_act_B, n_act, n_act)

        results.append({
            'window': w, 'entropy': ent, 'spread': spread,
            'pc1_pct': pc1, 'transfer_entropy': te,
        })
    return results


def run_cross_switch_experiment(game1_key, game2_key, T_phase1=2000,
                                 T_phase2=3000, seed=42):
    """
    Run a cross-switch experiment:
      Phase 1: Train on game1 for T_phase1 steps
      Phase 2: Switch to game2 for T_phase2 steps, preserving Q-values + tau
      Control: Train on game2 from scratch for T_phase2 steps
    """
    g1 = GAMES[game1_key]
    g2 = GAMES[game2_key]

    # Phase 1: train on game1
    traj1_A, traj1_B, act1_A, act1_B, Q_A_end, Q_B_end, tau_end = \
        run_q_learning_with_state(
            g1['payoff_A'], g1['payoff_B'], T=T_phase1,
            alpha=0.1, gamma=0.95, tau_init=1.0, tau_decay=0.9998, seed=seed)

    # Phase 2: switch to game2 with preserved state
    traj2_A, traj2_B, act2_A, act2_B, _, _, _ = \
        run_q_learning_with_state(
            g2['payoff_A'], g2['payoff_B'], T=T_phase2,
            alpha=0.1, gamma=0.95, tau_decay=0.9998,
            Q_A_init=Q_A_end, Q_B_init=Q_B_end, tau_start=tau_end,
            seed=seed + 1000)

    # Control: game2 from scratch, same tau start point for fair comparison
    traj_ctrl_A, traj_ctrl_B, act_ctrl_A, act_ctrl_B, _, _, _ = \
        run_q_learning_with_state(
            g2['payoff_A'], g2['payoff_B'], T=T_phase2,
            alpha=0.1, gamma=0.95, tau_init=tau_end, tau_decay=0.9998,
            seed=seed + 1000)

    return {
        'phase1': {
            'traj_A': traj1_A, 'traj_B': traj1_B,
            'act_A': act1_A, 'act_B': act1_B,
            'Q_A': Q_A_end, 'Q_B': Q_B_end, 'tau': tau_end,
        },
        'transfer': {
            'traj_A': traj2_A, 'traj_B': traj2_B,
            'act_A': act2_A, 'act_B': act2_B,
        },
        'control': {
            'traj_A': traj_ctrl_A, 'traj_B': traj_ctrl_B,
            'act_A': act_ctrl_A, 'act_B': act_ctrl_B,
        },
    }


def run_multi_seed_transfer(game1_key, game2_key, n_seeds=15,
                             T_phase1=2000, T_phase2=3000):
    """Run cross-switch for multiple seeds and aggregate."""
    transfer_ent = []
    control_ent = []
    transfer_spread = []
    control_spread = []

    for seed in range(n_seeds):
        res = run_cross_switch_experiment(game1_key, game2_key,
                                          T_phase1, T_phase2, seed=seed)

        # Entropy in first 500 steps of phase 2 (early adaptation)
        te_early = np.mean([compute_entropy(res['transfer']['traj_A'][t])
                           for t in range(min(500, T_phase2))])
        ce_early = np.mean([compute_entropy(res['control']['traj_A'][t])
                           for t in range(min(500, T_phase2))])
        transfer_ent.append(te_early)
        control_ent.append(ce_early)

        # Spread in first 500 steps
        transfer_spread.append(res['transfer']['traj_A'][:500].std(axis=0).mean())
        control_spread.append(res['control']['traj_A'][:500].std(axis=0).mean())

    return {
        'transfer_ent': np.array(transfer_ent),
        'control_ent': np.array(control_ent),
        'transfer_spread': np.array(transfer_spread),
        'control_spread': np.array(control_spread),
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_single_experiment(result, case_label, game1_name, game2_name):
    """Plot a single cross-switch experiment: phase1, transfer, control."""
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(f'{case_label}: {game1_name} → {game2_name}\n'
                 f'(Phase 1: {game1_name}, Phase 2: {game2_name} with preserved state vs. fresh control)',
                 fontsize=13, fontweight='bold')

    phases = [
        ('Phase 1\n' + game1_name, result['phase1'], 'tab:blue'),
        ('Phase 2 (transfer)\n' + game2_name, result['transfer'], 'tab:red'),
        ('Phase 2 (control)\n' + game2_name + ' fresh', result['control'], 'tab:green'),
    ]

    for row, (label, data, color) in enumerate(phases):
        traj_A = data['traj_A']
        traj_B = data['traj_B']
        act_A = data['act_A']
        T = len(traj_A)
        n_act = traj_A.shape[1]

        # Col 0: Strategy trajectories
        ax = axes[row, 0]
        for i in range(n_act):
            ax.plot(traj_A[:, i], alpha=0.7, linewidth=0.5)
        ax.set_ylabel(label, fontsize=9, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.set_title('Player A Strategy Trajectories', fontsize=10)
        if row == 2:
            ax.set_xlabel('Time step')

        # Col 1: Entropy over time
        ax = axes[row, 1]
        ent_ts = [compute_entropy(traj_A[t]) for t in range(T)]
        ax.plot(ent_ts, color=color, alpha=0.5, linewidth=0.5)
        # Smoothed
        w = min(100, T // 10)
        if w > 1:
            smoothed = np.convolve(ent_ts, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(ent_ts)), smoothed, color=color, linewidth=2)
        ax.set_ylabel('Entropy (bits)')
        max_ent = np.log2(n_act)
        ax.axhline(max_ent, color='gray', linestyle='--', linewidth=0.5)
        if row == 0:
            ax.set_title('Strategy Entropy', fontsize=10)
        if row == 2:
            ax.set_xlabel('Time step')

        # Col 2: Phase portrait
        ax = axes[row, 2]
        if n_act >= 3:
            x_s = traj_A[:, 1] + 0.5 * traj_A[:, 2]
            y_s = (np.sqrt(3) / 2) * traj_A[:, 2]
            ax.scatter(x_s, y_s, c=np.arange(T), cmap='viridis', s=1, alpha=0.5)
            ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', linewidth=0.3)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.0)
            ax.set_aspect('equal')
        else:
            ax.scatter(traj_A[:, 0], traj_B[:, 0], c=np.arange(T),
                       cmap='viridis', s=1, alpha=0.5)
            ax.set_xlabel('P_A(0)')
            ax.set_ylabel('P_B(0)')
        if row == 0:
            ax.set_title('Phase Portrait', fontsize=10)

        # Col 3: Action distribution (bar chart)
        ax = axes[row, 3]
        counts = np.bincount(act_A, minlength=n_act)
        ax.bar(range(n_act), counts / T, color=color, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Fraction')
        ax.set_xticks(range(n_act))
        if row == 0:
            ax.set_title('Action Distribution', fontsize=10)
        if row == 2:
            ax.set_xlabel('Action')

        # Annotate with summary stats
        mean_ent = np.mean(ent_ts[len(ent_ts)//2:])
        spr = traj_A[len(traj_A)//2:].std(axis=0).mean()
        ax.text(0.95, 0.95, f'H={mean_ent:.3f}\nσ={spr:.3f}',
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig


def plot_all_experiments(all_results, all_ms):
    """Create the master 4-case comparison figure."""
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Cross-Game Transfer: Does Adversarial Topology Leave Reusable Dynamical Structure?',
                 fontsize=15, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(4, 5, hspace=0.35, wspace=0.35)

    case_labels = ['Case A', 'Case B', 'Case C', 'Case D']
    case_descs = [
        'RPS → PD\n(adversarial → convergent)',
        'PD → RPS\n(convergent → adversarial)',
        'RPS → Stag Hunt\n(adversarial → coordination)',
        'Stag Hunt → RPS\n(coordination → adversarial)',
    ]

    for row, (case, desc, res, ms) in enumerate(
            zip(case_labels, case_descs, all_results, all_ms)):

        traj_transfer = res['transfer']['traj_A']
        traj_control = res['control']['traj_A']
        T = len(traj_transfer)
        n_act = traj_transfer.shape[1]

        # Col 0: Case label + Q-value state
        ax = fig.add_subplot(gs[row, 0])
        ax.text(0.5, 0.5, f'{case}\n\n{desc}', transform=ax.transAxes,
                fontsize=11, ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        # Show Q-values from phase 1
        Q_A = res['phase1']['Q_A']
        q_text = 'Q-values after Phase 1:\n' + \
                 ', '.join([f'{q:.2f}' for q in Q_A]) + \
                 f'\nτ = {res["phase1"]["tau"]:.4f}'
        ax.text(0.5, 0.1, q_text, transform=ax.transAxes,
                fontsize=7, ha='center', va='bottom', fontfamily='monospace')
        ax.axis('off')

        # Col 1: Entropy comparison (transfer vs control)
        ax = fig.add_subplot(gs[row, 1])
        ent_t = [compute_entropy(traj_transfer[t]) for t in range(T)]
        ent_c = [compute_entropy(traj_control[t]) for t in range(T)]
        w = 50
        if len(ent_t) > w:
            sm_t = np.convolve(ent_t, np.ones(w)/w, mode='valid')
            sm_c = np.convolve(ent_c, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(ent_t)), sm_t, 'r-', linewidth=2,
                    label='Transfer', alpha=0.9)
            ax.plot(range(w-1, len(ent_c)), sm_c, 'g-', linewidth=2,
                    label='Control', alpha=0.9)
        ax.set_ylabel('Entropy (bits)')
        if row == 0:
            ax.set_title('Entropy: Transfer vs Control', fontsize=10)
            ax.legend(fontsize=7)
        if row == 3:
            ax.set_xlabel('Time step')

        # Col 2: Strategy trajectory comparison
        ax = fig.add_subplot(gs[row, 2])
        for i in range(n_act):
            ax.plot(traj_transfer[:, i], alpha=0.5, linewidth=0.5)
        ax.set_ylabel('P(action)')
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.set_title('Transfer Trajectory', fontsize=10)
        if row == 3:
            ax.set_xlabel('Time step')

        # Col 3: Control trajectory
        ax = fig.add_subplot(gs[row, 3])
        for i in range(n_act):
            ax.plot(traj_control[:, i], alpha=0.5, linewidth=0.5)
        ax.set_ylabel('P(action)')
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.set_title('Control Trajectory', fontsize=10)
        if row == 3:
            ax.set_xlabel('Time step')

        # Col 4: Multi-seed early-adaptation comparison
        ax = fig.add_subplot(gs[row, 4])
        te = ms['transfer_ent']
        ce = ms['control_ent']
        bp = ax.boxplot([te, ce], labels=['Transfer', 'Control'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('salmon')
        bp['boxes'][1].set_facecolor('lightgreen')
        diff = np.mean(te) - np.mean(ce)
        from scipy.stats import mannwhitneyu
        try:
            stat, pval = mannwhitneyu(te, ce, alternative='two-sided')
        except:
            pval = 1.0
        ax.set_title(f'Early H (n=15)\nΔ={diff:+.3f}, p={pval:.3f}', fontsize=8)
        ax.set_ylabel('Entropy (bits)')

    plt.savefig(os.path.join(OUTPUT_DIR, 'cross_game_transfer.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_transfer_summary(all_ms, case_labels, case_descs):
    """Summary figure: early entropy difference across all 4 cases."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Cross-Game Transfer Summary: Early Adaptation Entropy (first 500 steps)\n'
                 'Red = transferred state, Green = fresh initialization',
                 fontsize=13, fontweight='bold')

    for i, (ms, label, desc) in enumerate(zip(all_ms, case_labels, case_descs)):
        ax = axes[i]
        te = ms['transfer_ent']
        ce = ms['control_ent']

        # Paired comparison
        ax.bar([0, 1], [np.mean(te), np.mean(ce)],
               yerr=[np.std(te), np.std(ce)],
               color=['salmon', 'lightgreen'], edgecolor='black',
               linewidth=0.5, capsize=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Transfer', 'Control'], fontsize=9)
        ax.set_ylabel('Entropy (bits)')

        from scipy.stats import mannwhitneyu
        try:
            stat, pval = mannwhitneyu(te, ce, alternative='two-sided')
        except:
            pval = 1.0
        diff = np.mean(te) - np.mean(ce)

        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.set_title(f'{label}: {desc}\nΔH = {diff:+.3f}, p = {pval:.4f} {sig}',
                     fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'transfer_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  CROSS-GAME TRANSFER EXPERIMENTS — Round 3")
    print("  Does adversarial topology leave reusable dynamical structure?")
    print("=" * 70)

    experiments = [
        ('RPS', 'PrisonersDilemma', 'Case A', 'RPS → PD'),
        ('PrisonersDilemma', 'RPS', 'Case B', 'PD → RPS'),
        ('RPS', 'StagHunt', 'Case C', 'RPS → Stag Hunt'),
        ('StagHunt', 'RPS', 'Case D', 'Stag Hunt → RPS'),
    ]

    all_results = []
    all_ms = []
    case_labels = []
    case_descs = []

    for g1, g2, label, desc in experiments:
        print(f"\n{'─'*60}")
        print(f"  {label}: {GAMES[g1]['name']} → {GAMES[g2]['name']}")
        print(f"{'─'*60}")

        # Single detailed run
        res = run_cross_switch_experiment(g1, g2, T_phase1=2000,
                                           T_phase2=3000, seed=42)

        # Print Q-values
        print(f"  Phase 1 final Q_A: {res['phase1']['Q_A']}")
        print(f"  Phase 1 final τ:   {res['phase1']['tau']:.6f}")

        # Entropy comparison
        tA_t = res['transfer']['traj_A']
        tA_c = res['control']['traj_A']
        ent_t = np.mean([compute_entropy(tA_t[t]) for t in range(500)])
        ent_c = np.mean([compute_entropy(tA_c[t]) for t in range(500)])
        print(f"  Early entropy (transfer): {ent_t:.4f}")
        print(f"  Early entropy (control):  {ent_c:.4f}")
        print(f"  Difference: {ent_t - ent_c:+.4f}")

        # Save individual experiment figure
        fig = plot_single_experiment(res, label, GAMES[g1]['name'], GAMES[g2]['name'])
        fig.savefig(os.path.join(OUTPUT_DIR, f'transfer_{label.replace(" ","_")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Multi-seed analysis
        print(f"  Running multi-seed analysis (15 seeds)...")
        ms = run_multi_seed_transfer(g1, g2, n_seeds=15,
                                      T_phase1=2000, T_phase2=3000)
        from scipy.stats import mannwhitneyu
        try:
            stat, pval = mannwhitneyu(ms['transfer_ent'], ms['control_ent'],
                                      alternative='two-sided')
        except:
            pval = 1.0
        print(f"  Multi-seed early H: transfer={np.mean(ms['transfer_ent']):.4f}±"
              f"{np.std(ms['transfer_ent']):.4f}, "
              f"control={np.mean(ms['control_ent']):.4f}±"
              f"{np.std(ms['control_ent']):.4f}, p={pval:.4f}")

        all_results.append(res)
        all_ms.append(ms)
        case_labels.append(label)
        case_descs.append(desc)

    # Master comparison figure
    print(f"\n{'='*60}")
    print("  Generating master comparison figures...")
    plot_all_experiments(all_results, all_ms)
    plot_transfer_summary(all_ms, case_labels,
                          ['RPS→PD', 'PD→RPS', 'RPS→SH', 'SH→RPS'])

    # Print summary
    print(f"\n{'='*70}")
    print("  SUMMARY: CROSS-GAME TRANSFER RESULTS")
    print(f"{'='*70}")
    for (g1, g2, label, desc), ms in zip(experiments, all_ms):
        te = ms['transfer_ent']
        ce = ms['control_ent']
        diff = np.mean(te) - np.mean(ce)
        try:
            _, pval = mannwhitneyu(te, ce, alternative='two-sided')
        except:
            pval = 1.0
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        print(f"  {label} ({desc}): ΔH = {diff:+.4f}, p = {pval:.4f} {sig}")
    print(f"{'='*70}")

    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
