#!/usr/bin/env python3
"""
Adversarial-to-Adversarial Transfer Experiments — Round 4
=========================================================
Tests whether adversarial learning preserves REUSABLE adaptive structure
(not merely avoids rigidity) by transferring between different zero-sum games.

Cases:
  E: RPS → Shapley's Game (3x3 adversarial → 3x3 adversarial)
  F: Shapley → RPS (3x3 adversarial → 3x3 adversarial)
  G: RPS → Matching Pennies (3-action adversarial → 2-action adversarial)
  H: Matching Pennies → RPS (2-action adversarial → 3-action adversarial)

Each compared against:
  - Fresh (naive) initialization (same tau)
  - Convergent→adversarial baselines from R3 (Cases B, D) for reference

Expected positive outcomes per reviewer:
  (1) Accelerated adaptation relative to naive
  (2) More rapid recovery of rotational exploration
  (3) Faster entropy stabilization
  (4) Broader policy support during early learning
  (5) Reduced catastrophic freezing vs convergent→adversarial
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/claude')
from structured_instability_simulations import GAMES, softmax, compute_entropy
from cross_game_transfer import run_q_learning_with_state, run_cross_switch_experiment

OUTPUT_DIR = '/home/claude/simulation_results_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_PHASE1 = 2000
T_PHASE2 = 3000
N_SEEDS = 15


def compute_adaptation_metrics(traj_A, n_windows=6):
    """Compute detailed temporal adaptation metrics."""
    T = len(traj_A)
    ws = T // n_windows
    results = []
    for w in range(n_windows):
        s, e = w * ws, (w + 1) * ws
        seg = traj_A[s:e]
        ent = np.mean([compute_entropy(seg[t]) for t in range(len(seg))])
        spread = seg.std(axis=0).mean()
        # Policy support: number of actions with P > 0.1
        mean_policy = seg.mean(axis=0)
        support = np.sum(mean_policy > 0.1)
        results.append({'window': w, 'entropy': ent, 'spread': spread,
                        'support': support, 'time_start': s, 'time_end': e})
    return results


def run_multi_seed_detailed(game1_key, game2_key, n_seeds=N_SEEDS):
    """Run transfer experiment for multiple seeds, collecting detailed metrics."""
    all_transfer = {'early_ent': [], 'mid_ent': [], 'late_ent': [],
                    'early_spread': [], 'early_support': [],
                    'time_to_cycling': []}
    all_control = {'early_ent': [], 'mid_ent': [], 'late_ent': [],
                   'early_spread': [], 'early_support': [],
                   'time_to_cycling': []}

    for seed in range(n_seeds):
        res = run_cross_switch_experiment(game1_key, game2_key,
                                          T_PHASE1, T_PHASE2, seed=seed)

        for label, data, store in [('transfer', res['transfer'], all_transfer),
                                    ('control', res['control'], all_control)]:
            traj = data['traj_A']
            T = len(traj)

            # Early (first 500), mid (500-1500), late (1500+)
            early_ent = np.mean([compute_entropy(traj[t]) for t in range(min(500, T))])
            mid_ent = np.mean([compute_entropy(traj[t]) for t in range(500, min(1500, T))])
            late_ent = np.mean([compute_entropy(traj[t]) for t in range(1500, T)])
            store['early_ent'].append(early_ent)
            store['mid_ent'].append(mid_ent)
            store['late_ent'].append(late_ent)

            # Early spread and support
            early_spread = traj[:500].std(axis=0).mean()
            mean_pol = traj[:500].mean(axis=0)
            support = np.sum(mean_pol > 0.1)
            store['early_spread'].append(early_spread)
            store['early_support'].append(support)

            # Time to cycling: first timestep where smoothed entropy > 0.5
            ents = [compute_entropy(traj[t]) for t in range(T)]
            w = 50
            if len(ents) > w:
                smoothed = np.convolve(ents, np.ones(w)/w, mode='valid')
                cycling_idx = np.where(smoothed > 0.5)[0]
                ttc = cycling_idx[0] if len(cycling_idx) > 0 else T
            else:
                ttc = T
            store['time_to_cycling'].append(ttc)

    # Convert to arrays
    for store in [all_transfer, all_control]:
        for k in store:
            store[k] = np.array(store[k])

    return all_transfer, all_control


def plot_master_figure(all_results, all_ms, conv_baselines):
    """Create the master comparison figure for adversarial→adversarial."""
    fig = plt.figure(figsize=(26, 22))
    fig.suptitle('Adversarial → Adversarial Transfer:\n'
                 'Does Adversarial Learning Preserve Reusable Adaptive Structure?',
                 fontsize=15, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.4)

    cases = [
        ('E', 'RPS → Shapley', 'RPS', 'Shapley'),
        ('F', 'Shapley → RPS', 'Shapley', 'RPS'),
        ('G', 'RPS → Match.Pen.', 'RPS', 'MatchingPennies'),
        ('H', 'Match.Pen. → RPS', 'MatchingPennies', 'RPS'),
    ]

    for row, (case_id, desc, g1, g2) in enumerate(cases):
        res = all_results[row]
        ms_t, ms_c = all_ms[row]

        traj_t = res['transfer']['traj_A']
        traj_c = res['control']['traj_A']
        T = len(traj_t)

        # Col 0: Case label
        ax = fig.add_subplot(gs[row, 0])
        Q_A = res['phase1']['Q_A']
        q_text = f'Case {case_id}\n\n{desc}\n(adversarial → adversarial)\n\n'
        q_text += 'Q after Phase 1:\n' + ', '.join([f'{q:.1f}' for q in Q_A])
        q_text += f'\nτ = {res["phase1"]["tau"]:.4f}'
        ax.text(0.5, 0.5, q_text, transform=ax.transAxes, fontsize=9,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax.axis('off')

        # Col 1: Entropy trajectories
        ax = fig.add_subplot(gs[row, 1])
        ent_t = [compute_entropy(traj_t[t]) for t in range(T)]
        ent_c = [compute_entropy(traj_c[t]) for t in range(T)]
        w = 50
        sm_t = np.convolve(ent_t, np.ones(w)/w, mode='valid')
        sm_c = np.convolve(ent_c, np.ones(w)/w, mode='valid')
        ax.plot(range(w-1, T), sm_t, 'r-', linewidth=2, label='Adv. transfer', alpha=0.9)
        ax.plot(range(w-1, T), sm_c, 'g-', linewidth=2, label='Fresh control', alpha=0.9)
        # Add convergent baseline if available
        if conv_baselines[row] is not None:
            cb = conv_baselines[row]
            ent_cb = [compute_entropy(cb['traj_A'][t]) for t in range(min(T, len(cb['traj_A'])))]
            if len(ent_cb) > w:
                sm_cb = np.convolve(ent_cb, np.ones(w)/w, mode='valid')
                ax.plot(range(w-1, len(ent_cb)), sm_cb, 'b--', linewidth=1.5,
                        label='Conv. transfer', alpha=0.7)
        ax.set_ylabel('Entropy (bits)')
        if row == 0:
            ax.set_title('Entropy Trajectories', fontsize=10)
            ax.legend(fontsize=6, loc='lower right')
        if row == 3:
            ax.set_xlabel('Time step')

        # Col 2: Early entropy box plot (transfer vs control vs conv baseline)
        ax = fig.add_subplot(gs[row, 2])
        data_boxes = [ms_t['early_ent'], ms_c['early_ent']]
        labels_boxes = ['Adv.\ntransfer', 'Fresh\ncontrol']
        colors_boxes = ['salmon', 'lightgreen']
        if conv_baselines[row] is not None and 'ms' in conv_baselines[row]:
            data_boxes.append(conv_baselines[row]['ms']['early_ent'])
            labels_boxes.append('Conv.\ntransfer')
            colors_boxes.append('lightblue')
        bp = ax.boxplot(data_boxes, labels=labels_boxes, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_boxes):
            patch.set_facecolor(color)
        try:
            _, pval = mannwhitneyu(ms_t['early_ent'], ms_c['early_ent'], alternative='two-sided')
        except:
            pval = 1.0
        diff = np.mean(ms_t['early_ent']) - np.mean(ms_c['early_ent'])
        ax.set_title(f'Early H (n={N_SEEDS})\nΔ={diff:+.3f} p={pval:.3f}', fontsize=8)
        if row == 0:
            ax.set_title(f'Early Entropy Comparison\nΔ={diff:+.3f} p={pval:.3f}', fontsize=9)

        # Col 3: Time to cycling box plot
        ax = fig.add_subplot(gs[row, 3])
        data_ttc = [ms_t['time_to_cycling'], ms_c['time_to_cycling']]
        labels_ttc = ['Adv.\ntransfer', 'Fresh\ncontrol']
        colors_ttc = ['salmon', 'lightgreen']
        if conv_baselines[row] is not None and 'ms' in conv_baselines[row]:
            data_ttc.append(conv_baselines[row]['ms']['time_to_cycling'])
            labels_ttc.append('Conv.\ntransfer')
            colors_ttc.append('lightblue')
        bp = ax.boxplot(data_ttc, labels=labels_ttc, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_ttc):
            patch.set_facecolor(color)
        try:
            _, pval_ttc = mannwhitneyu(ms_t['time_to_cycling'], ms_c['time_to_cycling'],
                                       alternative='less')
        except:
            pval_ttc = 1.0
        ax.set_ylabel('Timesteps')
        if row == 0:
            ax.set_title('Time to Cycling (H > 0.5)', fontsize=9)
        ax.set_title(f'Time to cycling\np={pval_ttc:.3f}', fontsize=8)

        # Col 4: Policy support (number of actions with P>0.1)
        ax = fig.add_subplot(gs[row, 4])
        data_sup = [ms_t['early_support'], ms_c['early_support']]
        labels_sup = ['Adv.\ntransfer', 'Fresh\ncontrol']
        colors_sup = ['salmon', 'lightgreen']
        if conv_baselines[row] is not None and 'ms' in conv_baselines[row]:
            data_sup.append(conv_baselines[row]['ms']['early_support'])
            labels_sup.append('Conv.\ntransfer')
            colors_sup.append('lightblue')
        bp = ax.boxplot(data_sup, labels=labels_sup, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors_sup):
            patch.set_facecolor(color)
        ax.set_ylabel('# actions with P > 0.1')
        if row == 0:
            ax.set_title('Early Policy Support', fontsize=9)

        # Col 5: Strategy trajectory (transfer)
        ax = fig.add_subplot(gs[row, 5])
        n_act = traj_t.shape[1]
        for i in range(n_act):
            ax.plot(traj_t[:, i], alpha=0.6, linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.set_title('Transfer Trajectory', fontsize=9)
        if row == 3:
            ax.set_xlabel('Time step')

    plt.savefig(os.path.join(OUTPUT_DIR, 'adversarial_transfer.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_comparison(all_ms, conv_baselines_ms):
    """Summary bar chart comparing adv→adv, fresh, and conv→adv."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    fig.suptitle('Adversarial → Adversarial Transfer: Summary Across All Five Reviewer Criteria',
                 fontsize=13, fontweight='bold')

    cases = ['E: RPS→Shapley', 'F: Shapley→RPS',
             'G: RPS→M.Pennies', 'H: M.Pennies→RPS']

    # Metric 1: Early entropy
    ax = axes[0]
    ax.set_title('(1) Early Entropy\n(higher = more exploration)', fontsize=9)
    x = np.arange(len(cases))
    w = 0.25
    t_vals = [np.mean(ms[0]['early_ent']) for ms in all_ms]
    c_vals = [np.mean(ms[1]['early_ent']) for ms in all_ms]
    ax.bar(x - w, t_vals, w, color='salmon', label='Adv. transfer', edgecolor='black', linewidth=0.5)
    ax.bar(x, c_vals, w, color='lightgreen', label='Fresh control', edgecolor='black', linewidth=0.5)
    if any(cb is not None for cb in conv_baselines_ms):
        cb_vals = [np.mean(cb['early_ent']) if cb is not None else 0 for cb in conv_baselines_ms]
        ax.bar(x + w, cb_vals, w, color='lightblue', label='Conv. transfer', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=7, rotation=15)
    ax.set_ylabel('Entropy (bits)')
    ax.legend(fontsize=6)

    # Metric 2: Time to cycling
    ax = axes[1]
    ax.set_title('(2) Time to Cycling\n(lower = faster recovery)', fontsize=9)
    t_vals = [np.mean(ms[0]['time_to_cycling']) for ms in all_ms]
    c_vals = [np.mean(ms[1]['time_to_cycling']) for ms in all_ms]
    ax.bar(x - w, t_vals, w, color='salmon', edgecolor='black', linewidth=0.5)
    ax.bar(x, c_vals, w, color='lightgreen', edgecolor='black', linewidth=0.5)
    if any(cb is not None for cb in conv_baselines_ms):
        cb_vals = [np.mean(cb['time_to_cycling']) if cb is not None else T_PHASE2 for cb in conv_baselines_ms]
        ax.bar(x + w, cb_vals, w, color='lightblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=7, rotation=15)
    ax.set_ylabel('Timesteps')

    # Metric 3: Entropy stabilization (late entropy)
    ax = axes[2]
    ax.set_title('(3) Late Entropy\n(stabilization level)', fontsize=9)
    t_vals = [np.mean(ms[0]['late_ent']) for ms in all_ms]
    c_vals = [np.mean(ms[1]['late_ent']) for ms in all_ms]
    ax.bar(x - w, t_vals, w, color='salmon', edgecolor='black', linewidth=0.5)
    ax.bar(x, c_vals, w, color='lightgreen', edgecolor='black', linewidth=0.5)
    if any(cb is not None for cb in conv_baselines_ms):
        cb_vals = [np.mean(cb['late_ent']) if cb is not None else 0 for cb in conv_baselines_ms]
        ax.bar(x + w, cb_vals, w, color='lightblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=7, rotation=15)
    ax.set_ylabel('Entropy (bits)')

    # Metric 4: Early policy support
    ax = axes[3]
    ax.set_title('(4) Early Policy Support\n(# actions with P > 0.1)', fontsize=9)
    t_vals = [np.mean(ms[0]['early_support']) for ms in all_ms]
    c_vals = [np.mean(ms[1]['early_support']) for ms in all_ms]
    ax.bar(x - w, t_vals, w, color='salmon', edgecolor='black', linewidth=0.5)
    ax.bar(x, c_vals, w, color='lightgreen', edgecolor='black', linewidth=0.5)
    if any(cb is not None for cb in conv_baselines_ms):
        cb_vals = [np.mean(cb['early_support']) if cb is not None else 0 for cb in conv_baselines_ms]
        ax.bar(x + w, cb_vals, w, color='lightblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=7, rotation=15)
    ax.set_ylabel('# actions')

    # Metric 5: Early spread
    ax = axes[4]
    ax.set_title('(5) Early Strategy Spread\n(higher = less frozen)', fontsize=9)
    t_vals = [np.mean(ms[0]['early_spread']) for ms in all_ms]
    c_vals = [np.mean(ms[1]['early_spread']) for ms in all_ms]
    ax.bar(x - w, t_vals, w, color='salmon', edgecolor='black', linewidth=0.5)
    ax.bar(x, c_vals, w, color='lightgreen', edgecolor='black', linewidth=0.5)
    if any(cb is not None for cb in conv_baselines_ms):
        cb_vals = [np.mean(cb['early_spread']) if cb is not None else 0 for cb in conv_baselines_ms]
        ax.bar(x + w, cb_vals, w, color='lightblue', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, fontsize=7, rotation=15)
    ax.set_ylabel('σ')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'adversarial_transfer_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("  ADVERSARIAL → ADVERSARIAL TRANSFER — Round 4")
    print("  Does adversarial learning preserve REUSABLE adaptive structure?")
    print("=" * 70)

    experiments = [
        ('RPS', 'Shapley', 'E', 'RPS → Shapley'),
        ('Shapley', 'RPS', 'F', 'Shapley → RPS'),
        ('RPS', 'MatchingPennies', 'G', 'RPS → Matching Pennies'),
        ('MatchingPennies', 'RPS', 'H', 'Matching Pennies → RPS'),
    ]

    # Also run convergent→adversarial baselines for comparison
    conv_targets = {
        'Shapley': ('PrisonersDilemma', 'Shapley'),
        'RPS': ('PrisonersDilemma', 'RPS'),
        'MatchingPennies': ('StagHunt', 'MatchingPennies'),
    }

    all_results = []
    all_ms = []
    conv_baselines = []
    conv_baselines_ms = []

    for g1, g2, label, desc in experiments:
        print(f"\n{'─'*60}")
        print(f"  Case {label}: {GAMES[g1]['name']} → {GAMES[g2]['name']}")
        print(f"{'─'*60}")

        # Single detailed run
        res = run_cross_switch_experiment(g1, g2, T_PHASE1, T_PHASE2, seed=42)
        print(f"  Phase 1 Q_A: {res['phase1']['Q_A']}")
        print(f"  Phase 1 τ:   {res['phase1']['tau']:.6f}")

        # Multi-seed
        print(f"  Running multi-seed ({N_SEEDS} seeds)...")
        ms_t, ms_c = run_multi_seed_detailed(g1, g2, N_SEEDS)

        # Stats
        diff_ent = np.mean(ms_t['early_ent']) - np.mean(ms_c['early_ent'])
        try:
            _, p_ent = mannwhitneyu(ms_t['early_ent'], ms_c['early_ent'], alternative='two-sided')
        except:
            p_ent = 1.0
        diff_ttc = np.mean(ms_t['time_to_cycling']) - np.mean(ms_c['time_to_cycling'])
        try:
            _, p_ttc = mannwhitneyu(ms_t['time_to_cycling'], ms_c['time_to_cycling'], alternative='less')
        except:
            p_ttc = 1.0
        print(f"  Early H: transfer={np.mean(ms_t['early_ent']):.4f}±{np.std(ms_t['early_ent']):.4f}, "
              f"control={np.mean(ms_c['early_ent']):.4f}±{np.std(ms_c['early_ent']):.4f}, "
              f"ΔH={diff_ent:+.4f}, p={p_ent:.4f}")
        print(f"  Time to cycling: transfer={np.mean(ms_t['time_to_cycling']):.0f}±{np.std(ms_t['time_to_cycling']):.0f}, "
              f"control={np.mean(ms_c['time_to_cycling']):.0f}±{np.std(ms_c['time_to_cycling']):.0f}, "
              f"Δ={diff_ttc:+.0f}, p={p_ttc:.4f}")
        print(f"  Policy support: transfer={np.mean(ms_t['early_support']):.1f}±{np.std(ms_t['early_support']):.1f}, "
              f"control={np.mean(ms_c['early_support']):.1f}±{np.std(ms_c['early_support']):.1f}")

        all_results.append(res)
        all_ms.append((ms_t, ms_c))

        # Convergent→adversarial baseline
        if g2 in conv_targets:
            cg1, cg2 = conv_targets[g2]
            print(f"  Running conv baseline: {cg1} → {cg2}...")
            conv_res = run_cross_switch_experiment(cg1, cg2, T_PHASE1, T_PHASE2, seed=42)
            conv_ms_t, conv_ms_c = run_multi_seed_detailed(cg1, cg2, N_SEEDS)
            conv_baselines.append({'traj_A': conv_res['transfer']['traj_A'], 'ms': conv_ms_t})
            conv_baselines_ms.append(conv_ms_t)
            print(f"  Conv baseline early H: {np.mean(conv_ms_t['early_ent']):.4f}±{np.std(conv_ms_t['early_ent']):.4f}")
        else:
            conv_baselines.append(None)
            conv_baselines_ms.append(None)

    # Plot
    print(f"\n{'='*60}")
    print("  Generating figures...")
    plot_master_figure(all_results, all_ms, conv_baselines)
    plot_summary_comparison(all_ms, conv_baselines_ms)

    # Final summary
    print(f"\n{'='*70}")
    print("  SUMMARY: ADVERSARIAL → ADVERSARIAL TRANSFER")
    print(f"{'='*70}")
    print(f"{'Case':<25} {'ΔH(early)':>10} {'p(H)':>8} {'ΔT(cyc)':>10} {'p(T)':>8} {'Supp_t':>7} {'Supp_c':>7}")
    for (g1, g2, label, desc), (ms_t, ms_c) in zip(experiments, all_ms):
        diff_e = np.mean(ms_t['early_ent']) - np.mean(ms_c['early_ent'])
        try: _, pe = mannwhitneyu(ms_t['early_ent'], ms_c['early_ent'], alternative='two-sided')
        except: pe = 1.0
        diff_t = np.mean(ms_t['time_to_cycling']) - np.mean(ms_c['time_to_cycling'])
        try: _, pt = mannwhitneyu(ms_t['time_to_cycling'], ms_c['time_to_cycling'], alternative='less')
        except: pt = 1.0
        sup_t = np.mean(ms_t['early_support'])
        sup_c = np.mean(ms_c['early_support'])
        sig_e = '***' if pe < 0.001 else '**' if pe < 0.01 else '*' if pe < 0.05 else 'ns'
        print(f"  {label} {desc:<20} {diff_e:>+10.4f} {pe:>7.4f}{sig_e:>3} {diff_t:>+10.0f} {pt:>7.4f} {sup_t:>7.1f} {sup_c:>7.1f}")
    print(f"{'='*70}")

    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == '__main__':
    main()
