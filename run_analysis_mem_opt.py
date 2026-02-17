# -*- coding: utf-8 -*-
"""
Master Analysis Script: Memory Optimized with 100% Logic Preservation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tax_model as tm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PART 1: DATA GENERATION (Memory Optimized)
# =============================================================================

def generate_grid_data(beta_vals, sigma_vals, n_agents=2000000):
    print(f"\n--- PRE-COMPUTING DATA GRID ({len(beta_vals)}x{len(sigma_vals)} scenarios) ---")
    results = {}
    total = len(beta_vals) * len(sigma_vals)
    count = 0
    
    for beta in beta_vals:
        for snu in sigma_vals:
            count += 1
            print(f"  Simulating {count}/{total}: Beta={beta:.2f}, Sigma={snu:.1f}...", end="\r")
            
            # Generate the raw scenario from the model
            df, _ = tm.get_calibrated_scenario('lognormal', beta=beta, sigma_nu=snu, mode='loglinear', n_agents=n_agents)
            
            # STORE AS NUMPY ARRAYS (float32) to save ~60-70% RAM
            results[(beta, snu)] = {
                "True": df["True"].to_numpy(dtype=np.float32),
                "Reported": df["Reported"].to_numpy(dtype=np.float32),
                "EvasionRate": df["EvasionRate"].to_numpy(dtype=np.float32)
            }
            
    print("\n--- DATA GENERATION COMPLETE (Memory Optimized) ---\n")
    return results

def gini(x):
    x = np.abs(x); sorted_x = np.sort(x); n = len(x); index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_x)) / (n * np.sum(sorted_x)) - (n + 1) / n

# =============================================================================
# PART 2: GRID-BASED PLOTS & TABLES
# =============================================================================

def run_table1(results):
    print("Generating Table 1: Decomposition...")
    scenarios = [("Progressive / Low Het", 0.10, 0.4), 
                 ("Progressive / High Het", 0.10, 1.4),
                 ("Regressive / Low Het", -0.05, 0.4), 
                 ("Regressive / High Het", -0.05, 1.4)]
    out_list = []
    
    for name, beta, sigma_nu in scenarios:
        b_key = np.round(beta, 2)
        s_key = np.round(sigma_nu, 1)
        
        try:
            # Re-wrap for Pandas power
            df = pd.DataFrame(results[(b_key, s_key)]) 
        except KeyError:
            print(f"Warning: Key ({b_key}, {s_key}) not found in grid. Skipping.")
            continue

        k = int(len(df) * 0.01)
        
        # 1. Shares
        s_true = df.nlargest(k, 'True')['True'].sum() / df['True'].sum()
        s_rep = df.nlargest(k, 'Reported')['Reported'].sum() / df['Reported'].sum()
        
        # 2. Counterfactual: Reported Share of the TRUE Top 1%
        top_true_idx = df.nlargest(k, 'True').index
        s_rep_given_true = df.loc[top_true_idx, 'Reported'].sum() / df['Reported'].sum()
        
        # 3. Decomposition
        total_gap = s_true - s_rep
        measurement = s_true - s_rep_given_true  # Evasion effect
        reranking = s_rep_given_true - s_rep     # Re-ranking effect
        
        out_list.append({
            "Scenario": name, 
            "True Share": s_true, 
            "Reported Share": s_rep,
            "Total Gap": total_gap, 
            "Measurement": measurement, 
            "Re-ranking": reranking
        })
    
    out = pd.DataFrame(out_list)
    # Format as percentages
    for col in out.columns[1:]: 
        out[col] = out[col].apply(lambda x: f"{x*100:+.1f}%")
        
    print("\n" + out.to_string(index=False) + "\n")
    out.to_csv("Tab1_Decomposition.csv", index=False)

def run_all_heatmaps(results, beta_vals, sigma_vals):
    print("Generating Heatmaps (Figs 1, 2, 3, Combined, Gini)...")
    rows, cols = len(beta_vals), len(sigma_vals)
    
    # --- 1. Pre-allocate Matrices ---
    rate_1pct, rate_01pct = np.zeros((rows, cols)), np.zeros((rows, cols))
    agg_gap, gap_1pct = np.zeros((rows, cols)), np.zeros((rows, cols))
    gap_01pct, gini_diff = np.zeros((rows, cols)), np.zeros((rows, cols))
    
    # --- 2. Calculate Values ---
    for i, beta in enumerate(beta_vals):
        for j, snu in enumerate(sigma_vals):
            # Normalize keys to standard Python floats to avoid KeyError
            b_key = float(np.round(beta, 2))
            s_key = float(np.round(snu, 1))

            # Use the normalized keys to unwrap the data
            df = pd.DataFrame(results[(b_key, s_key)])
            k1, k01 = int(len(df)*0.01), int(len(df)*0.001)
            
            top1, top01 = df.nlargest(k1, 'True'), df.nlargest(k01, 'True')
            
            rate_1pct[i,j] = (top1['True'] - top1['Reported']).sum() / top1['True'].sum()
            rate_01pct[i,j] = (top01['True'] - top01['Reported']).sum() / top01['True'].sum()
            
            agg_gap[i,j] = (df['True'].sum() - df['Reported'].sum()) / df['True'].sum()
            gini_diff[i,j] = gini(df['True'].values) - gini(df['Reported'].values)
            
            t1 = top1['True'].sum() / df['True'].sum()
            r1 = df.nlargest(k1, 'Reported')['Reported'].sum() / df['Reported'].sum()
            gap_1pct[i,j] = t1 - r1
            
            t01 = top01['True'].sum() / df['True'].sum()
            r01 = df.nlargest(k01, 'Reported')['Reported'].sum() / df['Reported'].sum()
            gap_01pct[i,j] = t01 - r01

    # --- 3. Plotting Setup ---
    y_labels = [f"{b:.2f}" for b in np.flip(beta_vals)]
    
    # Define a standard style for "Sequential" maps (Reds)
    seq_style = dict(annot=True, fmt=".1%", cmap="Reds", cbar=False, xticklabels=sigma_vals)
    
    # Define a standard style for "Diverging" maps (RdBu)
    div_style = dict(annot=True, fmt=".1%", cmap="RdBu_r", center=0, cbar=False, xticklabels=sigma_vals)

    # --- Fig 1: Evasion Rates ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(np.flipud(rate_1pct), ax=ax[0], yticklabels=y_labels, **seq_style)
    ax[0].set_title("Avg Evasion Rate (Top 1%)")
    ax[0].set_ylabel("Beta")
    ax[0].set_xlabel("Sigma")
    
    sns.heatmap(np.flipud(rate_01pct), ax=ax[1], yticklabels=False, **seq_style)
    ax[1].set_title("Avg Evasion Rate (Top 0.1%)")
    ax[1].set_xlabel("Sigma")
    
    plt.tight_layout()
    plt.savefig("Fig_EvasionRates.pdf", bbox_inches='tight')
    plt.close()

    # --- Fig 2: Tax Gap ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.flipud(agg_gap), yticklabels=y_labels, **seq_style)
    plt.title("Aggregate Tax Gap")
    plt.xlabel("Sigma")
    plt.ylabel("Beta")
    plt.tight_layout()
    plt.savefig("Fig_TaxGap.pdf", bbox_inches='tight')
    plt.close()

    # --- Fig 3: Reported Gap ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(np.flipud(gap_1pct), ax=ax[0], yticklabels=y_labels, **div_style)
    ax[0].set_title("Reported Income Gap: Top 1% Share")
    ax[0].set_ylabel("Beta")
    ax[0].set_xlabel("Sigma")
    
    sns.heatmap(np.flipud(gap_01pct), ax=ax[1], yticklabels=False, **div_style)
    ax[1].set_title("Reported Income Gap: Top 0.1% Share")
    ax[1].set_xlabel("Sigma")
    
    plt.tight_layout()
    plt.savefig("Fig_ReportedGap.pdf", bbox_inches='tight')
    plt.close()
    
    # --- Fig Combined: Evasion & Gap ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Avg Evasion Rate
    sns.heatmap(
        np.flipud(rate_1pct), 
        ax=ax[0], 
        yticklabels=y_labels, 
        **seq_style
    )
    ax[0].set_title("A. Avg Evasion Rate (Top 1%)")
    ax[0].set_ylabel("Beta")
    ax[0].set_xlabel("Sigma")
    
    # Panel B: Aggregate Tax Gap
    sns.heatmap(
        np.flipud(agg_gap), 
        ax=ax[1], 
        yticklabels=False, 
        **seq_style
    )
    ax[1].set_title("B. Aggregate Tax Gap")
    ax[1].set_xlabel("Sigma")
    
    plt.tight_layout() # <--- PULLS PLOTS CLOSER
    plt.savefig("Fig_Combined_EvasionGap.pdf", bbox_inches='tight')
    plt.close()

    # --- Fig Gini ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        np.flipud(gini_diff), 
        annot=True, 
        fmt=".3f", 
        cmap="RdBu_r", 
        center=0, 
        cbar=False, 
        xticklabels=sigma_vals, 
        yticklabels=y_labels
    )
    plt.title("Gini Gap (True - Reported)")
    plt.xlabel("Sigma")
    plt.ylabel("Beta")
    plt.tight_layout()
    plt.savefig("Fig_GiniGap.pdf", bbox_inches='tight')
    plt.close()
    
def run_fig_lines(results):
    print("Generating Share Lines Fig...")

    def plot_panel(ax, beta, sigma_nu, title):
        # Mechanical substitution: wrap into df
        # Normalize keys to standard Python floats to avoid KeyError
        b_key = float(np.round(beta, 2))
        s_key = float(np.round(sigma_nu, 1))

        # Access and re-wrap the microdata for this panel only
        df = pd.DataFrame(results[(b_key, s_key)])
        
        # --- 1. X-Axis Setup (Top 10% to Top 0.1%) ---
        start_log, end_log = 1.0, 3.0
        x_log = np.linspace(start_log, end_log, 50)
        q_grid = 1 - 10**(-x_log)
        
        # --- 2. Calculate Curves ---
        tot_r, tot_t = df['Reported'].sum(), df['True'].sum()
        sort_r = df.sort_values('Reported', ascending=False)
        sort_t = df.sort_values('True', ascending=False)
        
        rep_c, true_c = [], []
        for q in q_grid:
            k = max(int(len(df) * (1 - q)), 1)
            rep_c.append(sort_r['Reported'].iloc[:k].sum() / tot_r)
            true_c.append(sort_t['True'].iloc[:k].sum() / tot_t)
            
        # --- 3. Plot Main Curves (Left Axis) ---
        ax.plot(x_log, rep_c, 'k', label='Reported (Ranked by Rep)', linewidth=1)
        ax.plot(x_log, true_c, 'purple', label='True (Ranked by True)', 
                linewidth=1, alpha=0.8)
        
        ax.set_ylabel("Shares")
        ax.set_ylim(0, 0.6)
        
        # --- 4. Plot Evasion Rate (Right Axis) ---
        ax2 = ax.twinx()
        bins_log = np.linspace(start_log, end_log, 21)
        bins_q = 1 - 10**(-bins_log)
        
        df['Rank_True'] = df['True'].rank(pct=True)
        df['Bin'] = pd.cut(df['Rank_True'], bins=bins_q)
        ev_profile = df.groupby('Bin', observed=False)['EvasionRate'].mean()
        bin_centers_log = (bins_log[:-1] + bins_log[1:]) / 2
        
        ax2.plot(
            bin_centers_log, 
            ev_profile.values, 
            color='#C00000', 
            linestyle=':', 
            linewidth=2, 
            label='Evasion Rate'
        )
        ax2.set_ylabel("Evasion Rate", color='#C00000', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#C00000')
        ax2.set_ylim(0, 0.5)
        
        # --- 5. Formatting ---
        ax.set_title(title, fontsize=11, weight='bold')
        ax.set_xlim(start_log, end_log)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Top 10%", "Top 1%", "Top 0.1%"])
        ax.grid(True, linestyle='-', alpha=0.2)
        
        # --- 6. Bias Text Box ---
        idx_1pct = np.abs(x_log - 2.0).argmin() 
        idx_01pct = np.abs(x_log - 3.0).argmin() 
        
        gap_1 = true_c[idx_1pct] - rep_c[idx_1pct]
        gap_01 = true_c[idx_01pct] - rep_c[idx_01pct]
        
        text_str = (
            f"Gap (True - Rep):\n"
            f"Top 1%: {gap_1:+.2%}\n"
            f"Top 0.1%: {gap_01:+.2%}"
        )
        
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', 
                          alpha=0.9, edgecolor='gray')
        
        ax.text(0.03, 0.03, text_str, transform=ax.transAxes, fontsize=9, 
                verticalalignment='bottom', bbox=bbox_props)

    # Generate the 2x2 Grid
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    params = [
        (ax[0,0], 0.10, 0.4, "Beta=0.1, Sigma=0.4"),
        (ax[0,1], 0.10, 1.4, "Beta=0.1, Sigma=1.4"),
        (ax[1,0], -0.05, 0.4, "Beta=-0.05, Sigma=0.4"),
        (ax[1,1], -0.05, 1.4, "Beta=-0.05, Sigma=1.4")
    ]
    
    for axis, b, s, t in params:
        plot_panel(axis, b, s, t)
    
    plt.tight_layout()
    plt.savefig("Fig_ShareLines.pdf")
    plt.close()
    
def run_fig_evasion(results):
    print("Generating Evasion Profiles...")
    
    def plot_evasion(ax, beta, sigma, title):     
        # Normalize keys to standard Python floats to avoid KeyError
        b_key = float(np.round(beta, 2))
        s_key = float(np.round(sigma, 1))

        # Retrieve pre-computed data using normalized keys
        df = pd.DataFrame(results[(b_key, s_key)])
        
        # Define Log Bins (cutting off extreme outliers for smoothness)
        lower = df['True'].quantile(0.001)
        upper = df['True'].quantile(0.999)
        bins = np.logspace(np.log10(lower), np.log10(upper), 30)
        centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate Average Evasion in each bin
        # Split into separate variables for readability
        ev_true = df.groupby(
            pd.cut(df['True'], bins), 
            observed=False
        )['EvasionRate'].mean()
        
        ev_rep = df.groupby(
            pd.cut(df['Reported'], bins), 
            observed=False
        )['EvasionRate'].mean()
        
        # Plot Lines
        ax.plot(
            centers, 
            ev_true.values, 
            'b-', 
            label='vs True Income', 
            linewidth=2
        )
        ax.plot(
            centers, 
            ev_rep.values, 
            'g--', 
            label='vs Reported Income', 
            linewidth=2
        )
        
        ax.axhline(0.05, color='grey', alpha=0.5, linestyle=':', label='Baseline (5%)')
        
        # Formatting
        ax.set_xscale('log')
        ax.set_title(title)
        ax.set_xlabel("Income ($)")
        ax.set_ylabel("Average Evasion Rate")
        ax.set_xlim(lower, upper)
        ax.set_ylim(0, 0.55) 
        ax.legend()

    # Create Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_evasion(ax[0], 0.10, 1.4, "Beta=0.10, Sigma=1.4")
    plot_evasion(ax[1], -0.05, 1.4, "Beta=-0.05, Sigma=1.4")
    
    plt.tight_layout()
    plt.savefig("Fig_EvasionProfiles.pdf")
    plt.close()
    print("Saved to Fig_EvasionProfiles.pdf")
# =============================================================================
# PART 3: STANDALONE ROUTINES
# =============================================================================

def run_walkthrough_clean():
    print("Generating Walkthrough (Mean=$65k, Dual Evasion & Statistics)...")
    
    # --- 1. CONFIGURATION & CALIBRATION ---
    TARGET_MEAN = 65000
    BETA = 0.05
    SIGMA_NU = 1.4
    N = 5000000
    
    cal_sigma = tm.solve_for_reported_share(
        dist_type='lognormal', 
        beta=BETA, 
        sigma_nu=SIGMA_NU, 
        mode='loglinear', 
        z_type='log_income', 
        target=0.20, 
        n_agents=N
    )
    
    # Target MEAN = $65k: mu = ln(Mean) - sigma^2/2
    mu_mean = np.log(TARGET_MEAN) - (cal_sigma**2 / 2)
    
    # --- 2. GENERATE DATA ---
    np.random.seed(42)
    y_true = np.random.lognormal(mu_mean, cal_sigma, N)
    
    y_rep, ev_rates = tm.apply_evasion(
        y_true, 
        BETA, 
        SIGMA_NU, 
        mode='loglinear', 
        z_type='log_income', 
        seed=43
    )
    
    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PANEL A: Distributions & Cutoff Box
    ax1 = axes[0]
    sns.kdeplot(x=np.log(y_true), ax=ax1, color='#d62728', linewidth=2, label='True')
    sns.kdeplot(x=np.log(y_rep), ax=ax1, color='#1f77b4', linewidth=2, linestyle='--', label='Reported')
    
    l_mean = np.log(TARGET_MEAN)
    c_t = np.percentile(y_true, 99)
    c_r = np.percentile(y_rep, 99)
    
    ax1.axvline(l_mean, color='k', alpha=0.4)
    ax1.axvline(np.log(c_t), color='#d62728', linestyle=':', ymax=0.6)
    ax1.axvline(np.log(c_r), color='#1f77b4', linestyle=':', ymax=0.5)
    
    txt_box = (
        f"Top 1% Cutoff:\n"
        f"True: ${c_t:,.0f}\n"
        f"Reported: ${c_r:,.0f}\n"
        f"Gap: ${c_t - c_r:,.0f}"
    )
    
    ax1.text(
        0.95, 0.70, txt_box, transform=ax1.transAxes, fontsize=10, 
        ha='right', va='top', bbox=dict(boxstyle="round,pad=0.4", 
        facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    ax1.legend(loc='upper left')
    ax1.set_xlabel("Log Income")
    ax1.set_ylabel("Density of People")
    ax1.set_title("A. Distributions & Cutoffs")
    
    # PANEL B: Top Shares & Dual Evasion Lines
    ax2 = axes[1]
    ax2t = ax2.twinx()
    grid_pct = np.logspace(0, -2, 100) 
    
    tsort, rsort = np.sort(y_true), np.sort(y_rep)
    idx_r, idx_t = np.argsort(y_rep), np.argsort(y_true)
    ev_by_rep, ev_by_true = ev_rates[idx_r], ev_rates[idx_t]
    
    ts, rs, es_rep, es_true = [], [], [], []
    for p in grid_pct:
        k = max(int(N * (p/100)), 1)
        ts.append(tsort[-k:].sum() / tsort.sum())
        rs.append(rsort[-k:].sum() / rsort.sum())
        es_rep.append(ev_by_rep[-k:].mean())   # Selection
        es_true.append(ev_by_true[-k:].mean())  # Intensity

    ax2.plot(grid_pct, ts, color='#d62728', linewidth=2.5, label='True Share')
    ax2.plot(grid_pct, rs, color='#1f77b4', linewidth=2.5, linestyle='--', label='Reported Share')
    
    ax2t.plot(grid_pct, es_rep, color='green', linewidth=2, linestyle=':', 
              label='Avg Evasion (Reported Top %)')
    ax2t.plot(grid_pct, es_true, color='darkgreen', linewidth=1.5, linestyle='-.', 
              label='Avg Evasion (True Top %)')
    
    ax2.set_xscale('log')
    ax2.invert_xaxis()
    ax2.set_xticks([1, 0.1, 0.01])
    ax2.set_xticklabels(["1%", "0.1%", "0.01%"])
    ax2.set_xlabel("Top Percentile")
    ax2.set_ylabel("Cumulative Income Share")
    
    ax2t.set_ylabel("Average Evasion Rate", color='green')
    ax2t.tick_params(axis='y', labelcolor='green')
    ax2t.set_ylim(0, 0.15) 
    ax2t.set_title("B. Top Shares & Evasion Intensity")
    
    # Combined Legend
    lines, lbls = ax2.get_legend_handles_labels()
    l2, lb2 = ax2t.get_legend_handles_labels()
    ax2.legend(lines + l2, lbls + lb2, loc='lower left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("Fig_Walkthrough_Clean.pdf")
    plt.show()
    plt.close()

    # --- 4. PRINT DIAGNOSTICS ---
    print("\n" + "="*35)
    print(f"{'WALKTHROUGH SCENARIO DIAGNOSTICS':^35}")
    print("="*35)
    
    top1_true_idx = np.where(y_true >= c_t)[0]
    top1_rep_idx = np.where(y_rep >= c_r)[0]
    overlap = len(np.intersect1d(top1_true_idx, top1_rep_idx))
    reranking_rate = 1 - (overlap / len(top1_true_idx))
    
    # FIXED: Compute ranks on full population, then subset
    all_rep_ranks = pd.Series(y_rep).rank(pct=True)
    median_rank = all_rep_ranks.iloc[top1_true_idx].median() * 100

    print(f"{'Re-ranking Rate:':<25} {reranking_rate:>8.1%}")
    print(f"{'Avg Evasion (True Top 1%):':<25} {ev_rates[top1_true_idx].mean():>8.1%}")
    print(f"{'Avg Evasion (Rep Top 1%):':<25} {ev_rates[top1_rep_idx].mean():>8.1%}")
    print(f"{'Median Rep. Rank True Rich:':<25} P{median_rank:>7.1f}")
    print("="*35 + "\n")
    
    
def run_robustness_figs():
    print("Generating Robustness Figs 6 & 7 (Pareto & Additive)...")
    
    # --- 1. SETUP GRID ---
    beta_vals = np.round(np.arange(-0.10, 0.11, 0.05), 2)
    sigma_vals = np.round(np.arange(0.0, 1.8, 0.2), 1)
    rows, cols = len(beta_vals), len(sigma_vals)
    N_AGENTS = 2000000
    
    # Pre-allocate Matrices
    log_add_1pct = np.zeros((rows, cols))
    log_add_01pct = np.zeros((rows, cols))
    par_mult_1pct = np.zeros((rows, cols))
    par_add_1pct = np.zeros((rows, cols))
    
    # --- 2. SIMULATION LOOPS ---
    for i, beta in enumerate(beta_vals):
        for j, snu in enumerate(sigma_vals):
            print(f"  Simulating cell: Beta={beta}, Sigma={snu}...", end="\r")
            
            # Scenario A: LOGNORMAL ADDITIVE
            df_la, _ = tm.get_calibrated_scenario(
                'lognormal', beta=beta, sigma_nu=snu, mode='additive', n_agents=N_AGENTS
            )
            k1, k01 = int(len(df_la)*0.01), int(len(df_la)*0.001)
            
            log_add_1pct[i,j] = (df_la.nlargest(k1, 'True')['True'].sum() / df_la['True'].sum()) - \
                                (df_la.nlargest(k1, 'Reported')['Reported'].sum() / df_la['Reported'].sum())
            log_add_01pct[i,j] = (df_la.nlargest(k01, 'True')['True'].sum() / df_la['True'].sum()) - \
                                 (df_la.nlargest(k01, 'Reported')['Reported'].sum() / df_la['Reported'].sum())

            # Scenario B: PARETO MULTIPLICATIVE
            df_pm, _ = tm.get_calibrated_scenario(
                'pareto', beta=beta, sigma_nu=snu, mode='loglinear', n_agents=N_AGENTS
            )
            k1p = int(len(df_pm)*0.01)
            par_mult_1pct[i,j] = (df_pm.nlargest(k1p, 'True')['True'].sum() / df_pm['True'].sum()) - \
                                 (df_pm.nlargest(k1p, 'Reported')['Reported'].sum() / df_pm['Reported'].sum())
                                
            # Scenario C: PARETO ADDITIVE
            df_pa, _ = tm.get_calibrated_scenario(
                'pareto', beta=beta, sigma_nu=snu, mode='additive', n_agents=N_AGENTS
            )
            par_add_1pct[i,j] = (df_pa.nlargest(k1p, 'True')['True'].sum() / df_pa['True'].sum()) - \
                                (df_pa.nlargest(k1p, 'Reported')['Reported'].sum() / df_pa['Reported'].sum())

    # --- 3. PLOTTING ---
    y_lbl = [f"{b:.2f}" for b in np.flip(beta_vals)]
    x_lbl = [f"{s:.1f}" for s in sigma_vals]
    heat_kw = dict(annot=True, fmt=".1%", cmap="RdBu_r", center=0, cbar=False, xticklabels=x_lbl)

    # Fig 6: Additive
    fig6, ax6 = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(np.flipud(log_add_1pct), ax=ax6[0], yticklabels=y_lbl, **heat_kw)
    ax6[0].set_title("A. Additive: Top 1% Gap")
    sns.heatmap(np.flipud(log_add_01pct), ax=ax6[1], yticklabels=False, **heat_kw)
    ax6[1].set_title("B. Additive: Top 0.1% Gap")
    plt.tight_layout(); plt.savefig("Fig6_Robustness_Additive.pdf", bbox_inches='tight'); plt.close()

    # Fig 7: Pareto
    fig7, ax7 = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(np.flipud(par_mult_1pct), ax=ax7[0], yticklabels=y_lbl, **heat_kw)
    ax7[0].set_title("A. Pareto Multiplicative")
    sns.heatmap(np.flipud(par_add_1pct), ax=ax7[1], yticklabels=False, **heat_kw)
    ax7[1].set_title("B. Pareto Additive")
    plt.tight_layout(); plt.savefig("Fig7_Robustness_Pareto.pdf", bbox_inches='tight'); plt.close()

def run_alpha_robustness():
    print("Generating Calibrated Pareto Alpha Heatmaps (Fixed Axes & Colors)...")
    
    # --- 1. SETUP GRID ---
    beta_vals = np.round(np.arange(-0.10, 0.11, 0.05), 2)
    sigma_nu_vals = np.round(np.arange(0.0, 1.8, 0.2), 1)
    N_AGENTS = 5000000  # High resolution for stable Alpha
    
    alpha_mult = np.zeros((len(beta_vals), len(sigma_nu_vals)))
    alpha_add = np.zeros((len(beta_vals), len(sigma_nu_vals)))
    
    # --- 2. RUN SIMULATIONS ---
    for i, beta in enumerate(beta_vals):
        for j, snu in enumerate(sigma_nu_vals):
            print(f"  Alpha Cell: Beta={beta}, Sigma={snu}...", end="\r")
            
            # Multiplicative (Log-Linear)
            _, am = tm.get_calibrated_scenario(
                'pareto', beta=beta, sigma_nu=snu, mode='loglinear', n_agents=N_AGENTS
            )
            alpha_mult[i,j] = am
            
            # Additive
            _, aa = tm.get_calibrated_scenario(
                'pareto', beta=beta, sigma_nu=snu, mode='additive', n_agents=N_AGENTS
            )
            alpha_add[i,j] = aa

    # --- 3. PLOTTING ---
    y_labels = [f"{b:.2f}" for b in np.flip(beta_vals)]
    x_labels = [f"{s:.1f}" for s in sigma_nu_vals]
    
    # Shared Heatmap Configuration
    alpha_kw = dict(
        annot=True, 
        fmt=".2f", 
        cmap="viridis", 
        cbar=False,
        xticklabels=x_labels, 
        yticklabels=y_labels
    )
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Multiplicative
    sns.heatmap(np.flipud(alpha_mult), ax=ax[0], **alpha_kw)
    ax[0].set_title("A. Pareto Alpha (Multiplicative)")
    ax[0].set_ylabel("Beta (Progressivity)")
    ax[0].set_xlabel("Sigma (Evasion Heterogeneity)")
    
    # Panel B: Additive
    sns.heatmap(np.flipud(alpha_add), ax=ax[1], **alpha_kw)
    ax[1].set_yticks([]) # Hide redundant Y-labels
    ax[1].set_title("B. Pareto Alpha (Additive)")
    ax[1].set_xlabel("Sigma (Evasion Heterogeneity)")
    
    plt.tight_layout()
    plt.savefig("Fig_ParetoAlpha_Robustness.pdf", bbox_inches='tight')
    plt.close()
    print("Saved to Fig_ParetoAlpha_Robustness.pdf")

def run_extreme_diagnostics():
    print("Generating Extreme Diagnostics (Dual Evasion & Statistics)...")
    
    # --- 1. CONFIGURATION ---
    BETA, SIGMA_NU, CAP, N = 0.5, 8.0, 0.99, 2000000
    T_MEAN, T_SHARE = 65000, 0.20
    
   # --- 2. CALIBRATION ---
    print("  Calibrating sigma for extreme scenario...")
    low, high, cal_sigma = 0.1, 8.0, 1.0
    
    for i in range(25):
        guess = (low + high) / 2
        mu_temp = np.log(T_MEAN) - (guess**2 / 2)
        
        # FIXED: Seed the draw to remove Monte Carlo noise
        np.random.seed(1234) 
        y_temp = np.random.lognormal(mu_temp, guess, 100000)
        
        yr_temp, _ = tm.apply_evasion_extreme(
            y_temp, 
            BETA, 
            SIGMA_NU, 
            cap=CAP, 
            seed=42
        )
        
        # Calculate share on the stable sample
        share = np.sort(yr_temp)[-1000:].sum() / yr_temp.sum()
        
        if abs(share - T_SHARE) < 0.002:
            cal_sigma = guess
            break
        elif share < T_SHARE: 
            low = guess
        else: 
            high = guess
            
    # --- 3. GENERATE FULL DATA ---
    mu = np.log(T_MEAN) - (cal_sigma**2 / 2)
    np.random.seed(42)
    yt = np.random.lognormal(mu, cal_sigma, N)
    yr, ev = tm.apply_evasion_extreme(yt, BETA, SIGMA_NU, cap=CAP, seed=43)
    yun = yt - yr 

    # --- 4. PLOTTING ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Densities
    ax1 = ax[0]
    sns.kdeplot(x=np.log(yt), ax=ax1, color='#d62728', lw=3, label='True')
    sns.kdeplot(x=np.log(np.maximum(yr, 1)), ax=ax1, color='#1f77b4', lw=3, ls='--', label='Reported')
    sns.kdeplot(x=np.log(np.maximum(yr, 1)), weights=yun, ax=ax1, color='purple', lw=3, ls=':', label='Unreported $')
    ax1.set_title("A. Distributions & Unreported $"); ax1.legend(loc='upper left')

    # Panel B: Dual Evasion Lines
    ax2, ax2t = ax[1], ax[1].twinx()
    grid = np.logspace(0, -2, 50)
    idx_r, idx_t = np.argsort(yr), np.argsort(yt)
    tsort = np.sort(yt); rsort = yr[idx_r]
    ev_by_rep, ev_by_true = ev[idx_r], ev[idx_t]
    
    ts, rs, es_rep, es_true = [], [], [], []
    for p in grid:
        k = max(int(N * (p/100)), 100)
        ts.append(tsort[-k:].sum() / tsort.sum())
        rs.append(rsort[-k:].sum() / rsort.sum())
        es_rep.append(ev_by_rep[-k:].mean())
        es_true.append(ev_by_true[-k:].mean())

    ax2.plot(grid, ts, color='#d62728', lw=3, label='True Share')
    ax2.plot(grid, rs, color='#1f77b4', lw=3, ls='--', label='Rep Share')
    ax2t.plot(grid, es_rep, color='green', lw=2, ls=':', label='Evasion (Top Rep %)')
    ax2t.plot(grid, es_true, color='darkgreen', lw=1.5, ls='-.', label='Evasion (Top True %)')
    
    ax2.set_xscale('log'); ax2.invert_xaxis()
    ax2.set_xticks([1, 0.1, 0.01]); ax2.set_xticklabels(["1%", "0.1%", "0.01%"])
    ax2t.set_ylabel("Avg Evasion Rate", color='green'); ax2t.set_ylim(0, 0.40)
    ax2t.tick_params(axis='y', labelcolor='green')
    
    lines, lbls = ax2.get_legend_handles_labels()
    l2, lb2 = ax2t.get_legend_handles_labels()
    ax2.legend(lines+l2, lbls+lb2, loc='upper right', fontsize=8)
    ax2.set_title("B. Top Shares & Evasion Intensity")
    
    plt.tight_layout(); plt.savefig("Fig_Extreme.pdf"); plt.close()

    # --- 5. PRINT DIAGNOSTICS ---
    print("\n" + "="*35)
    print(f"{'EXTREME SCENARIO DIAGNOSTICS':^35}")
    print("="*35)
    
    ct, cr = np.percentile(yt, 99), np.percentile(yr, 99)
    top1t, top1r = np.where(yt >= ct)[0], np.where(yr >= cr)[0]
    rerank = 1 - (len(np.intersect1d(top1t, top1r)) / len(top1t))
    
    # FIXED: Compute ranks on full population, then subset
    all_rep_ranks = pd.Series(yr).rank(pct=True)
    median_rank = all_rep_ranks.iloc[top1t].median() * 100
    
    print(f"{'Top 1% Cutoff (True):':<25} ${ct:>10,.0f}")
    print(f"{'Top 1% Cutoff (Rep):':<25} ${cr:>10,.0f}")
    print(f"{'Re-ranking Rate:':<25} {rerank:>11.1%}")
    print(f"{'Median Rep Rank True Rich:':<25} P{median_rank:>10.1f}")
    print("="*35 + "\n")

def run_fixed_true_robustness():
    print("Generating Robustness Check: Fixed True Inequality...")

    # --- 1. ESTABLISH BASELINE POPULATION ---
    N_AGENTS = 2000000
    print("  Calibrating baseline True Distribution (Beta=0, Sigma=0)...")
    
    baseline_sigma = tm.solve_for_reported_share(
        dist_type='lognormal', 
        beta=0.0, 
        sigma_nu=0.0, 
        mode='loglinear', 
        z_type='log_income', 
        target=0.20, 
        n_agents=N_AGENTS
    )
    
    y_true_fixed = tm.generate_true_income(
        N_AGENTS, 
        'lognormal', 
        baseline_sigma, 
        seed=2026
    )
    
    k = int(N_AGENTS * 0.01)
    
    # --- 2. SWEEP PARAMETERS ---
    beta_vals = np.round(np.arange(-0.10, 0.11, 0.05), 2)
    sigma_nu_vals = np.round(np.arange(0.0, 1.8, 0.2), 1)
    rep_map = np.zeros((len(beta_vals), len(sigma_nu_vals)))

    for i, beta in enumerate(beta_vals):
        for j, snu in enumerate(sigma_nu_vals):
            print(f"  Fixed True Cell: Beta={beta}, Sigma={snu}...", end="\r")
            
            y_rep, _ = tm.apply_evasion(
                y_true_fixed, 
                beta=beta, 
                sigma_nu=snu, 
                mode='loglinear', 
                z_type='log_income', 
                seed=999
            )
            
            # Calculate Reported Share of the Top 1%
            rep_top_sum = np.sort(y_rep)[-k:].sum()
            rep_map[i,j] = rep_top_sum / y_rep.sum()

    # --- 3. PLOTTING ---
    plt.figure(figsize=(9, 7))
    
    y_lbl = [f"{b:.2f}" for b in np.flip(beta_vals)]
    
    sns.heatmap(
        np.flipud(rep_map), 
        annot=True, 
        fmt=".1%", 
        cmap="RdBu", 
        center=0.20, 
        cbar=False, 
        xticklabels=sigma_nu_vals, 
        yticklabels=y_lbl
    )
    
    plt.title(
        f"Reported Top 1% Share\n"
        f"(Holding True Inequality Fixed)"
    )
    plt.xlabel("Sigma (Evasion Heterogeneity)")
    plt.ylabel("Beta (Progressivity)")
    
    plt.tight_layout()
    plt.savefig("Fig_FixedTrue_Robustness.pdf")
    plt.close()
    print("Saved to Fig_FixedTrue_Robustness.pdf")
    
    
def run_fixed_true_gap_heatmap():
    print("Generating Robustness: Inequality Gap with Fixed True Population...")

    # --- 1. ESTABLISH STABLE BASELINE ---
    N_AGENTS = 2000000
    # Calibrate once at the neutral point (Beta=0, Sigma=0)
    baseline_sigma = tm.solve_for_reported_share(
        dist_type='lognormal', 
        beta=0.0, 
        sigma_nu=0.0, 
        mode='loglinear', 
        z_type='log_income', 
        target=0.20, 
        n_agents=N_AGENTS
    )
    
    y_true_fixed = tm.generate_true_income(N_AGENTS, 'lognormal', baseline_sigma, seed=2026)
    
    # Calculate the fixed 1% True Share (should be approx 20%)
    k = int(N_AGENTS * 0.01)
    true_share_fixed = y_true_fixed[np.argsort(y_true_fixed)[-k:]].sum() / y_true_fixed.sum()
    
    # --- 2. SWEEP PARAMETERS ---
    beta_vals = np.round(np.arange(-0.10, 0.11, 0.05), 2)
    sigma_nu_vals = np.round(np.arange(0.0, 1.8, 0.2), 1)
    gap_map = np.zeros((len(beta_vals), len(sigma_nu_vals)))

    for i, beta in enumerate(beta_vals):
        for j, snu in enumerate(sigma_nu_vals):
            # Apply evasion to the fixed population
            y_rep, _ = tm.apply_evasion(
                y_true_fixed, beta=beta, sigma_nu=snu, 
                mode='loglinear', z_type='log_income', seed=999
            )
            
            # Calculate Reported Top 1% Share
            rep_share = np.sort(y_rep)[-k:].sum() / y_rep.sum()
            
            # The Gap: True Share (Fixed) - Reported Share (Variable)
            gap_map[i, j] = true_share_fixed - rep_share

    # --- 3. PLOTTING ---
    plt.figure(figsize=(10, 8))
    y_lbl = [f"{b:.2f}" for b in np.flip(beta_vals)]
    
    sns.heatmap(
        np.flipud(gap_map), 
        annot=True, 
        fmt=".1%", 
        cmap="RdBu_r", # Red = Understatement, Blue = Overstatement
        center=0, 
        xticklabels=sigma_nu_vals, 
        yticklabels=y_lbl,
        cbar= False
    )
    
    plt.title(f"Fixed True Share = {true_share_fixed:.1%}\n"
              r"Calibration: $\beta$ = $\sigma_{\nu}$ = 0")
    plt.xlabel("Sigma (Evasion Heterogeneity)")
    plt.ylabel("Beta (Progressivity)")
    
    plt.tight_layout()
    plt.savefig("Fig_Robustness_FixedTrue_Gap.pdf")
    plt.show()
    plt.close()
    print("Saved to Fig_Robustness_FixedTrue_Gap.pdf")
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    #beta_vals = np.round(np.arange(-0.10, 0.11, 0.05), 2); sigma_vals = np.round(np.arange(0.0, 1.8, 0.2), 1)
    #grid_data = generate_grid_data(beta_vals, sigma_vals, n_agents=2000000)
    #run_table1(grid_data)
    #run_all_heatmaps(grid_data, beta_vals, sigma_vals)
    #run_fig_lines(grid_data)
    #run_fig_evasion(grid_data)
    #run_walkthrough_clean()
    #run_robustness_figs()
    #run_alpha_robustness()
    #run_extreme_diagnostics()
    #run_fixed_true_robustness()
    run_fixed_true_gap_heatmap()
    print("\n=== MASTER ANALYSIS COMPLETE ===")