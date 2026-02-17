"""
CORE ENGINE: tax_model.py
Contains all logic for generating income distributions, applying evasion,
and calibrating parameters.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# --- CONFIGURATION ---
DEFAULT_N = 10000000
TARGET_REPORTED_SHARE = 0.199
TOLERANCE = 0.00005
BASE_EVASION = 0.05
LOGLINEAR_BASE = 0.05
MAX_EVASION = 0.95

def generate_true_income(n, dist_type, inequality_param, seed=None):
    if seed is not None: np.random.seed(seed)
    if dist_type == 'lognormal':
        return np.random.lognormal(mean=10.5, sigma=inequality_param, size=n)
    elif dist_type == 'pareto':
        # Shifted Pareto (Lomax) so min income is 1.0
        return (np.random.pareto(inequality_param, size=n) + 1)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

def get_z_score(y_true, z_type):
    if z_type == 'log_income':
        log_y = np.log(y_true)
        if log_y.std() == 0: return np.zeros_like(log_y)
        return (log_y - log_y.mean()) / log_y.std()
    elif z_type == 'rank':
        n = len(y_true)
        ranks = np.argsort(np.argsort(y_true)) + 1
        return norm.ppf(ranks / (n + 1))
    else: raise ValueError(f"Unknown z_type: {z_type}")

def apply_evasion(y_true, beta, sigma_nu, mode='loglinear', z_type='log_income', seed=None):
    if seed is not None: np.random.seed(seed)
    z = get_z_score(y_true, z_type)
    noise = np.random.normal(0, 1, len(y_true))
    
    if mode == 'additive':
        raw_evasion = BASE_EVASION + (beta * z) + (sigma_nu * noise)
    elif mode == 'loglinear':
        raw_evasion = LOGLINEAR_BASE * np.exp((beta * z) + (sigma_nu * noise))
    
    evasion_rate = np.clip(raw_evasion, 0.0, MAX_EVASION)
    y_reported = y_true * (1 - evasion_rate)
    return y_reported, evasion_rate

def apply_evasion_extreme(y_true, beta, sigma_nu, cap=0.99, seed=None):
    """
    Applies evasion with a custom cap (0.99) to allow for extreme hiding behavior.
    Used for the Diagnostics/Extreme scenario.
    """
    if seed is not None: np.random.seed(seed)
    
    # Calculate Z-score (log income)
    log_y = np.log(y_true)
    if log_y.std() == 0: z = np.zeros_like(log_y)
    else: z = (log_y - log_y.mean()) / log_y.std()
    
    noise = np.random.normal(0, 1, len(y_true))
    
    # Log-linear specification
    # Note: Base is hardcoded to 0.05 as per original script
    base_evasion = 0.05 
    raw_evasion = base_evasion * np.exp((beta * z) + (sigma_nu * noise))
    
    # CLIP AT CUSTOM CAP (e.g. 0.99)
    evasion_rate = np.clip(raw_evasion, 0.0, cap)
    
    y_rep = y_true * (1 - evasion_rate)
    return y_rep, evasion_rate

def solve_for_reported_share(dist_type, beta, sigma_nu, mode, z_type, target=TARGET_REPORTED_SHARE, n_agents=DEFAULT_N):
    if dist_type == 'lognormal': 
        low, high = 0.1, 4.0
    else: 
        low, high = 1.001, 8.0 
        
    solved_param = (low + high) / 2
    SOLVER_SEED = 42
    
    for i in range(50):
        guess = (low + high) / 2
        y_true = generate_true_income(n_agents, dist_type, guess, seed=SOLVER_SEED)
        y_rep, _ = apply_evasion(y_true, beta, sigma_nu, mode=mode, z_type=z_type, seed=SOLVER_SEED+1)
        k = int(n_agents * 0.01)
        rep_share = np.sort(y_rep)[-k:].sum() / y_rep.sum()
        
        if abs(rep_share - target) < TOLERANCE:
            solved_param = guess
            break
        
        if dist_type == 'lognormal':
            if rep_share < target: low = guess
            else: high = guess
        else:
            if rep_share < target: high = guess 
            else: low = guess
            
    return solved_param

def get_calibrated_scenario(dist_type, beta, sigma_nu, mode='loglinear', z_type='log_income', n_agents=DEFAULT_N, seed=1000):
    calibrated_param = solve_for_reported_share(dist_type, beta, sigma_nu, mode, z_type, n_agents=n_agents)
    y_true = generate_true_income(n_agents, dist_type, calibrated_param, seed=seed)
    y_rep, ev_rate = apply_evasion(y_true, beta, sigma_nu, mode=mode, z_type=z_type, seed=seed+1)
    return pd.DataFrame({'True': y_true, 'Reported': y_rep, 'EvasionRate': ev_rate}), calibrated_param