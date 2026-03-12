import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # Fixes the "xcb" / WSL error
import matplotlib.pyplot as plt

def analyze_run(name):
    filename = f"{name}_final_state.npz"
    if not os.path.exists(filename):
        print(f"Skipping {name}: File not found.")
        return

    data = np.load(filename)
    
    # 1. Extraction for Questions 1, 2, 5
    g_final = data["grad_time"][-1]
    s_final = data["sat_time"][-1]
    g_final = g_final[np.isfinite(g_final)]
    s_final = s_final[np.isfinite(s_final)]
    
    rho_whh = data["rho_Whh"]
    total_grad_norm = data["gradient_norm"]

    # --- PLOT A: Question 1 (Vanishing/Exploding) ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(np.log10(g_final + 1e-12), bins=60, color='blue', alpha=0.7)
    plt.title(rf"Q1: $\log_{{10}} ||\partial \mathcal{{L}} / \partial h_t||$ - {name}")
    plt.xlabel("Log Norm (< -10 = Vanishing)")

    # --- PLOT B: Question 2 (Saturation) ---
    plt.subplot(1, 2, 2)
    plt.hist(s_final, bins=60, range=(0, 1), color='orange', alpha=0.7)
    plt.title(f"Q2: Saturation Distance - {name}")
    plt.xlabel("Distance (0=Sat, 1=Active)")
    plt.savefig(f"{name}_Q1_Q2.png")
    plt.close()

    # --- PLOT C: Question 5 (Spectral Radius) ---
    plt.figure()
    plt.plot(rho_whh, label=r"$\rho(W_{hh})$")
    plt.axhline(y=1.0, color='r', linestyle='--', label="Explosion Threshold")
    plt.title(f"Q5: Spectral Radius vs Training - {name}")
    plt.ylabel("Radius")
    plt.legend()
    plt.savefig(f"{name}_Q5_Spectral.png")
    plt.close()

    # --- PLOT D: Question 4 (GRU Gates Only) ---
    if "gate_z_sat_time" in data:
        z_gate = data["gate_z_sat_time"][-1]
        r_gate = data["gate_r_sat_time"][-1]
        z_gate, r_gate = z_gate[np.isfinite(z_gate)], r_gate[np.isfinite(r_gate)]
        
        plt.figure()
        plt.hist(z_gate, bins=60, range=(0,1), alpha=0.5, label='Update (z)')
        plt.hist(r_gate, bins=60, range=(0,1), alpha=0.5, label='Reset (r)')
        plt.title(f"Q4: GRU Gate Saturation - {name}")
        plt.legend()
        plt.savefig(f"{name}_Q4_Gates.png")
        plt.close()

# List of experiments
runs = ["A1_mem_rnn_tanh_noclip", "A2_mem_rnn_tanh_clip005", "A3_mem_rnn_tanh_clip001", "A4_mem_gru_noclip", "A5_mem_gru_clip005", "B1_mul_rnn_tanh_noclip", "B2_mul_rnn_tanh_noclip"]
for r in runs:
    analyze_run(r)