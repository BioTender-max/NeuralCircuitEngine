
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, shutil

OUT_DIR = '/mnt/shared-workspace/shared'
os.makedirs(OUT_DIR, exist_ok=True)

N_NEURONS = 500
P_CONNECT = 0.10
N_EXC = int(N_NEURONS * 0.8)
N_INH = N_NEURONS - N_EXC

# ── Connectivity matrix (sparse) ──────────────────────────────────────────────
conn_matrix = (np.random.random((N_NEURONS, N_NEURONS)) < P_CONNECT).astype(float)
np.fill_diagonal(conn_matrix, 0)
# Excitatory neurons: positive weights; inhibitory: negative
weights = conn_matrix.copy()
weights[:N_EXC, :] *= np.random.exponential(0.5, (N_EXC, N_NEURONS))
weights[N_EXC:, :] *= -np.random.exponential(0.5, (N_INH, N_NEURONS))

# ── Interneuron classification ────────────────────────────────────────────────
inh_neurons = np.arange(N_EXC, N_NEURONS)
# PV: fast-spiking (high firing rate, low CV)
# SST: regular-spiking (medium rate, medium CV)
# VIP: irregular (low rate, high CV)
n_pv = int(N_INH * 0.4)
n_sst = int(N_INH * 0.35)
n_vip = N_INH - n_pv - n_sst
interneuron_types = np.array(['PV'] * n_pv + ['SST'] * n_sst + ['VIP'] * n_vip)
firing_rates_inh = np.concatenate([
    np.random.normal(60, 10, n_pv),
    np.random.normal(30, 8, n_sst),
    np.random.normal(15, 5, n_vip)
])
cv_inh = np.concatenate([
    np.random.normal(0.3, 0.05, n_pv),
    np.random.normal(0.6, 0.1, n_sst),
    np.random.normal(1.2, 0.2, n_vip)
])

# ── Dopamine neuron identification ────────────────────────────────────────────
d1_expr = np.random.lognormal(1, 1, N_EXC)
d2_expr = np.random.lognormal(1, 1, N_EXC)
# D1 neurons: high D1, low D2
d1_neurons = (d1_expr > np.percentile(d1_expr, 70)) & (d2_expr < np.percentile(d2_expr, 40))
d2_neurons = (d2_expr > np.percentile(d2_expr, 70)) & (d1_expr < np.percentile(d1_expr, 40))

# ── Connectome motifs ─────────────────────────────────────────────────────────
# Reciprocal: i→j and j→i
reciprocal = 0
chain = 0
fan_out = 0
sample_n = 200
for _ in range(sample_n):
    i, j, k = np.random.choice(N_NEURONS, 3, replace=False)
    if conn_matrix[i, j] and conn_matrix[j, i]:
        reciprocal += 1
    if conn_matrix[i, j] and conn_matrix[j, k] and not conn_matrix[i, k]:
        chain += 1
    if conn_matrix[i, j] and conn_matrix[i, k] and not conn_matrix[j, k]:
        fan_out += 1

# ── Spike train statistics ────────────────────────────────────────────────────
firing_rates_exc = np.abs(np.random.normal(15, 8, N_EXC))
all_firing_rates = np.concatenate([firing_rates_exc, firing_rates_inh])
# ISI distribution (exponential for Poisson neurons)
isi_samples = np.random.exponential(1.0 / (all_firing_rates.mean() / 1000), 2000)  # ms
# CV of firing
cv_exc = np.abs(np.random.normal(0.8, 0.2, N_EXC))
cv_all = np.concatenate([cv_exc, cv_inh])
# Fano factor
fano = np.abs(np.random.normal(1.2, 0.4, N_NEURONS))

# ── E/I balance ───────────────────────────────────────────────────────────────
exc_input = np.random.exponential(2, N_NEURONS)
inh_input = np.random.exponential(1.5, N_NEURONS)
ei_ratio = exc_input / (inh_input + 1e-6)

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Neural Circuit Engine — 500 Neurons (80% Exc / 20% Inh)',
             color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1: Connectivity matrix (subsample)
ax = axes[0, 0]
sub_n = 100
im = ax.imshow(conn_matrix[:sub_n, :sub_n], cmap='Blues', aspect='auto', vmin=0, vmax=1)
ax.axhline(N_EXC * sub_n / N_NEURONS, color='#f78166', lw=1.5, linestyle='--')
ax.axvline(N_EXC * sub_n / N_NEURONS, color='#f78166', lw=1.5, linestyle='--')
style_ax(ax, f'Connectivity Matrix (first {sub_n} neurons)', 'Neuron j', 'Neuron i')

# Panel 2: Interneuron classification
ax = axes[0, 1]
colors_int = {'PV': '#f78166', 'SST': '#3fb950', 'VIP': '#d2a8ff'}
for itype in ['PV', 'SST', 'VIP']:
    mask = interneuron_types == itype
    ax.scatter(firing_rates_inh[mask], cv_inh[mask], color=colors_int[itype],
               alpha=0.7, s=30, label=f'{itype} (n={mask.sum()})', edgecolors='none')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Interneuron Subtype Classification', 'Firing Rate (Hz)', 'CV of ISI')

# Panel 3: Dopamine neuron markers
ax = axes[0, 2]
ax.scatter(d1_expr[~d1_neurons & ~d2_neurons], d2_expr[~d1_neurons & ~d2_neurons],
           color='#58a6ff', alpha=0.3, s=10, label='Other', edgecolors='none')
ax.scatter(d1_expr[d1_neurons], d2_expr[d1_neurons],
           color='#3fb950', alpha=0.8, s=25, label=f'D1 ({d1_neurons.sum()})', edgecolors='none')
ax.scatter(d1_expr[d2_neurons], d2_expr[d2_neurons],
           color='#f78166', alpha=0.8, s=25, label=f'D2 ({d2_neurons.sum()})', edgecolors='none')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Dopamine Neuron Identification', 'D1 Receptor Expression', 'D2 Receptor Expression')

# Panel 4: Connectome motifs
ax = axes[1, 0]
motif_names = ['Reciprocal', 'Chain', 'Fan-out']
motif_counts = [reciprocal, chain, fan_out]
motif_colors = ['#58a6ff', '#3fb950', '#ffa657']
bars = ax.bar(motif_names, motif_counts, color=motif_colors, edgecolor='#0d1117', linewidth=0.5)
for bar, val in zip(bars, motif_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', color='white', fontsize=10)
style_ax(ax, f'Connectome Motifs (from {sample_n} samples)', 'Motif Type', 'Count')

# Panel 5: ISI distribution
ax = axes[1, 1]
ax.hist(isi_samples[isi_samples < 200], bins=50, color='#79c0ff', alpha=0.85,
        edgecolor='#0d1117', linewidth=0.3)
ax.axvline(np.median(isi_samples), color='#f78166', lw=2, linestyle='--',
           label=f'Median={np.median(isi_samples):.1f} ms')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'ISI Distribution', 'Inter-Spike Interval (ms)', 'Count')

# Panel 6: CV of firing
ax = axes[1, 2]
ax.hist(cv_all, bins=40, color='#56d364', alpha=0.85, edgecolor='#0d1117', linewidth=0.3)
ax.axvline(1.0, color='#ffa657', lw=1.5, linestyle='--', label='Poisson (CV=1)')
ax.axvline(cv_all.mean(), color='#f78166', lw=2, linestyle='--',
           label=f'Mean={cv_all.mean():.2f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'CV of Firing Rate Distribution', 'Coefficient of Variation', 'Count')

# Panel 7: Fano factor
ax = axes[2, 0]
ax.hist(fano, bins=40, color='#d2a8ff', alpha=0.85, edgecolor='#0d1117', linewidth=0.3)
ax.axvline(1.0, color='#ffa657', lw=1.5, linestyle='--', label='Poisson (Fano=1)')
ax.axvline(fano.mean(), color='#f78166', lw=2, linestyle='--',
           label=f'Mean={fano.mean():.2f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Fano Factor Distribution', 'Fano Factor', 'Count')

# Panel 8: E/I balance
ax = axes[2, 1]
ax.hist(ei_ratio, bins=40, color='#ffa657', alpha=0.85, edgecolor='#0d1117', linewidth=0.3)
ax.axvline(ei_ratio.mean(), color='#f78166', lw=2, linestyle='--',
           label=f'Mean E/I={ei_ratio.mean():.2f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'E/I Balance Distribution', 'E/I Ratio', 'Count')

# Panel 9: Summary
ax = axes[2, 2]
ax.set_facecolor('#161b22')
ax.axis('off')
n_connections = int(conn_matrix.sum())
summary_text = (
    f"  Neural Circuit Summary\n"
    f"  {'─'*32}\n"
    f"  Total neurons:         {N_NEURONS}\n"
    f"  Excitatory:            {N_EXC} (80%)\n"
    f"  Inhibitory:            {N_INH} (20%)\n"
    f"  PV interneurons:       {n_pv}\n"
    f"  SST interneurons:      {n_sst}\n"
    f"  VIP interneurons:      {n_vip}\n"
    f"  Total connections:     {n_connections}\n"
    f"  Connection prob:       {n_connections/(N_NEURONS*(N_NEURONS-1)):.3f}\n"
    f"  D1 neurons:            {d1_neurons.sum()}\n"
    f"  D2 neurons:            {d2_neurons.sum()}\n"
    f"  Mean firing rate:      {all_firing_rates.mean():.1f} Hz\n"
    f"  Mean CV:               {cv_all.mean():.3f}\n"
    f"  Mean Fano factor:      {fano.mean():.3f}\n"
    f"  Mean E/I ratio:        {ei_ratio.mean():.3f}\n"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        color='#e6edf3', bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))
ax.set_title('Summary Statistics', color='white', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = f'{OUT_DIR}/neural_circuit_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {out_png}")

print("\n=== NeuralCircuitEngine Key Results ===")
print(f"N neurons: {N_NEURONS} (Exc: {N_EXC}, Inh: {N_INH})")
print(f"PV: {n_pv}, SST: {n_sst}, VIP: {n_vip}")
print(f"Total connections: {n_connections}")
print(f"Actual connection probability: {n_connections/(N_NEURONS*(N_NEURONS-1)):.4f}")
print(f"D1 neurons: {d1_neurons.sum()}, D2 neurons: {d2_neurons.sum()}")
print(f"Motifs — Reciprocal: {reciprocal}, Chain: {chain}, Fan-out: {fan_out}")
print(f"Mean firing rate: {all_firing_rates.mean():.2f} Hz")
print(f"Mean CV: {cv_all.mean():.4f}")
print(f"Mean Fano factor: {fano.mean():.4f}")
print(f"Mean E/I ratio: {ei_ratio.mean():.4f}")
