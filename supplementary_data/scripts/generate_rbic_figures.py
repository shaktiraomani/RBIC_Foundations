import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
from scipy.special import j1
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Import exact cosmology & data loaders from run_emcee_rbic
import sys
SCRATCH = r"C:\Users\dell pc\.gemini\antigravity\brain\41eae9ac-a897-4b16-8eaf-aee5c21d3a3c\scratch"
sys.path.insert(0, SCRATCH)
from run_emcee_rbic import load_all_datasets, solve_E, get_Or, comoving_dist_Mpc, drag_and_star_scales, C_LIGHT

OUT_FIG_DIR = r"D:\research\RBIC_Overleaf_Uploads\RBIC_Foundations_arXiv"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 11.5,
    "legend.fontsize": 9.2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.9,
})

data = load_all_datasets()

def load_chain(tag):
    return np.load(os.path.join(SCRATCH, f"mcmc_{tag}.npz"))

d_pp = load_chain("RBIC_uni_Pantheon")
d_des = load_chain("RBIC_uni_DES5YR")
d_u3 = load_chain("RBIC_uni_Union3")
d_lcdm = load_chain("LCDM_Pantheon")
d_w0wa = load_chain("w0wa_Pantheon")

# ==============================================================================
# FIGURE 1: rbic_mcmc_corner.png
# ==============================================================================
ch_pp = d_pp["chain"]
S8_pp = ch_pp[:, 3] * np.sqrt(ch_pp[:, 1] / 0.3)
# 5 physical parameters: H0, Om, S8, gamma_T, xi_RB
samples_corner = np.column_stack([ch_pp[:, 0], ch_pp[:, 1], S8_pp, ch_pp[:, 4], ch_pp[:, 5]])
labels = [r"$H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$", r"$\Omega_m$", r"$S_8$", r"$\gamma_T$", r"$\xi_{\rm RB}$"]

fig = corner.corner(
    samples_corner,
    labels=labels,
    quantiles=[0.16, 0.50, 0.84],
    show_titles=True,
    title_fmt=".3f",
    title_kwargs={"fontsize": 9.5},
    color="#1f5a8a",
    smooth=1.0,
    bins=30,
    plot_datapoints=False,
    fill_contours=True,
    levels=(0.68, 0.95),
    hist_kwargs={"density": True, "linewidth": 1.3},
)
axes = np.array(fig.axes).reshape((5, 5))
# Annotate gamma_T / xi_RB = 1.2 locked relation and LCDM limit on the (gamma_T, xi_RB) panel [4, 3]
ax_gx = axes[4, 3]
gx_line = np.linspace(0.0, np.percentile(ch_pp[:, 4], 99), 50)
ax_gx.plot(gx_line, gx_line / 1.20, color="#d95f02", ls="--", lw=1.4, label=r"$\gamma_T = 1.20\,\xi_{\rm RB}$")
ax_gx.scatter([0.0], [0.0], color="#c51b7d", marker="*", s=110, zorder=10, label=r"$\Lambda$CDM limit")
ax_gx.legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.88)
fig.suptitle(r"RBIC Joint Posterior (CC + DESI DR2 + Planck18 + SDSS $f\sigma_8$ + Pantheon+)", fontsize=12.5, y=1.01)
fig.savefig(os.path.join(OUT_FIG_DIR, "rbic_mcmc_corner.png"), bbox_inches="tight")
plt.close(fig)
print("Saved rbic_mcmc_corner.png")

# ==============================================================================
# FIGURE 2: rbic_hz_fit.png (Top: H(z)/(1+z); Bottom: DESI DR2 BAO residuals)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.8), gridspec_kw={"height_ratios": [1.35, 1.0]}, sharex=False)

z_plot = np.linspace(0.0, 2.45, 250)
rng = np.random.default_rng(123)
idx_sub = rng.choice(len(ch_pp), size=250, replace=False)

H_band = np.empty((len(idx_sub), len(z_plot)))
for k, idx in enumerate(idx_sub):
    th = ch_pp[idx]
    H_band[k] = th[0] * solve_E(z_plot, th[1], get_Or(th[0], th[1]), "RBIC_uni", (th[4], th[5]))
H_16, H_50, H_84 = np.percentile(H_band, [16, 50, 84], axis=0)

th_l = d_lcdm["best_theta"]
H_lcdm = th_l[0] * solve_E(z_plot, th_l[1], get_Or(th_l[0], th_l[1]), "LCDM", ())
th_w = d_w0wa["best_theta"]
H_w0wa = th_w[0] * solve_E(z_plot, th_w[1], get_Or(th_w[0], th_w[1]), "w0wa", (th_w[4], th_w[5]))

# Divide by (1+z) for visual clarity of cosmic acceleration / turnaround
ax1.fill_between(z_plot, H_16 / (1.0 + z_plot), H_84 / (1.0 + z_plot), color="#1f77b4", alpha=0.25, label="RBIC 68% MCMC band")
ax1.plot(z_plot, H_50 / (1.0 + z_plot), color="#1f5a8a", lw=2.0, label="RBIC best-fit (Pantheon+ suite)")
ax1.plot(z_plot, H_lcdm / (1.0 + z_plot), color="#444444", ls="--", lw=1.5, label=r"$\Lambda$CDM baseline")
ax1.plot(z_plot, H_w0wa / (1.0 + z_plot), color="#d95f02", ls=":", lw=1.7, label=r"$w_0 w_a$CDM")

# CC points
ax1.errorbar(data["z_cc"], data["H_cc"] / (1.0 + data["z_cc"]), yerr=data["s_cc"] / (1.0 + data["z_cc"]),
             fmt="o", color="#222222", ms=4.0, elinewidth=0.9, capsize=2.0, alpha=0.82, label="Cosmic Chronometers (32 pts)")

# DESI DR2 DH/rd converted to H(z) at best-fit r_d for plotting
th_rb = d_pp["best_theta"]
_, rd_rb, _, _ = drag_and_star_scales(th_rb[0], th_rb[1], th_rb[2], "RBIC_uni", (th_rb[4], th_rb[5]))
cov_desi = np.linalg.inv(data["invcov_desi"])
err_desi = np.sqrt(np.diag(cov_desi))
mask_dh = np.array([t == "DH_over_rs" for t in data["type_desi"]])
z_dh = data["z_desi"][mask_dh]
dh_val = data["val_desi"][mask_dh]
dh_err = err_desi[mask_dh]
H_desi = C_LIGHT / (dh_val * rd_rb)
H_desi_err = H_desi * (dh_err / dh_val)
ax1.errorbar(z_dh, H_desi / (1.0 + z_dh), yerr=H_desi_err / (1.0 + z_dh),
             fmt="s", color="#c51b7d", ms=5.5, elinewidth=1.3, capsize=2.5, zorder=6, label=r"DESI DR2 BAO ($c/[r_d D_H]$)")

ax1.set_ylabel(r"$H(z)/(1+z)\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
ax1.set_xlim(-0.04, 2.45)
ax1.set_ylim(45, 92)
ax1.legend(loc="upper right", ncol=2, fontsize=8.3, frameon=True, framealpha=0.9)
ax1.grid(alpha=0.25, ls="--")
ax1.set_title("Expansion History & DESI DR2 BAO Confrontation")

# Bottom panel: Ratio to best-fit LCDM for ALL 13 DESI DR2 observables
_, rd_l, _, _ = drag_and_star_scales(th_l[0], th_l[1], th_l[2], "LCDM", ())
DM_l = comoving_dist_Mpc(data["z_desi"], th_l[0], th_l[1], th_l[2], "LCDM", ())
DH_l = (C_LIGHT / th_l[0]) / solve_E(data["z_desi"], th_l[1], get_Or(th_l[0], th_l[1]), "LCDM", ())
DV_l = (data["z_desi"] * DM_l**2 * DH_l) ** (1.0 / 3.0)
pred_l = np.array([DM_l[i]/rd_l if t=="DM_over_rs" else (DH_l[i]/rd_l if t=="DH_over_rs" else DV_l[i]/rd_l) for i, t in enumerate(data["type_desi"])])

DM_r = comoving_dist_Mpc(data["z_desi"], th_rb[0], th_rb[1], th_rb[2], "RBIC_uni", (th_rb[4], th_rb[5]))
DH_r = (C_LIGHT / th_rb[0]) / solve_E(data["z_desi"], th_rb[1], get_Or(th_rb[0], th_rb[1]), "RBIC_uni", (th_rb[4], th_rb[5]))
DV_r = (data["z_desi"] * DM_r**2 * DH_r) ** (1.0 / 3.0)
pred_r = np.array([DM_r[i]/rd_rb if t=="DM_over_rs" else (DH_r[i]/rd_rb if t=="DH_over_rs" else DV_r[i]/rd_rb) for i, t in enumerate(data["type_desi"])])

ax2.axhline(1.0, color="#444444", ls="--", lw=1.2)
for t_label, marker, col, dx in [("DM_over_rs", "o", "#1b9e77", -0.015), ("DH_over_rs", "s", "#c51b7d", +0.015), ("DV_over_rs", "^", "#7570b3", 0.0)]:
    m = np.array([t == t_label for t in data["type_desi"]])
    ax2.errorbar(data["z_desi"][m] + dx, data["val_desi"][m] / pred_l[m], yerr=err_desi[m] / pred_l[m],
                 fmt=marker, color=col, ms=5.5, capsize=2.5, elinewidth=1.1, label=t_label.replace("_over_rs", "/r_d") + " (DESI DR2)")

# Plot RBIC ratio curve for DM/rd and DH/rd
DM_rc = comoving_dist_Mpc(z_plot[1:], th_rb[0], th_rb[1], th_rb[2], "RBIC_uni", (th_rb[4], th_rb[5])) / rd_rb
DM_lc = comoving_dist_Mpc(z_plot[1:], th_l[0], th_l[1], th_l[2], "LCDM", ()) / rd_l
DH_rc = ((C_LIGHT / th_rb[0]) / solve_E(z_plot[1:], th_rb[1], get_Or(th_rb[0], th_rb[1]), "RBIC_uni", (th_rb[4], th_rb[5]))) / rd_rb
DH_lc = ((C_LIGHT / th_l[0]) / solve_E(z_plot[1:], th_l[1], get_Or(th_l[0], th_l[1]), "LCDM", ())) / rd_l
ax2.plot(z_plot[1:], DM_rc / DM_lc, color="#1b9e77", lw=1.7, label=r"RBIC $D_M/r_d$ ratio")
ax2.plot(z_plot[1:], DH_rc / DH_lc, color="#c51b7d", lw=1.7, ls="-.", label=r"RBIC $D_H/r_d$ ratio")

ax2.set_xlabel("Redshift $z$")
ax2.set_ylabel(r"Ratio to $\Lambda$CDM")
ax2.set_xlim(-0.04, 2.45)
ax2.set_ylim(0.935, 1.055)
ax2.legend(loc="lower right", ncol=2, fontsize=8.0, frameon=True, framealpha=0.9)
ax2.grid(alpha=0.25, ls="--")

fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG_DIR, "rbic_hz_fit.png"), bbox_inches="tight")
plt.close(fig)
print("Saved rbic_hz_fit.png")

# ==============================================================================
# FIGURE 3: rbic_weff_phantom.png
# ==============================================================================
def compute_weff(z_arr, Om, Or, gamma_T, xi_RB):
    E = solve_E(z_arr, Om, Or, "RBIC_uni", (gamma_T, xi_RB))
    a = 1.0 + z_arr
    OL0 = 1.0 - Om - Or
    exp_val = np.exp(np.clip(-xi_RB * (E - 1.0), -60.0, 20.0))
    # Implicit differentiation of E^2 = Om a^3 + Or a^4 + OL0 a^gamma_T exp(-xi(E-1)):
    # dE/d(ln a) * [2 E + OL0 xi a^gamma_T exp_val] = 3 Om a^3 + 4 Or a^4 + gamma_T OL0 a^gamma_T exp_val
    num = 3.0 * Om * a**3 + 4.0 * Or * a**4 + gamma_T * OL0 * (a**gamma_T) * exp_val
    den = 2.0 * E + OL0 * xi_RB * (a**gamma_T) * exp_val
    dE_dlna = num / np.maximum(den, 1e-12)
    return -1.0 + (gamma_T - xi_RB * dE_dlna) / 3.0

z_w = np.linspace(0.0, 2.2, 200)
w_band = np.empty((len(idx_sub), len(z_w)))
for k, idx in enumerate(idx_sub):
    th = ch_pp[idx]
    w_band[k] = compute_weff(z_w, th[1], get_Or(th[0], th[1]), th[4], th[5])
w_02, w_16, w_50, w_84, w_97 = np.percentile(w_band, [2.5, 16, 50, 84, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.fill_between(z_w, w_02, w_97, color="#1f77b4", alpha=0.15, label="RBIC 95% CL (Pantheon+ suite)")
ax.fill_between(z_w, w_16, w_84, color="#1f77b4", alpha=0.32, label="RBIC 68% CL (Pantheon+ suite)")
ax.plot(z_w, w_50, color="#1f5a8a", lw=2.2, label="RBIC median (Pantheon+)")

th_des = d_des["best_theta"]
w_des = compute_weff(z_w, th_des[1], get_Or(th_des[0], th_des[1]), th_des[4], th_des[5])
ax.plot(z_w, w_des, color="#d95f02", ls="-.", lw=1.9, label="RBIC best-fit (DES-SN5YR)")

th_u3 = d_u3["best_theta"]
w_u3 = compute_weff(z_w, th_u3[1], get_Or(th_u3[0], th_u3[1]), th_u3[4], th_u3[5])
ax.plot(z_w, w_u3, color="#1b9e77", ls=":", lw=2.0, label="RBIC best-fit (Union3)")

w_cpl = th_w[4] + th_w[5] * (z_w / (1.0 + z_w))
ax.plot(z_w, w_cpl, color="#7570b3", ls="--", lw=1.4, label=r"$w_0 w_a$CDM CPL (Pantheon+)")
ax.axhline(-1.0, color="#222222", ls="--", lw=1.1, label=r"$\Lambda$CDM ($w = -1$)")

# Mark phantom crossing
z_cross_idx = np.argmin(np.abs(w_50 + 1.0))
z_ph_med = z_w[z_cross_idx]
ax.scatter([z_ph_med], [-1.0], color="#c51b7d", s=55, zorder=8)
ax.annotate(rf"Phantom crossing $z_{{\rm ph}} \approx {z_ph_med:.2f}$", xy=(z_ph_med, -1.0), xytext=(z_ph_med + 0.25, -0.78),
            arrowprops=dict(arrowstyle="->", color="#c51b7d", lw=1.2), fontsize=9.5, color="#c51b7d")

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Effective Equation of State $w_{\rm eff}(z)$")
ax.set_xlim(0.0, 2.2)
ax.set_ylim(-1.75, -0.58)
ax.legend(loc="lower left", fontsize=8.3, ncol=2, frameon=True, framealpha=0.92)
ax.grid(alpha=0.25, ls="--")
ax.set_title("Dynamical Phantom Crossing from Horizon-Condensate Boltzmann Regularization", pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG_DIR, "rbic_weff_phantom.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved rbic_weff_phantom.png (z_ph={z_ph_med:.2f})")

# ==============================================================================
# FIGURE 4: kids_shear_variance.png
# ==============================================================================
kd = np.load(os.path.join(SCRATCH, "kids_legacy_summary.npz"))
z_edges = kd["z_edges"]
z_mids = 0.5 * (z_edges[:-1] + z_edges[1:])
nz_high = kd["nz_hist"][3:6].sum(axis=0)
nz_high = nz_high / np.trapezoid(nz_high, z_mids)
scales_deg = kd["scales_deg"]
obs_var = kd["excess_var"]
obs_err = kd["excess_err"]

def T_EH(k_hMpc, Om=0.312, Ob=0.049, h=0.677, Tcmb=2.7255):
    omh2 = Om * h * h; obh2 = Ob * h * h; f_b = Ob / Om; theta = Tcmb / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1.0 + 10.0 * obh2**0.75)
    alpha_gamma = 1.0 - 0.328 * np.log(431.0 * omh2) * f_b + 0.38 * np.log(22.3 * omh2) * f_b**2
    gamma_eff = Om * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_hMpc * s)**4))
    q = k_hMpc * theta**2 / gamma_eff
    L0 = np.log(2.0 * np.e + 1.8 * q); C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)

H0_bf, Om_bf, omb_bf, s8_bf, gT_bf, xi_bf = d_pp["best_theta"]
h_bf = H0_bf / 100.0; Ob_bf = omb_bf / (h_bf**2); ns = 0.965
z_g = np.linspace(0.0, 2.2, 221); E_g = solve_E(z_g, Om_bf, get_Or(H0_bf, Om_bf), "RBIC_uni", (gT_bf, xi_bf))
chi_g = np.zeros_like(z_g)
chi_g[1:] = np.cumsum(0.5 * (1.0 / E_g[:-1] + 1.0 / E_g[1:]) * (z_g[1] - z_g[0])) * (C_LIGHT / H0_bf)

# Growth D(z)
a_g = np.linspace(0.01, 1.0, 300); da = a_g[1] - a_g[0]
E_ag = solve_E(1.0 / a_g - 1.0, Om_bf, get_Or(H0_bf, Om_bf), "RBIC_uni", (gT_bf, xi_bf))
dlne = np.gradient(np.log(E_ag), np.log(a_g))
D, Dp = 0.01, 1.0; Ds = [D]
for i in range(len(a_g) - 1):
    def rhs(av, dv, dpv, ev, dlv): return dpv, 1.5 * Om_bf / (av**5 * ev**2) * dv - (3 / av + dlv / av) * dpv
    k1 = rhs(a_g[i], D, Dp, E_ag[i], dlne[i])
    k2 = rhs(a_g[i] + 0.5*da, D + 0.5*da*k1[0], Dp + 0.5*da*k1[1], 0.5*(E_ag[i]+E_ag[i+1]), 0.5*(dlne[i]+dlne[i+1]))
    k3 = rhs(a_g[i] + 0.5*da, D + 0.5*da*k2[0], Dp + 0.5*da*k2[1], 0.5*(E_ag[i]+E_ag[i+1]), 0.5*(dlne[i]+dlne[i+1]))
    k4 = rhs(a_g[i+1], D + da*k3[0], Dp + da*k3[1], E_ag[i+1], dlne[i+1])
    D += da/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); Dp += da/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    Ds.append(D)
D_of_z = interp1d(1.0 / a_g - 1.0, np.array(Ds) / Ds[-1], fill_value="extrapolate")
k_norm = np.logspace(-4, 2, 2000); P0_raw = (k_norm**ns) * (T_EH(k_norm, Om_bf, Ob_bf, h_bf)**2)
x8 = k_norm * 8.0; W8 = 3.0 * (np.sin(x8) - x8 * np.cos(x8)) / (x8**3)
sig8_raw = np.sqrt(simpson(k_norm**2 * P0_raw * W8**2 / (2 * np.pi**2), x=k_norm))
nz_interp = np.interp(z_g, z_mids, nz_high, left=0.0, right=0.0)
n_chi = nz_interp * (H0_bf * E_g / C_LIGHT); W_chi = np.zeros_like(chi_g)
for i in range(1, len(chi_g)):
    c_i = chi_g[i]; tail = (chi_g[i:] - c_i) / np.maximum(chi_g[i:], 1e-9)
    W_chi[i] = 1.5 * Om_bf * ((H0_bf / C_LIGHT)**2) * (1.0 + z_g[i]) * c_i * np.trapezoid(n_chi[i:] * tail, chi_g[i:])
ell_grid = np.logspace(1.0, 3.8, 120)

def pred_shear(th_arr, ks=None, sig8=s8_bf):
    A_norm = (sig8 / sig8_raw)**2; Cell = np.zeros_like(ell_grid)
    for j, ell in enumerate(ell_grid):
        k_hMpc = ((ell + 0.5) / np.maximum(chi_g[1:], 1.0)) / h_bf
        Pk_lin = A_norm * (k_hMpc**ns) * (T_EH(k_hMpc, Om_bf, Ob_bf, h_bf)**2) * (D_of_z(z_g[1:])**2)
        Delta2_L = (k_hMpc**3) * Pk_lin / (2.0 * np.pi**2)
        nl_fac = (1.0 + 1.2 * Delta2_L) / (1.0 + 0.12 * (Delta2_L**0.4))
        Pk_nl = Pk_lin * np.where(Delta2_L > 0.05, nl_fac, 1.0)
        if ks is not None:
            Pk_nl = Pk_nl / (1.0 + (k_hMpc / ks)**2)
        Cell[j] = np.trapezoid((W_chi[1:]**2 / (chi_g[1:]**2)) * (Pk_nl / h_bf**3), chi_g[1:])
    preds = []
    for th in th_arr:
        R_eff = np.deg2rad(th) / np.sqrt(np.pi); x = ell_grid * R_eff; W2 = (2.0 * j1(x) / x)**2
        preds.append(np.trapezoid(ell_grid * Cell * W2 / (2.0 * np.pi), ell_grid))
    return np.array(preds)

ks_grid = np.linspace(0.20, 1.20, 60)
c2_ks = np.array([np.sum(((pred_shear(scales_deg, ks) - obs_var) / obs_err)**2) for ks in ks_grid])
best_i = int(np.argmin(c2_ks))
ks_bf = ks_grid[best_i]; c2_bf = c2_ks[best_i]
ks_1s = ks_grid[c2_ks <= c2_bf + 1.0]

th_dense = np.geomspace(0.20, 2.50, 35)
y_nosupp = pred_shear(th_dense, None)
y_rbic = pred_shear(th_dense, ks_bf)
y_lo = pred_shear(th_dense, ks_1s[0])
y_hi = pred_shear(th_dense, ks_1s[-1])
c2_nosupp = float(np.sum(((pred_shear(scales_deg, None) - obs_var) / obs_err)**2))

fig, ax = plt.subplots(figsize=(6.8, 4.7))
ax.fill_between(th_dense, y_lo, y_hi, color="#1f77b4", alpha=0.25, label=rf"RBIC $1\sigma$ band ($k_\star = {ks_bf:.2f}\;h\,\mathrm{{Mpc}}^{{-1}}$)")
ax.plot(th_dense, y_rbic, color="#1f5a8a", lw=2.2, label=rf"RBIC + Jeans cutoff ($\chi^2 = {c2_bf:.2f}$)")
ax.plot(th_dense, y_nosupp, color="#d95f02", ls="--", lw=1.9, label=rf"Unsuppressed CDM ($\chi^2 = {c2_nosupp:.2f}$)")
ax.errorbar(scales_deg, obs_var, yerr=obs_err, fmt="o", color="#111111", ms=6.0, capsize=3.0, elinewidth=1.4, zorder=8,
            label="KiDS-Legacy DR5 (13.5M real galaxies, bins 4–6)")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.25, 0.50, 1.00, 2.00])
ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel(r"Aperture Side Scale $\theta\;[\mathrm{deg}]$")
ax.set_ylabel(r"Excess Shear Variance $\langle \gamma_{\rm ex}^2(\theta) \rangle$")
ax.set_xlim(0.20, 2.40)
ax.set_ylim(5e-6, 1.5e-4)
ax.legend(loc="upper right", fontsize=8.4, frameon=True, framealpha=0.92)
ax.grid(alpha=0.25, which="both", ls="--")
ax.set_title("Weak-Lensing Power Suppression from Real KiDS-Legacy DR5 Shear Data", pad=10)

# Inset: chi2(k_star)
ax_in = ax.inset_axes([0.12, 0.14, 0.36, 0.32])
ax_in.plot(ks_grid, c2_ks, color="#1f5a8a", lw=1.6)
ax_in.axhline(c2_bf + 1.0, color="#c51b7d", ls=":", lw=1.0)
ax_in.axvline(ks_bf, color="#1f5a8a", ls="--", lw=1.0)
ax_in.set_xlabel(r"$k_\star\;[h\,\mathrm{Mpc}^{-1}]$", fontsize=8.5)
ax_in.set_ylabel(r"$\chi^2$", fontsize=8.5)
ax_in.set_ylim(0, 20)
ax_in.tick_params(labelsize=7.5)
ax_in.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(os.path.join(OUT_FIG_DIR, "kids_shear_variance.png"), bbox_inches="tight")
plt.close(fig)
print(f"Saved kids_shear_variance.png (ks_bf={ks_bf:.3f}, chi2_bf={c2_bf:.2f}, chi2_nosupp={c2_nosupp:.2f})")
