import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.interpolate import CubicSpline
import emcee

BASE = r"D:\research\Overleaf Projects (6 items)\data"
OUT_DIR = r"C:\Users\dell pc\.gemini\antigravity\brain\41eae9ac-a897-4b16-8eaf-aee5c21d3a3c\scratch"
C_LIGHT = 299792.458

GL_X, GL_W = np.polynomial.legendre.leggauss(64)
GL_U, GL_DU = 0.5 * (GL_X + 1.0), 0.5 * GL_W
GL_X_RD, GL_W_RD = np.polynomial.legendre.leggauss(48)
GL_U_RD, GL_DU_RD = 0.5 * (GL_X_RD + 1.0), 0.5 * GL_W_RD
S_GRID = np.linspace(0.0, np.log1p(2.5), 257)
Z_GRID_1D = np.expm1(S_GRID)

# Worker globals populated on import / init
_DATA = {}

def load_all_datasets():
    if _DATA:
        return _DATA
    cc_data = np.loadtxt(os.path.join(BASE, "Cosmic_chronometer_data-main", "Cosmic_chronometer_data-main", "HzTable_MM_BC32.txt"))
    _DATA["z_cc"], _DATA["H_cc"], _DATA["s_cc"] = cc_data[:, 0], cc_data[:, 1], cc_data[:, 2]

    desi_rows = [l.split() for l in open(os.path.join(BASE, "bao_data-master", "desi_bao_dr2", "desi_gaussian_bao_ALL_GCcomb_mean.txt")) if l.strip() and not l.startswith("#")]
    _DATA["z_desi"] = np.array([float(r[0]) for r in desi_rows])
    _DATA["val_desi"] = np.array([float(r[1]) for r in desi_rows])
    _DATA["type_desi"] = [r[2] for r in desi_rows]
    _DATA["invcov_desi"] = np.linalg.inv(np.loadtxt(os.path.join(BASE, "bao_data-master", "desi_bao_dr2", "desi_gaussian_bao_ALL_GCcomb_cov.txt")))

    lrg_rows = [l.split() for l in open(os.path.join(BASE, "bao_data-master", "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8.dat")) if l.strip() and not l.startswith("#")]
    lrg_cov = np.loadtxt(os.path.join(BASE, "bao_data-master", "sdss_DR16_BAOplus_LRG_FSBAO_DMDHfs8_covtot.txt"))
    qso_rows = [l.split() for l in open(os.path.join(BASE, "bao_data-master", "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8.dat")) if l.strip() and not l.startswith("#")]
    qso_cov = np.loadtxt(os.path.join(BASE, "bao_data-master", "sdss_DR16_BAOplus_QSO_FSBAO_DMDHfs8_covtot.txt"))
    _DATA["z_fs8"] = np.array([float(lrg_rows[i][0]) for i in (2, 5, 8)] + [float(qso_rows[2][0])])
    _DATA["val_fs8"] = np.array([float(lrg_rows[i][1]) for i in (2, 5, 8)] + [float(qso_rows[2][1])])
    cov_fs8 = np.zeros((4, 4))
    cov_fs8[:3, :3] = lrg_cov[np.ix_([2, 5, 8], [2, 5, 8])]
    cov_fs8[3, 3] = qso_cov[2, 2]
    _DATA["invcov_fs8"] = np.linalg.inv(cov_fs8)

    pp_df = pd.read_csv(os.path.join(BASE, "DataRelease-main", "Pantheon+_Data", "4_DISTANCES_AND_COVAR", "Pantheon+SH0ES.dat"), sep=r"\s+")
    with open(os.path.join(BASE, "DataRelease-main", "Pantheon+_Data", "4_DISTANCES_AND_COVAR", "Pantheon+SH0ES_STAT+SYS.cov")) as f:
        n_pp = int(f.readline().strip())
        pp_cov_full = np.loadtxt(f).reshape((n_pp, n_pp))
    mask_pp = pp_df["zHD"].values > 0.01
    _DATA["z_pp"] = pp_df["zHD"].values[mask_pp]
    _DATA["zhel_pp"] = pp_df["zHEL"].values[mask_pp]
    _DATA["mb_pp"] = pp_df["m_b_corr"].values[mask_pp]
    invcov_pp = cho_solve(cho_factor(pp_cov_full[np.ix_(mask_pp, mask_pp)]), np.eye(len(_DATA["z_pp"])))
    ones_pp = np.ones(len(_DATA["z_pp"]))
    _DATA["invcov_pp"] = invcov_pp
    _DATA["Cinv_1_pp"] = invcov_pp @ ones_pp
    _DATA["denom_pp"] = float(ones_pp @ _DATA["Cinv_1_pp"])

    des_z, des_zhel, des_mu = [], [], []
    with open(os.path.join(BASE, "DES-SN5YR-main", "DES-SN5YR-main", "4_DISTANCES_COVMAT", "DES-Dovekie_HD.csv")) as f:
        for line in f:
            if line.startswith("SN:"):
                p = line.split()
                des_z.append(float(p[3]))
                des_zhel.append(float(p[4]))
                des_mu.append(float(p[5]))
    _DATA["z_des"] = np.array(des_z)
    _DATA["zhel_des"] = np.array(des_zhel)
    _DATA["mu_des"] = np.array(des_mu)
    npz_des = np.load(os.path.join(BASE, "DES-SN5YR-main", "DES-SN5YR-main", "4_DISTANCES_COVMAT", "STAT+SYS.npz"))
    n_des = int(npz_des[npz_des.files[0]][0])
    invcov_des = np.zeros((n_des, n_des))
    invcov_des[np.triu_indices(n_des)] = npz_des[npz_des.files[1]]
    il = np.tril_indices(n_des, -1)
    invcov_des[il] = invcov_des.T[il]
    ones_des = np.ones(n_des)
    _DATA["invcov_des"] = invcov_des
    _DATA["Cinv_1_des"] = invcov_des @ ones_des
    _DATA["denom_des"] = float(ones_des @ _DATA["Cinv_1_des"])

    u3 = fits.getdata(os.path.join(BASE, "union3_release-main", "mu_mat_union3_cosmo=2_mu.fits"))
    _DATA["z_u3"] = np.asarray(u3[0, 1:], float)
    _DATA["mu_u3"] = np.asarray(u3[1:, 0], float)
    _DATA["invcov_u3"] = np.asarray(u3[1:, 1:], float)
    ones_u3 = np.ones(len(_DATA["z_u3"]))
    _DATA["Cinv_1_u3"] = _DATA["invcov_u3"] @ ones_u3
    _DATA["denom_u3"] = float(ones_u3 @ _DATA["Cinv_1_u3"])
    return _DATA


def get_Or(H0, Om):
    h = H0 / 100.0
    omh2 = Om * h * h
    return Om / (1.0 + 2.5e4 * omh2 * (2.7255 / 2.7) ** (-4))


def solve_E(z, Om, Or, model, extra):
    z = np.asarray(z, dtype=float)
    a = 1.0 + z
    rho_m_r = Om * a**3 + Or * a**4
    OL0 = 1.0 - Om - Or
    if model == "LCDM":
        return np.sqrt(np.maximum(rho_m_r + OL0, 1e-12))
    elif model == "w0wa":
        w0, wa = extra
        return np.sqrt(np.maximum(rho_m_r + OL0 * a ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / a), 1e-12))
    elif model in ("RBIC_uni", "RBIC_1p"):
        if model == "RBIC_1p":
            xi_RB = extra[0]
            gamma_T = 1.2 * xi_RB
        else:
            gamma_T, xi_RB = extra
        torsion_pref = OL0 * (a ** gamma_T)
        E = np.sqrt(np.maximum(rho_m_r + OL0, 1e-12))
        for _ in range(14):
            arg = np.clip(-xi_RB * (E - 1.0), -60.0, 20.0)
            exp_val = np.exp(arg)
            f = E * E - rho_m_r - torsion_pref * exp_val
            df = 2.0 * E + torsion_pref * xi_RB * exp_val
            df = np.where(np.abs(df) < 1e-12, 1e-12, df)
            step = f / df
            E = np.maximum(E - step, 1e-6)
            if np.max(np.abs(step)) < 1e-11:
                break
        return E


def comoving_dist_Mpc(z_arr, H0, Om, ombh2, model, extra):
    Or = get_Or(H0, Om)
    if len(z_arr) > 40 and np.max(z_arr) <= 2.5:
        E_1d = solve_E(Z_GRID_1D, Om, Or, model, extra)
        cs_int = CubicSpline(S_GRID, (1.0 + Z_GRID_1D) / E_1d).antiderivative()
        return (C_LIGHT / H0) * (cs_int(np.log1p(z_arr)) - cs_int(0.0))
    s_max = np.log1p(z_arr)
    z_grid = np.expm1(s_max[:, None] * GL_U[None, :])
    E_grid = solve_E(z_grid, Om, Or, model, extra)
    return (C_LIGHT / H0) * np.sum(GL_DU[None, :] * (1.0 + z_grid) / E_grid, axis=1) * s_max


def drag_and_star_scales(H0, Om, ombh2, model, extra):
    h = H0 / 100.0
    omh2 = Om * h * h
    Or = get_Or(H0, Om)
    g1 = (0.0783 * ombh2 ** (-0.238)) / (1.0 + 39.5 * ombh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * ombh2 ** 1.81)
    z_star = 1048.0 * (1.0 + 0.00124 * ombh2 ** (-0.738)) * (1.0 + g1 * omh2 ** g2)
    omcbh2 = omh2 - 0.000645
    if omcbh2 <= 0.01:
        return 1059.94, 147.0, z_star, 144.0
    rd_aub = 55.154 * np.exp(-72.3 * (0.000645 + 0.0006) ** 2) / (omcbh2 ** 0.25351 * ombh2 ** 0.12807)
    z_d = 1059.94
    Rb = 31500.0 * ombh2 * (2.7255 / 2.7) ** (-4)

    def calc_rs(z_lim, m_name, ext):
        a_lim = 1.0 / (1.0 + z_lim)
        a_pts = np.maximum(a_lim * GL_U_RD, 1e-9)
        z_pts = 1.0 / a_pts - 1.0
        E_pts = solve_E(z_pts, Om, Or, m_name, ext)
        return (C_LIGHT / H0) * np.sum(GL_DU_RD / (np.sqrt(3.0 * (1.0 + Rb * a_pts)) * a_pts * a_pts * E_pts)) * a_lim

    rd = rd_aub if model == "LCDM" else rd_aub * (calc_rs(z_d, model, extra) / calc_rs(z_d, "LCDM", ()))
    return z_d, rd, z_star, calc_rs(z_star, model, extra)


def calc_fsig8(z_arr, H0, Om, sig8_0, model, extra):
    Or = get_Or(H0, Om)
    N_steps = 200
    a_grid = np.linspace(0.01, 1.0, N_steps + 1)
    da = a_grid[1] - a_grid[0]
    E_grid = solve_E(1.0 / a_grid - 1.0, Om, Or, model, extra)
    dlnE_dlna = np.gradient(np.log(E_grid), np.log(a_grid))
    D, Dp = 0.01, 1.0
    D_arr, Dp_arr = np.zeros_like(a_grid), np.zeros_like(a_grid)
    D_arr[0], Dp_arr[0] = D, Dp
    for i in range(N_steps):
        ai, Ei, dlne = a_grid[i], E_grid[i], dlnE_dlna[i]
        an, En, dlnen = a_grid[i + 1], E_grid[i + 1], dlnE_dlna[i + 1]
        am, Em, dlnem = 0.5 * (ai + an), 0.5 * (Ei + En), 0.5 * (dlne + dlnen)

        def rhs(a_v, d_v, dp_v, e_v, dln_v):
            return dp_v, (1.5 * Om / (a_v**5 * e_v**2)) * d_v - (3.0 / a_v + dln_v / a_v) * dp_v

        k1 = rhs(ai, D, Dp, Ei, dlne)
        k2 = rhs(am, D + 0.5 * da * k1[0], Dp + 0.5 * da * k1[1], Em, dlnem)
        k3 = rhs(am, D + 0.5 * da * k2[0], Dp + 0.5 * da * k2[1], Em, dlnem)
        k4 = rhs(an, D + da * k3[0], Dp + da * k3[1], En, dlnen)
        D += (da / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Dp += (da / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        D_arr[i + 1], Dp_arr[i + 1] = D, Dp
    return np.interp(1.0 / (1.0 + z_arr), a_grid, sig8_0 * (a_grid * Dp_arr / D_arr[-1]))


def check_prior(theta, model):
    H0, Om, ombh2, sig8 = theta[:4]
    extra = theta[4:]
    if not (55.0 < H0 < 85.0 and 0.15 < Om < 0.50 and 0.018 < ombh2 < 0.026 and 0.60 < sig8 < 1.00):
        return False
    if model == "w0wa":
        if not (-2.5 < extra[0] < 0.0 and -3.0 < extra[1] < 2.0):
            return False
    elif model == "RBIC_uni":
        if not (-2.0 < extra[0] < 5.0 and 0.0 <= extra[1] < 5.0):
            return False
    elif model == "RBIC_1p":
        if not (0.0 <= extra[0] < 4.0):
            return False
    return True


def log_prob(theta, model, sn_suite):
    if not check_prior(theta, model):
        return -np.inf
    d = _DATA
    H0, Om, ombh2, sig8 = theta[:4]
    extra = tuple(theta[4:])
    Or = get_Or(H0, Om)

    chi2_cc = float(np.sum(((H0 * solve_E(d["z_cc"], Om, Or, model, extra) - d["H_cc"]) / d["s_cc"]) ** 2))

    _, rd, z_star, rstar = drag_and_star_scales(H0, Om, ombh2, model, extra)
    DM_d = comoving_dist_Mpc(d["z_desi"], H0, Om, ombh2, model, extra)
    DH_d = (C_LIGHT / H0) / solve_E(d["z_desi"], Om, Or, model, extra)
    DV_d = (d["z_desi"] * DM_d**2 * DH_d) ** (1.0 / 3.0)
    p_d = np.array([
        DM_d[i] / rd if t == "DM_over_rs" else (DH_d[i] / rd if t == "DH_over_rs" else DV_d[i] / rd)
        for i, t in enumerate(d["type_desi"])
    ])
    diff_d = p_d - d["val_desi"]
    chi2_desi = float(diff_d @ d["invcov_desi"] @ diff_d)

    DM_s = comoving_dist_Mpc(np.array([z_star]), H0, Om, ombh2, model, extra)[0]
    chi2_cmb = (
        ((np.sqrt(Om) * (H0 / C_LIGHT) * DM_s - 1.7502) / 0.0046) ** 2
        + ((np.pi * DM_s / rstar - 301.471) / 0.090) ** 2
        + ((ombh2 - 0.02236) / 0.00015) ** 2
    )

    d_f = calc_fsig8(d["z_fs8"], H0, Om, sig8, model, extra) - d["val_fs8"]
    chi2_fs8 = float(d_f @ d["invcov_fs8"] @ d_f)

    if sn_suite == "Pantheon+":
        dl = (1.0 + d["zhel_pp"]) * comoving_dist_Mpc(d["z_pp"], H0, Om, ombh2, model, extra)
        del_ = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0 - d["mb_pp"]
        chi2_sn = float(del_ @ (d["invcov_pp"] @ del_)) - (float(del_ @ d["Cinv_1_pp"]) ** 2) / d["denom_pp"]
    elif sn_suite == "DES-SN5YR":
        dl = (1.0 + d["zhel_des"]) * comoving_dist_Mpc(d["z_des"], H0, Om, ombh2, model, extra)
        del_ = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0 - d["mu_des"]
        chi2_sn = float(del_ @ (d["invcov_des"] @ del_)) - (float(del_ @ d["Cinv_1_des"]) ** 2) / d["denom_des"]
    else:
        dl = (1.0 + d["z_u3"]) * comoving_dist_Mpc(d["z_u3"], H0, Om, ombh2, model, extra)
        del_ = 5.0 * np.log10(np.maximum(dl, 1e-10)) + 25.0 - d["mu_u3"]
        chi2_sn = float(del_ @ (d["invcov_u3"] @ del_)) - (float(del_ @ d["Cinv_1_u3"]) ** 2) / d["denom_u3"]

    val = -0.5 * (chi2_cc + chi2_desi + chi2_cmb + chi2_fs8 + chi2_sn)
    return val if np.isfinite(val) else -np.inf


def run_chain(pool, model, sn_suite, x0, sig_init, nwalkers=32, nburn=700, nprod=2000, tag=""):
    t_start = time.time()
    ndim = len(x0)
    rng = np.random.default_rng(42)

    # First polish MLE quickly
    res = minimize(lambda th: -2.0 * log_prob(th, model, sn_suite) if np.isfinite(log_prob(th, model, sn_suite)) else 1e12,
                   x0, method="Nelder-Mead", options={"maxfev": 600})
    x_map = res.x
    p0 = []
    while len(p0) < nwalkers:
        cand = x_map + sig_init * rng.normal(size=ndim)
        if check_prior(cand, model):
            p0.append(cand)
    p0 = np.array(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(model, sn_suite), pool=pool)
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nprod, progress=False)

    chain = sampler.get_chain(flat=True)
    logp = sampler.get_log_prob(flat=True)
    acc = float(np.mean(sampler.acceptance_fraction))
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = np.full(ndim, np.nan)

    best_idx = int(np.argmax(logp))
    best_theta = chain[best_idx]
    chi2_min = -2.0 * logp[best_idx]

    q16, q50, q84 = np.percentile(chain, [16, 50, 84], axis=0)
    S8_chain = chain[:, 3] * np.sqrt(chain[:, 1] / 0.3)
    s8_q16, s8_q50, s8_q84 = np.percentile(S8_chain, [16, 50, 84])

    out_path = os.path.join(OUT_DIR, f"mcmc_{tag}.npz")
    np.savez_compressed(
        out_path,
        chain=chain,
        raw_chain=sampler.get_chain(),
        log_prob=logp,
        best_theta=best_theta,
        chi2_min=chi2_min,
        tau=tau,
        acc=acc,
        q16=q16,
        q50=q50,
        q84=q84,
        S8_q=(s8_q16, s8_q50, s8_q84),
    )
    dt = time.time() - t_start
    print(f"[{tag}] done in {dt:.1f}s | acc={acc:.3f} | max(tau)={np.nanmax(tau):.1f} | chi2_min={chi2_min:.2f}", flush=True)
    names = ["H0", "Om", "ombh2", "sig8"] + (["w0", "wa"] if model == "w0wa" else (["gamma_T", "xi_RB"] if model == "RBIC_uni" else (["xi_RB"] if model == "RBIC_1p" else [])))
    for i, nm in enumerate(names):
        print(f"   {nm:8s} = {q50[i]:.4f} (+{q84[i]-q50[i]:.4f} / -{q50[i]-q16[i]:.4f})  [MAP={best_theta[i]:.4f}]", flush=True)
    print(f"   S8       = {s8_q50:.4f} (+{s8_q84-s8_q50:.4f} / -{s8_q50-s8_q16:.4f})", flush=True)
    if model == "RBIC_uni":
        ratio = chain[:, 4] / np.maximum(chain[:, 5], 1e-6)
        r16, r50, r84 = np.percentile(ratio[chain[:, 5] > 0.05], [16, 50, 84]) if np.any(chain[:, 5] > 0.05) else (0, 0, 0)
        print(f"   gT/xi    = {r50:.3f} (+{r84-r50:.3f} / -{r50-r16:.3f})", flush=True)
    return out_path


if __name__ == "__main__":
    load_all_datasets()
    n_procs = min(6, os.cpu_count() or 4)
    print(f"Starting emcee MCMC with {n_procs} worker processes...", flush=True)
    with mp.Pool(processes=n_procs, initializer=load_all_datasets) as pool:
        # 1. RBIC_uni on all 3 SN suites (Pantheon+, DES-SN5YR, Union3)
        run_chain(
            pool, "RBIC_uni", "Pantheon+",
            [67.74, 0.3120, 0.02236, 0.856, 0.584, 0.486],
            [0.25, 0.004, 0.0001, 0.015, 0.15, 0.12],
            nwalkers=32, nburn=700, nprod=2000, tag="RBIC_uni_Pantheon"
        )
        run_chain(
            pool, "RBIC_uni", "DES-SN5YR",
            [67.68, 0.3128, 0.02236, 0.856, 0.643, 0.536],
            [0.25, 0.004, 0.0001, 0.015, 0.15, 0.12],
            nwalkers=32, nburn=700, nprod=2000, tag="RBIC_uni_DES5YR"
        )
        run_chain(
            pool, "RBIC_uni", "Union3",
            [67.50, 0.3140, 0.02236, 0.856, 0.70, 0.58],
            [0.30, 0.005, 0.0001, 0.015, 0.18, 0.14],
            nwalkers=32, nburn=700, nprod=2000, tag="RBIC_uni_Union3"
        )
        # 2. LCDM baseline on Pantheon+ and DES-SN5YR
        run_chain(
            pool, "LCDM", "Pantheon+",
            [68.3, 0.303, 0.02236, 0.850],
            [0.25, 0.004, 0.0001, 0.015],
            nwalkers=32, nburn=500, nprod=1500, tag="LCDM_Pantheon"
        )
        run_chain(
            pool, "RBIC_1p", "Pantheon+",
            [67.74, 0.3120, 0.02236, 0.856, 0.486],
            [0.25, 0.004, 0.0001, 0.015, 0.10],
            nwalkers=32, nburn=600, nprod=1800, tag="RBIC_1p_Pantheon"
        )
        run_chain(
            pool, "w0wa", "Pantheon+",
            [67.7, 0.312, 0.02236, 0.856, -0.84, -0.55],
            [0.25, 0.004, 0.0001, 0.015, 0.04, 0.15],
            nwalkers=32, nburn=600, nprod=1800, tag="w0wa_Pantheon"
        )
    print("ALL CHAINS COMPLETE!", flush=True)
