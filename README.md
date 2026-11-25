# Relativistic Bose Interval Cosmology (RBIC): Foundations, Vacuum Thermodynamics, and Observational Synthesis

**Author:** Shakti Rao Mani (ORCID: [0009-0005-6066-8584](https://orcid.org/0009-0005-6066-8584))  
**Affiliation:** School of Sciences, Indira Gandhi National Open University, Maidan Garhi, New Delhi, 110068, India  
**Contact:** `shaktimanibhu@gmail.com`  
**Preprint:** SSRN Electronic Journal (`doi:10.2139/ssrn.7656002`)

---

## Overview

This repository contains the reproducibility data, extracted weak-lensing shear variance summaries, MCMC posterior tables, Python analysis/plotting scripts, and LaTeX manuscript sources for:

> *Relativistic Bose Interval Cosmology: Foundations, Vacuum Thermodynamics, and Observational Synthesis*

### Key Results
- **MMT–BEC Unification:** Identifies the dimensionless Modified Measure Theory (MMT) ratio $\chi \equiv \Phi/\sqrt{-g}$ with the normalized Bose–Einstein Condensate (BEC) density $|\Psi_I|^2/\rho_0$.
- **Horizon–Condensate Boltzmann Damping:** Motivates $\mathcal{B}_{\rm RB}(z)/\mathcal{B}_{\rm RB}(0) = \exp[-\xi_{\rm RB}(E(z)-1)]$, recovering flat $\Lambda$CDM identically as $(\gamma_T, \xi_{\rm RB}) \to (0,0)$.
- **Multi-Probe MCMC (CC + DESI DR2 BAO + Planck 2018 + SDSS $f\sigma_8$ + Pantheon+/DES-SN5YR/Union3):**
  - Full RBIC ($k=6$): $\Delta\mathrm{AIC} = -4.82$ to $-8.92$ ($\Delta\mathrm{BIC} = +5.99, +4.36, -4.31$).
  - Calibrated $\mathrm{RBIC}_{1p}$ ($\gamma_T = 1.20\,\xi_{\rm RB}$, $k=5$): $\Delta\mathrm{AIC} = -6.82$ to $-10.05$, $\Delta\mathrm{BIC} = -1.41, -3.17, -7.74$.
- **KiDS-Legacy DR5 Shear Variance (4 bins, $13.5 \times 10^6$ galaxies):** Lower bound $k_\star \gtrsim 0.35\;h\,\mathrm{Mpc}^{-1}$ ($>2.9\sigma$ exclusion for $k_\star \le 0.25\;h\,\mathrm{Mpc}^{-1}$), consistent with free-$\sigma_8$ $\Lambda$CDM ($\chi^2 = 1.12$) and RBIC at $k_\star = 0.51\;h\,\mathrm{Mpc}^{-1}, \sigma_8 = 0.850$ ($\chi^2 = 3.19$).

---

## Repository Structure

```text
RBIC_Foundations/
├── main.tex
├── references.bib
├── figures/
│   ├── thermo_regularization.png
│   ├── rbic_mcmc_corner.png
│   ├── rbic_hz_fit.png
│   ├── rbic_weff_phantom.png
│   └── kids_shear_variance.png
└── supplementary_data/
    ├── README_PROOFS_AND_DATA.txt
    ├── RBIC_Foundations_Proof_Tables.xlsx
    ├── kids_legacy_summary.npz
    ├── kids_legacy_summary.xlsx
    ├── kids_dr5_shear_variance.csv
    ├── mcmc_model_comparison.csv
    ├── mcmc_posterior_percentiles.csv
    ├── weff_phantom_crossing.csv
    └── scripts/
        ├── run_emcee_rbic.py
        ├── extract_kids_legacy_summary.py
        ├── kids_shear_final_figure.py
        ├── generate_rbic_figures.py
        └── verify_all_proofs.py
```

---

## Reproducing the Checks

```bash
pip install numpy scipy pandas emcee matplotlib openpyxl
python supplementary_data/scripts/verify_all_proofs.py
python supplementary_data/scripts/kids_shear_final_figure.py
```
