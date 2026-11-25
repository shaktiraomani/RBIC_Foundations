RBIC FOUNDATIONAL PAPER — DATA, PROOFS & REPRODUCIBILITY PACKAGE
==================================================================
Paper: "Relativistic Bose Interval Cosmology: Unified Modified Measure
        Theory, Vacuum Thermodynamics, and Multi-Probe Cosmology"
Folder: RBIC_Foundations_arXiv/supplementary_data/

1. WHAT PROBLEMS DOES THIS PAPER SOLVE?
---------------------------------------
(A) Origin of the MMT Measure Field:
    Identifies the auxiliary measure field phi of Modified Measure Theory (MMT)
    with the macroscopic wavefunction Psi_I of an ultralight Bose-Einstein
    Condensate (BEC) dark sector. The conformal mismatch generates spacetime
    torsion and a physical Interval scale Delta L from the quantum-torsional
    Raychaudhuri identity.

(B) Cosmological Constant Catastrophe (~10^120 discrepancy):
    Couples Gibbons-Hawking apparent-horizon thermodynamics to the cold BEC
    bath, yielding a non-perturbative Boltzmann damping factor
    exp[-xi_RB * (E(z) - 1)] that regularizes the vacuum energy without
    fine-tuning.

(C) Dynamical Dark Energy & Phantom Crossing (DESI DR2):
    Derives the modified Friedmann equation:
      E^2(z) = Om*(1+z)^3 + Or*(1+z)^4 + (1-Om-Or)*(1+z)^gamma_T * exp[-xi_RB*(E(z)-1)]
    which analytically predicts a phantom-divide crossing (w_eff = -1) when
      (1+z) * dE/dz|_{z_ph} = gamma_T / xi_RB.
    Across Pantheon+, DES-SN5YR, and Union3, unconstrained emcee chains
    converge to gamma_T / xi_RB = 1.16 - 1.31 (z_ph ~ 0.49).
    Locking gamma_T = 1.20 * xi_RB (1-parameter RBIC_1p, k=5) beats BOTH
    LCDM and w0waCDM in AIC and BIC across all three SN suites:
      - Pantheon+ (N=1642):  Delta AIC = -6.82,  Delta BIC = -1.41
      - DES-SN5YR (N=1872):  Delta AIC = -8.71,  Delta BIC = -3.17
      - Union3    (N=74):    Delta AIC = -10.05, Delta BIC = -7.74

(D) Small-Scale Weak Lensing Suppression (KiDS-Legacy DR5):
    Measures real-space aperture excess shear variance across 13.5M high-z
    galaxies (bins 4-6) from the unblinded KiDS-Legacy DR5 catalog, finding
    k_star = 0.51 +0.14/-0.10 h/Mpc (Delta L ~ 18 Mpc), improving chi2 from
    23.27 (unsuppressed CDM) to 3.46 (Delta chi2 = -19.81, >4 sigma).

(E) Preserves All Previous LCDM & GR Successes:
    In the exact limit gamma_T -> 0, xi_RB -> 0 (and m_eff -> infinity),
    RBIC reduces identically to flat LCDM: E_RBIC(z) == E_LCDM(z),
    preserving BBN, CMB acoustic peaks, and c_T = c (LVK O4b).

2. CONTENTS OF THIS DIRECTORY
-----------------------------
- RBIC_Foundations_Proof_Tables.xlsx : 6-sheet Excel workbook containing
    Executive_Summary, MCMC_Model_Comparison, Posterior_Percentiles,
    Phantom_Crossing_weff_z, KiDS_DR5_Shear_Variance, KiDS_DR5_Tomo_Bins.
- mcmc_model_comparison.csv          : Full Table 1 (AIC/BIC/H0/Om/S8).
- mcmc_posterior_percentiles.csv     : Exact P16/P50/P84 from all 6 emcee chains.
- weff_phantom_crossing.csv          : Redshift table of E(z), H(z)/(1+z), w_eff(z).
- kids_dr5_shear_variance.csv        : Multi-scale KiDS-Legacy DR5 shear data.
- kids_legacy_summary.npz            : Extracted N(z) and shear variance arrays.
- scripts/run_emcee_rbic.py          : Full 6-chain real-data emcee pipeline.
- scripts/generate_rbic_figures.py   : Generates corner, H(z), and w_eff(z) plots.
- scripts/extract_kids_legacy_summary.py : Streams KiDS-Legacy DR5 FITS catalog.
- scripts/kids_shear_final_figure.py : Generates kids_shear_variance.png.
- scripts/verify_all_proofs.py       : Standalone proof & identity verifier.

3. HOW TO RUN VERIFICATION
--------------------------
  python scripts/verify_all_proofs.py
