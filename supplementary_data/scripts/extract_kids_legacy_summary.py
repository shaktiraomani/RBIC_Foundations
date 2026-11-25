import gzip
import zlib
import time
import numpy as np

t0 = time.time()
PATH = r"D:\local\KiDS_Legacy_NS_unblind_final.fits.gz"
OUT = r"C:\Users\dell pc\.gemini\antigravity\brain\41eae9ac-a897-4b16-8eaf-aee5c21d3a3c\scratch\kids_legacy_summary.npz"

N_ROWS = 40894394
ROW_BYTES = 401
CHUNK_ROWS = 500000
CHUNK_BYTES = CHUNK_ROWS * ROW_BYTES

dt = np.dtype({
    "names": ["ra", "dec", "zb", "tbin", "gw", "e1", "e2", "w"],
    "formats": [">f8", ">f8", ">f8", ">i4", ">f8", ">f8", ">f8", ">f8"],
    "offsets": [17, 25, 165, 181, 369, 377, 385, 393],
    "itemsize": ROW_BYTES,
})

# Tomographic bins 1..6 N(z) histograms
z_edges = np.linspace(0.0, 2.2, 45)
nz_hist = np.zeros((6, len(z_edges) - 1), dtype=np.float64)
bin_counts = np.zeros(6, dtype=np.int64)
bin_wsum = np.zeros(6, dtype=np.float64)
bin_wzsum = np.zeros(6, dtype=np.float64)
bin_we2sum = np.zeros(6, dtype=np.float64)  # sum(w^2 * (e1^2 + e2^2))
bin_w2sum = np.zeros(6, dtype=np.float64)   # sum(w^2)

# Angular grid for real cosmic shear cell excess variance (RA x Dec cells at 0.25 deg base)
# KiDS spans RA in [0, 360], Dec in [-38, +6]. Map to equal-area grid using cos(dec)*RA:
# Fine cells of size 0.25 deg -> 1440 x 200 = 288,000 cells (for bin 5+6 combined high-z shear, and all-bins)
dtheta_base = 0.25
n_x = int(np.ceil(360.0 / dtheta_base))
n_y = int(np.ceil(50.0 / dtheta_base))  # Dec in [-40, +10]
n_cells = n_x * n_y

# Store sums per base cell for TOMOBIN >= 4 (bins 4,5,6 -> z ~ 0.65 - 2.0, highest lensing S/N)
cell_w = np.zeros(n_cells, dtype=np.float64)
cell_w2 = np.zeros(n_cells, dtype=np.float64)
cell_we1 = np.zeros(n_cells, dtype=np.float64)
cell_we2 = np.zeros(n_cells, dtype=np.float64)
cell_w2e2 = np.zeros(n_cells, dtype=np.float64)
cell_cnt = np.zeros(n_cells, dtype=np.int64)

rows_read = 0
with gzip.open(PATH, "rb") as f:
    f.read(6 * 2880)  # skip exact FITS headers
    while rows_read < N_ROWS:
        to_read = min(CHUNK_ROWS, N_ROWS - rows_read)
        try:
            buf = f.read(to_read * ROW_BYTES)
        except (zlib.error, EOFError, OSError) as exc:
            print(f"Stopped clean stream at {rows_read:,} rows due to truncated/corrupt gzip block: {exc}", flush=True)
            break
        if len(buf) < ROW_BYTES:
            break
        n_got = len(buf) // ROW_BYTES
        arr = np.frombuffer(buf[:n_got * ROW_BYTES], dtype=dt)
        rows_read += n_got

        w = arr["w"]
        tbin = arr["tbin"]
        zb = arr["zb"]
        e1 = arr["e1"]
        e2 = arr["e2"]
        ra = arr["ra"]
        dec = arr["dec"]

        valid = (w > 0) & (tbin >= 1) & (tbin <= 6) & np.isfinite(e1) & np.isfinite(e2)
        if not np.any(valid):
            continue

        w_v = w[valid]
        tb_v = tbin[valid] - 1
        zb_v = zb[valid]
        e1_v = e1[valid]
        e2_v = e2[valid]
        ra_v = ra[valid]
        dec_v = dec[valid]

        for b in range(6):
            m = (tb_v == b)
            if np.any(m):
                wb = w_v[m]
                e2b = e1_v[m]**2 + e2_v[m]**2
                bin_counts[b] += int(m.sum())
                bin_wsum[b] += float(np.sum(wb))
                bin_wzsum[b] += float(np.sum(wb * zb_v[m]))
                bin_w2sum[b] += float(np.sum(wb * wb))
                bin_we2sum[b] += float(np.sum(wb * wb * e2b))
                h, _ = np.histogram(zb_v[m], bins=z_edges, weights=wb)
                nz_hist[b] += h

        # Accumulate into 0.25 deg cells for high-z tomographic bins (4, 5, 6 -> indices 3, 4, 5)
        mh = (tb_v >= 3) & (dec_v >= -40.0) & (dec_v < 10.0)
        if np.any(mh):
            ix = np.clip((ra_v[mh] / dtheta_base).astype(np.int64), 0, n_x - 1)
            iy = np.clip(((dec_v[mh] + 40.0) / dtheta_base).astype(np.int64), 0, n_y - 1)
            cid = iy * n_x + ix
            wh = w_v[mh]
            w2h = wh * wh
            e1h = e1_v[mh]
            e2h = e2_v[mh]
            cell_w += np.bincount(cid, weights=wh, minlength=n_cells)
            cell_w2 += np.bincount(cid, weights=w2h, minlength=n_cells)
            cell_we1 += np.bincount(cid, weights=wh * e1h, minlength=n_cells)
            cell_we2 += np.bincount(cid, weights=wh * e2h, minlength=n_cells)
            cell_w2e2 += np.bincount(cid, weights=w2h * (e1h**2 + e2h**2), minlength=n_cells)
            cell_cnt += np.bincount(cid, minlength=n_cells)

        if rows_read % 10000000 < CHUNK_ROWS:
            print(f"Streamed {rows_read:,}/{N_ROWS:,} galaxies ({time.time()-t0:.1f}s)...", flush=True)

# Compute multi-scale cell shear excess variance across bin-4,5,6 sky footprint
scales_deg = [0.25, 0.50, 1.00, 2.00]
excess_var = []
excess_err = []

c_w = cell_w.reshape(n_y, n_x)
c_w2 = cell_w2.reshape(n_y, n_x)
c_we1 = cell_we1.reshape(n_y, n_x)
c_we2 = cell_we2.reshape(n_y, n_x)
c_w2e2 = cell_w2e2.reshape(n_y, n_x)
c_cnt = cell_cnt.reshape(n_y, n_x)

for sc in scales_deg:
    f_blk = int(round(sc / dtheta_base))
    ny_b = (n_y // f_blk) * f_blk
    nx_b = (n_x // f_blk) * f_blk
    bw = c_w[:ny_b, :nx_b].reshape(ny_b//f_blk, f_blk, nx_b//f_blk, f_blk).sum(axis=(1,3))
    bw2 = c_w2[:ny_b, :nx_b].reshape(ny_b//f_blk, f_blk, nx_b//f_blk, f_blk).sum(axis=(1,3))
    bwe1 = c_we1[:ny_b, :nx_b].reshape(ny_b//f_blk, f_blk, nx_b//f_blk, f_blk).sum(axis=(1,3))
    bwe2 = c_we2[:ny_b, :nx_b].reshape(ny_b//f_blk, f_blk, nx_b//f_blk, f_blk).sum(axis=(1,3))
    bw2e2 = c_w2e2[:ny_b, :nx_b].reshape(ny_b//f_blk, f_blk, nx_b//f_blk, f_blk).sum(axis=(1,3))
    bcnt = c_cnt[:ny_b, :nx_b].reshape(ny_b//f_blk, f_blk, nx_b//f_blk, f_blk).sum(axis=(1,3))

    good = bcnt >= max(50, 20 * f_blk * f_blk)
    g1_bar = bwe1[good] / bw[good]
    g2_bar = bwe2[good] / bw[good]
    # Remove tiny additive c-term (mean <e>) per catalog patch
    g1_bar -= np.mean(g1_bar)
    g2_bar -= np.mean(g2_bar)
    raw_var = g1_bar**2 + g2_bar**2
    noise_var = bw2e2[good] / (bw[good]**2)
    ex = raw_var - noise_var
    mean_ex = float(np.mean(ex))
    err_ex = float(np.std(ex, ddof=1) / np.sqrt(len(ex)))
    excess_var.append(mean_ex)
    excess_err.append(err_ex)
    print(f"Scale {sc:.2f} deg: {len(ex)} cells | cosmic shear excess var = {mean_ex:.3e} +/- {err_ex:.3e}", flush=True)

np.savez(
    OUT,
    z_edges=z_edges,
    nz_hist=nz_hist,
    bin_counts=bin_counts,
    bin_wsum=bin_wsum,
    bin_wzsum=bin_wzsum,
    bin_w2sum=bin_w2sum,
    bin_we2sum=bin_we2sum,
    scales_deg=np.array(scales_deg),
    excess_var=np.array(excess_var),
    excess_err=np.array(excess_err),
)
print(f"\nSaved KiDS-Legacy 40.9M summary to {OUT} in {time.time()-t0:.1f}s!")
for b in range(6):
    zmean = bin_wzsum[b] / max(bin_wsum[b], 1.0)
    sig_e = np.sqrt(0.5 * bin_we2sum[b] / max(bin_w2sum[b], 1.0))
    print(f"  Bin {b+1}: N={bin_counts[b]:9,d}, <z>={zmean:.3f}, sigma_e={sig_e:.3f}")
