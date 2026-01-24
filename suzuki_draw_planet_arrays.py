
"""Generate s/q arrays sampled from the Suzuki et al. (2016) mass-ratio function.

This script mirrors the structure of ``sumi2023_draw_planet_arrays.py`` and
``uniform_draw_planet_arrays.py`` but draws only projected separations (``s``)
and planet-to-host mass ratios (``q``). Sampling follows the broken power-law
form reported by Suzuki et al. (2016, arXiv:1612.03939), combining
inverse-transform draws in log-space for ``q`` with a single power-law in ``s``.

Execution (CLI)
---------------
Run directly::

        python suzuki_draw_planet_arrays.py

Configuration
-------------
Modify module-level constants below (``nl``, ``nf``, ``rundes`` etc.). Set
``FIXED_BASE_SEED`` to an integer to obtain reproducible draws across
multiprocessing workers.

Output
------
Each generated file (text or ``.npy``) contains two columns ordered as::

        1. planet-to-host mass ratio q
        2. projected separation s (Einstein-radius units)

Notes
-----
* The joint probability density factorizes into independent log-space
  distributions in ``q`` (broken power-law) and ``s`` (single power-law).
* Existing files are removed when ``overwrite_existing`` is True.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time

import numpy as np
from typing import Optional, Tuple

# Resolve paths relative to this script so it works from any CWD
_BASE_DIR = os.path.dirname(__file__)

# Optional: path to a pickled binned-GMM artifact trained on (i, phi, P) | (s, q)
# If set to a readable file, the script will append three columns per row:
#   i_deg, phi_deg, P_years
# Otherwise, it will output only q and s as before.
GMM_ARTIFACT_PATH: Optional[str] = os.path.join(_BASE_DIR, 'binned_gmm_artifact.pkl')
_GMM_ARTIFACT = None


def _load_gmm_artifact() -> None:
    """Load GMM artifact if available. Only called when running as main script."""
    global _GMM_ARTIFACT
    try:
        if GMM_ARTIFACT_PATH and os.path.exists(GMM_ARTIFACT_PATH):
            # Lazy import to avoid hard dependency when not used
            from binned_gmm import load_artifact
            _GMM_ARTIFACT = load_artifact(GMM_ARTIFACT_PATH)
            print(f"Loaded GMM artifact: {GMM_ARTIFACT_PATH}")
        else:
            if GMM_ARTIFACT_PATH:
                print(f"GMM artifact not found at {GMM_ARTIFACT_PATH}; proceeding without orbital sampling.")
    except Exception as _e:
        print(f"Warning: failed to load GMM artifact: {_e}. Proceeding without orbital sampling.")
        _GMM_ARTIFACT = None

# -------------------------------------------------------------------------
# Suzuki et al. (2016) broken power-law parameters (All sample, q_br fixed)
# -------------------------------------------------------------------------
SUZUKI_A = 0.61             # dex^-2 star^-1 (unused for sampling; provided for reference)
SUZUKI_Q_BREAK = 1.7e-4     # mass-ratio break
SUZUKI_N = -0.92            # slope for q >= q_break (log-space derivative exponent)
SUZUKI_P = 0.44             # slope for q < q_break
SUZUKI_M = 0.50             # separation exponent (per dlog s)
SUZUKI_S0 = 1.0             # separation pivot (s is dimensionless; s0 keeps equation form)

LOG10_Q_BREAK = math.log10(SUZUKI_Q_BREAK)

# Default sampling bounds (tunable)
LOG10_Q_MIN = math.log10(2.6e-5)   # lower limit from Table 6 discussion (~2.6e-5)
LOG10_Q_MAX = math.log10(3.0e-2)   # upper limit used in MOA sample (0.03)
LOG10_S_MIN = math.log10(0.1)      # 0.1 <= s <= 10 range from survey sensitivity
LOG10_S_MAX = math.log10(10.0)

rundes = 'test_suzuki_draw'
sources_file = './gulls_surot2d_H2023.sources'
file_ext = ''
nl = 10000      # draws per file
nf = 1          # files per field
overwrite_existing = True

delineator = ","
header = True
_HEADER_LINE = 'q, s'

FIXED_BASE_SEED: int | None = None


def get_field_numbers(sources_path: str | os.PathLike[str]) -> list[int]:
    """Extract integer field identifiers from a sources file."""
    field_numbers: list[int] = []
    with open(sources_path, 'r') as fh:
        for line in fh:
            if line.strip():
                field_numbers.append(int(line.split()[0]))
    return field_numbers


# Default data directory: current working directory (can be changed)
data_dir = './'
if not data_dir.endswith('/'):
    data_dir += '/'


def _segment_integral(slope: float, lower: float, upper: float, pivot: float) -> float:
    """Integrate 10^{slope * (x - pivot)} between ``lower`` and ``upper``."""
    if upper <= lower:
        return 0.0
    if abs(slope) < 1e-12:
        return upper - lower
    start = 10.0 ** (slope * (lower - pivot))
    stop = 10.0 ** (slope * (upper - pivot))
    return (stop - start) / (slope * math.log(10.0))


def _sample_segment(u: np.ndarray, slope: float, lower: float, upper: float, pivot: float) -> np.ndarray:
    """Inverse-transform sampling for a single power-law segment in log-space."""
    if abs(slope) < 1e-12:
        return lower + (upper - lower) * u
    start = 10.0 ** (slope * (lower - pivot))
    stop = 10.0 ** (slope * (upper - pivot))
    vals = start + u * (stop - start)
    return pivot + np.log10(vals) / slope


def sample_log_break_powerlaw(size: int, *,
                              log_min: float,
                              log_max: float,
                              log_break: float,
                              slope_low: float,
                              slope_high: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Sample from a broken power-law in log10-space with a single break."""
    if log_min >= log_max:
        raise ValueError("log_min must be < log_max")

    low_hi = min(log_break, log_max)
    high_lo = max(log_break, log_min)

    low_weight = _segment_integral(slope_low, log_min, low_hi, log_break)
    high_weight = _segment_integral(slope_high, high_lo, log_max, log_break)
    total = low_weight + high_weight
    if total <= 0:
        raise ValueError("Degenerate sampling interval; total weight is zero")

    u = rng.random(size)
    samples = np.empty(size, dtype=float)

    if low_weight > 0 and high_weight > 0:
        split = low_weight / total
        split = min(max(split, 1e-12), 1.0 - 1e-12)
        low_mask = u < split
        high_mask = ~low_mask

        u_low = u[low_mask] / split
        u_high = (u[high_mask] - split) / (1.0 - split)

        samples[low_mask] = _sample_segment(u_low, slope_low, log_min, low_hi, log_break)
        samples[high_mask] = _sample_segment(u_high, slope_high, high_lo, log_max, log_break)
    elif low_weight > 0:
        samples[:] = _sample_segment(u, slope_low, log_min, low_hi, log_break)
    else:
        samples[:] = _sample_segment(u, slope_high, high_lo, log_max, log_break)

    return samples


def sample_log_powerlaw(size: int, *,
                        log_min: float,
                        log_max: float,
                        slope: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Sample from a single power-law in log10-space."""
    if log_min >= log_max:
        raise ValueError("log_min must be < log_max")
    u = rng.random(size)
    return _sample_segment(u, slope, log_min, log_max, 0.0)


def draw_s_and_q(size: int,
                 *,
                 log10_q_min: float = LOG10_Q_MIN,
                 log10_q_max: float = LOG10_Q_MAX,
                 log10_s_min: float = LOG10_S_MIN,
                 log10_s_max: float = LOG10_S_MAX,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``size`` samples of (q, s) from the Suzuki mass-ratio function."""
    if rng is None:
        rng = np.random.default_rng()
    log_q = sample_log_break_powerlaw(size,
                                      log_min=log10_q_min,
                                      log_max=log10_q_max,
                                      log_break=LOG10_Q_BREAK,
                                      slope_low=SUZUKI_P,
                                      slope_high=SUZUKI_N,
                                      rng=rng)
    log_s = sample_log_powerlaw(size,
                                log_min=log10_s_min,
                                log_max=log10_s_max,
                                slope=SUZUKI_M,
                                rng=rng)
    q = 10.0 ** log_q
    s = 10.0 ** log_s
    return q, s


def _inverse_transform_targets(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of the modeling transform used for (i, phi, P).

    y columns: [mu=cos(i), sin(phi), cos(phi), log10(P_days)]
    Returns i_deg, phi_deg, P_years.
    """
    mu = y[:, 0]
    sin_phi = y[:, 1]
    cos_phi = y[:, 2]
    log10P = y[:, 3]
    inc_rad = np.arccos(np.clip(mu, -1.0, 1.0))
    i_deg = np.rad2deg(inc_rad)
    phi_rad = np.arctan2(sin_phi, cos_phi)
    phi_deg = (np.rad2deg(phi_rad)) % 360.0
    P_days = 10.0 ** log10P
    P_years = P_days / 365.25
    return i_deg, phi_deg, P_years


def _sample_orbits_for_arrays(artifact, s_array: np.ndarray, q_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sampling of (i, phi, P) for many (s, q) pairs.

    Groups by (bin_x, bin_q) to sample efficiently from each bin's GMM.
    Falls back to nearest available bin if an exact bin has no model.
    """
    from binned_gmm import find_bin  # local import
    xs_edges = artifact['xs_edges']
    xq_edges = artifact['xq_edges']
    models = artifact['models']

    xs = np.log10(np.clip(s_array, 1e-12, None))
    xq = np.log10(np.clip(q_array, 1e-12, None))

    # Compute bin indices for all rows
    ix = np.searchsorted(xs_edges, xs, side='right') - 1
    iq = np.searchsorted(xq_edges, xq, side='right') - 1
    ix = np.clip(ix, 0, len(xs_edges) - 2)
    iq = np.clip(iq, 0, len(xq_edges) - 2)

    # Map available model keys for quick nearest-neighbor fallback
    available = np.array(list(models.keys()), dtype=int)

    def _nearest_key(ix0: int, iq0: int) -> Tuple[int, int]:
        if available.size == 0:
            raise KeyError("No fitted bins available in artifact.")
        diffs = np.abs(available - np.array([ix0, iq0]))
        d = diffs.sum(axis=1)  # Manhattan distance over grid
        j = int(np.argmin(d))
        return int(available[j, 0]), int(available[j, 1])

    n = len(s_array)
    out = np.empty((n, 4), dtype=float)

    # Group by (ix, iq)
    keys = (ix.astype(int), iq.astype(int))
    # Build a dict of indices for each key
    bin_map = {}
    for idx, k in enumerate(zip(*keys)):
        bin_map.setdefault(k, []).append(idx)

    for key, idxs in bin_map.items():
        gmm_entry = models.get(key)
        if gmm_entry is None:
            # Fallback to nearest available bin
            nk = _nearest_key(*key)
            gmm_entry = models.get(nk)
        gmm = gmm_entry['gmm']
        k = len(idxs)
        samples = gmm.sample(n_samples=k)[0]
        out[idxs, :] = samples

    i_deg, phi_deg, P_years = _inverse_transform_targets(out)
    return i_deg, phi_deg, P_years


def worker(task: tuple[int, int]) -> None:
    """Generate a single s/q sample file for the provided task."""
    field_number, file_index = task
    base = f"{data_dir}/planets/{rundes}/{rundes}.planets"
    pfile = f"{base}.{field_number}.{file_index}{file_ext}"

    if os.path.exists(pfile):
        if overwrite_existing:
            try:
                os.remove(pfile)
                print(f"Removed existing {pfile} (overwrite enabled).")
            except OSError as exc:
                print(f"Could not remove {pfile}: {exc}")
        else:
            print(f"File {pfile} exists; skipping (overwrite disabled).")
            return

    if FIXED_BASE_SEED is not None:
        local_seed = FIXED_BASE_SEED + field_number * 100003 + file_index
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()

    q_array, s_array = draw_s_and_q(nl, rng=rng)

    # Build output, optionally appending (i, phi, P)
    if _GMM_ARTIFACT is not None:
        try:
            i_deg, phi_deg, P_years = _sample_orbits_for_arrays(_GMM_ARTIFACT, s_array, q_array)
            combined = np.empty((nl, 5), dtype=float)
            combined[:, 0] = q_array
            combined[:, 1] = s_array
            combined[:, 2] = i_deg
            combined[:, 3] = phi_deg
            combined[:, 4] = P_years
            header_line = 'q, s, i_deg, phi_deg, P_years'
        except Exception as exc:
            print(f"Orbital sampling failed ({exc}); writing q,s only.")
            combined = np.empty((nl, 2), dtype=float)
            combined[:, 0] = q_array
            combined[:, 1] = s_array
            header_line = _HEADER_LINE
    else:
        combined = np.empty((nl, 2), dtype=float)
        combined[:, 0] = q_array
        combined[:, 1] = s_array
        header_line = _HEADER_LINE

    if file_ext == '.npy':
        np.save(pfile, combined)
    else:
        save_kwargs = {"delimiter": delineator}
        if header:
            save_kwargs["header"] = header_line
            save_kwargs["comments"] = "# "
        np.savetxt(pfile, combined, **save_kwargs)


def main() -> None:
    """Entry point: orchestrate Suzuki s/q sampling across all survey fields."""
    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    dir_name = f"{data_dir}/planets/{rundes}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # Resolve sources file relative to script if necessary
    src_path = sources_file
    if not os.path.isabs(src_path):
        candidate = os.path.join(_BASE_DIR, os.path.basename(src_path))
        if os.path.exists(candidate):
            src_path = candidate
    field_ids = get_field_numbers(src_path)
    tasks = [(field, i) for field in field_ids for i in range(nf)]
    print(f"Generated {len(tasks)} unique tasks. Handing them off to workers.")
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(worker, tasks)


if __name__ == "__main__":
    _load_gmm_artifact()
    start = time.time()
    main()
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")
