"""Generate multiplanet system arrays for GULLS General lightcurve generator.

This script produces planet files in the new format required by GULLS 3.0.0:
    Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType

Each system consists of 3 rows:
    - OrbitType 1: planet 1
    - OrbitType 2: planet 2  
    - OrbitType 3: moon
Objects with mass=0 indicate no planet/moon in that slot.

Planet count is determined by integrating the Suzuki mass function over a small
bin around each drawn (m, a), treating m=q and a=s. This gives a local expected
planet count, capped at 3 (simulation limit).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from typing import Optional

import numpy as np

# Resolve paths relative to this script
_BASE_DIR = os.path.dirname(__file__)

# -------------------------------------------------------------------------
# Suzuki et al. (2016) broken power-law parameters
# From Table 3, "All" sample with q_br fixed (best-fit values)
#
# Mass ratio function:
#   d²N_pl / (d log q d log s) = A × (q/q_br)^n × s^m   for q >= q_br
#                              = A × (q/q_br)^p × s^m   for q <  q_br
# -------------------------------------------------------------------------
SUZUKI_A = 0.61             # planets/star/dex² at (q_br, s=1)
SUZUKI_Q_BREAK = 1.7e-4     # mass-ratio break (~20 Earth masses for 0.6 Msun host)
SUZUKI_N = -0.93            # slope for q >= q_break (giant planets)
SUZUKI_P = 0.6              # slope for q < q_break (Neptunes/super-Earths)
SUZUKI_M = 0.49             # separation exponent (roughly log-flat)

LOG10_Q_BREAK = math.log10(SUZUKI_Q_BREAK)

# Bin size for local Suzuki integration (dex)
# Using 1.0 matches Suzuki's convention: A=0.61 means 0.61 planets/star per 1×1 dex² bin
LOCAL_BIN_SIZE = 1.0

# -------------------------------------------------------------------------
# Sampling bounds
# -------------------------------------------------------------------------
LOG10_MASS_MIN = math.log10(1e-7)    # ~0.03 Earth masses
LOG10_MASS_MAX = math.log10(3e-2)    # ~10000 Earth masses
LOG10_A_MIN = math.log10(0.3)        # 0.3 AU
LOG10_A_MAX = math.log10(30.0)       # 30 AU

# Moon bounds
LOG10_MOON_A_MIN = math.log10(0.001)
LOG10_MOON_A_MAX = math.log10(0.01)
LOG10_MOON_MASS_MIN = math.log10(1e-9)
LOG10_MOON_MASS_MAX = math.log10(1e-5)

# -------------------------------------------------------------------------
# Orbital element parameters
# -------------------------------------------------------------------------
ECCENTRICITY_SIGMA = 0.3
ECCENTRICITY_MAX = 0.95
PERIOD_RATIO_MIN = 1.3
INCLINATION_BASE = 1000.0
INCLINATION_SCATTER_SIGMA = 5.0
MOON_PROBABILITY = 0.1

# -------------------------------------------------------------------------
# Run configuration
# -------------------------------------------------------------------------
rundes = 'test_multiplanet'
sources_file = './gulls_surot2d_H2023.sources'
file_ext = ''
nl = 1000       # systems per file (reduced for testing)
nf = 1          # files per field
overwrite_existing = True

HEADER_LINE = 'Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType'
DELIMITER = ' '

FIXED_BASE_SEED: int | None = 42  # Set for reproducibility during testing

data_dir = './'
if not data_dir.endswith('/'):
    data_dir += '/'


def get_field_numbers(sources_path: str | os.PathLike[str]) -> list[int]:
    """Extract integer field identifiers from a sources file."""
    field_numbers: list[int] = []
    with open(sources_path, 'r') as fh:
        for line in fh:
            if line.strip():
                field_numbers.append(int(line.split()[0]))
    return field_numbers


# -------------------------------------------------------------------------
# Suzuki mass function evaluation
# -------------------------------------------------------------------------

def suzuki_density(log_q: float, log_s: float) -> float:
    """Evaluate Suzuki mass function at a point.
    
    Returns d²N/(d log q d log s) at the given (log_q, log_s).
    
    Parameters
    ----------
    log_q : float
        log10 of mass ratio (or mass in solar masses as proxy).
    log_s : float
        log10 of separation (in AU, as proxy for Einstein radius units).
    
    Returns
    -------
    float
        Mass function density (planets per star per dex²).
    """
    # q-dependence (broken power law)
    if log_q < LOG10_Q_BREAK:
        q_factor = 10.0 ** (SUZUKI_P * (log_q - LOG10_Q_BREAK))
    else:
        q_factor = 10.0 ** (SUZUKI_N * (log_q - LOG10_Q_BREAK))
    
    # s-dependence (single power law, pivot at s=1)
    s_factor = 10.0 ** (SUZUKI_M * log_s)
    
    return SUZUKI_A * q_factor * s_factor


def local_expected_planets(log_m: float, log_a: float, 
                           bin_size: float = LOCAL_BIN_SIZE) -> float:
    """Integrate Suzuki mass function to get expected planets per star.
    
    From Suzuki et al. (2016):
        d²N_pl / (d log q d log s) = A × (q/q_br)^n × s^m  [planets/star/dex²]
    
    Integrating over a bin_size × bin_size dex² region:
        N_pl = density × bin_size²  [planets/star]
    
    With A=0.61 and bin_size=1.0, at the break (q_br, s=1) we get
    0.61 planets per star — matching the paper's convention.
    
    Parameters
    ----------
    log_m : float
        log10 of planet mass (solar masses), used as proxy for q.
    log_a : float
        log10 of semi-major axis (AU), used as proxy for s.
    bin_size : float
        Size of integration bin in dex.
    
    Returns
    -------
    float
        Expected number of planets per star in the bin.
    """
    density = suzuki_density(log_m, log_a)  # planets/star/dex²
    area = bin_size ** 2  # dex²
    return density * area  # planets/star


# -------------------------------------------------------------------------
# Vectorized sampling functions
# -------------------------------------------------------------------------

def draw_log_uniform_vec(size: int, log_min: float, log_max: float, 
                         rng: np.random.Generator) -> np.ndarray:
    """Draw log-uniform samples (vectorized)."""
    log_vals = log_min + (log_max - log_min) * rng.random(size)
    return 10.0 ** log_vals


def draw_eccentricity_vec(size: int, rng: np.random.Generator,
                          sigma: float = ECCENTRICITY_SIGMA,
                          max_ecc: float = ECCENTRICITY_MAX) -> np.ndarray:
    """Draw eccentricities (vectorized with rejection sampling)."""
    ecc = np.abs(rng.normal(0.0, sigma, size))
    mask = ecc > max_ecc
    n_bad = mask.sum()
    while n_bad > 0:
        ecc[mask] = np.abs(rng.normal(0.0, sigma, n_bad))
        mask = ecc > max_ecc
        n_bad = mask.sum()
    return ecc


def check_period_ratio(a1: float, a2: float) -> bool:
    """Check if period ratio > PERIOD_RATIO_MIN."""
    if a1 <= 0 or a2 <= 0:
        return True
    ratio = max(a2/a1, a1/a2) ** 1.5
    return ratio >= PERIOD_RATIO_MIN


# -------------------------------------------------------------------------
# System generation
# -------------------------------------------------------------------------

def generate_system(rng: np.random.Generator) -> np.ndarray:
    """Generate a single planetary system (3 rows).
    
    Uses local Suzuki density at drawn (m, a) to determine planet count.
    """
    system = np.zeros((3, 7), dtype=float)
    system[0, 6] = 1  # OrbitType
    system[1, 6] = 2
    system[2, 6] = 3
    
    # First, draw a candidate (m, a) to evaluate local density
    log_m = LOG10_MASS_MIN + (LOG10_MASS_MAX - LOG10_MASS_MIN) * rng.random()
    log_a = LOG10_A_MIN + (LOG10_A_MAX - LOG10_A_MIN) * rng.random()
    
    # Get expected planets from local Suzuki density
    expected = local_expected_planets(log_m, log_a)
    
    # Draw planet count from Poisson, cap at 2 (plus possible moon = 3 total)
    n_planets = min(rng.poisson(expected), 2)
    
    if n_planets == 0:
        return system
    
    # Planet 1 uses the already-drawn (m, a)
    m1 = 10.0 ** log_m
    a1 = 10.0 ** log_a
    ecc1 = draw_eccentricity_vec(1, rng)[0]
    inc1 = INCLINATION_BASE + rng.normal(0.0, INCLINATION_SCATTER_SIGMA)
    omega1 = 360.0 * rng.random()
    Omega1 = 360.0 * rng.random()
    system[0, :] = [m1, a1, ecc1, inc1, omega1, Omega1, 1]
    
    if n_planets >= 2:
        # Draw second planet, ensuring period ratio constraint
        for _ in range(50):  # Max attempts
            log_m2 = LOG10_MASS_MIN + (LOG10_MASS_MAX - LOG10_MASS_MIN) * rng.random()
            log_a2 = LOG10_A_MIN + (LOG10_A_MAX - LOG10_A_MIN) * rng.random()
            a2 = 10.0 ** log_a2
            if check_period_ratio(a1, a2):
                m2 = 10.0 ** log_m2
                ecc2 = draw_eccentricity_vec(1, rng)[0]
                inc2 = INCLINATION_BASE + rng.normal(0.0, INCLINATION_SCATTER_SIGMA)
                omega2 = 360.0 * rng.random()
                Omega2 = 360.0 * rng.random()
                system[1, :] = [m2, a2, ecc2, inc2, omega2, Omega2, 2]
                break
    
    # Possibly add moon
    if rng.random() < MOON_PROBABILITY and n_planets > 0:
        moon_m = draw_log_uniform_vec(1, LOG10_MOON_MASS_MIN, LOG10_MOON_MASS_MAX, rng)[0]
        moon_a = draw_log_uniform_vec(1, LOG10_MOON_A_MIN, LOG10_MOON_A_MAX, rng)[0]
        moon_ecc = draw_eccentricity_vec(1, rng, sigma=0.1)[0]
        moon_inc = INCLINATION_BASE + rng.normal(0.0, 10.0)
        moon_omega = 360.0 * rng.random()
        moon_Omega = 360.0 * rng.random()
        system[2, :] = [moon_m, moon_a, moon_ecc, moon_inc, moon_omega, moon_Omega, 3]
    
    return system


def generate_systems_batch(n_systems: int, rng: np.random.Generator,
                           benchmark: bool = False) -> np.ndarray:
    """Generate multiple systems with optional benchmarking."""
    t0 = time.perf_counter()
    
    systems = []
    for i in range(n_systems):
        systems.append(generate_system(rng))
        
        # Progress every 10%
        if benchmark and (i + 1) % (n_systems // 10 or 1) == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_systems} systems ({rate:.1f}/sec)")
    
    combined = np.vstack(systems)
    
    if benchmark:
        elapsed = time.perf_counter() - t0
        print(f"  Generated {n_systems} systems in {elapsed:.2f}s ({n_systems/elapsed:.1f}/sec)")
    
    return combined


def worker(task: tuple[int, int]) -> dict:
    """Generate a planet file. Returns timing info."""
    field_number, file_index = task
    timings = {'field': field_number, 'index': file_index}
    
    t_start = time.perf_counter()
    
    base = f"{data_dir}/planets/{rundes}/{rundes}.planets"
    pfile = f"{base}.{field_number}.{file_index}{file_ext}"
    
    if os.path.exists(pfile):
        if overwrite_existing:
            os.remove(pfile)
        else:
            return timings
    
    # RNG setup
    t_rng = time.perf_counter()
    if FIXED_BASE_SEED is not None:
        local_seed = FIXED_BASE_SEED + field_number * 100003 + file_index
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()
    timings['rng_setup'] = time.perf_counter() - t_rng
    
    # Generate systems
    t_gen = time.perf_counter()
    combined = generate_systems_batch(nl, rng, benchmark=False)
    timings['generation'] = time.perf_counter() - t_gen
    
    # Write file
    t_write = time.perf_counter()
    if file_ext == '.npy':
        np.save(pfile, combined)
    else:
        fmt = ['%.6e', '%.6e', '%.6f', '%.4f', '%.4f', '%.4f', '%d']
        np.savetxt(pfile, combined, delimiter=DELIMITER, header=HEADER_LINE, 
                   comments='', fmt=fmt)
    timings['write'] = time.perf_counter() - t_write
    
    timings['total'] = time.perf_counter() - t_start
    return timings


def main() -> None:
    """Entry point with benchmarking."""
    print("=" * 60)
    print("MULTIPLANET GENERATOR - BENCHMARKED RUN")
    print("=" * 60)
    
    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    
    print(f"\nConfiguration:")
    print(f"  Systems per file: {nl}")
    print(f"  Files per field: {nf}")
    print(f"  Local bin size: {LOCAL_BIN_SIZE} dex")
    
    # Test local density calculation
    print(f"\nSuzuki density sanity check:")
    test_points = [
        (LOG10_MASS_MIN, LOG10_A_MIN),
        (LOG10_Q_BREAK, 0.0),  # at break, s=1
        (LOG10_MASS_MAX, LOG10_A_MAX),
    ]
    for log_m, log_a in test_points:
        d = suzuki_density(log_m, log_a)
        exp = local_expected_planets(log_m, log_a)
        print(f"  (log_m={log_m:.2f}, log_a={log_a:.2f}): density={d:.4f}, expected={exp:.4f}")
    
    # Create output directory
    dir_name = f"{data_dir}/planets/{rundes}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Get field numbers
    src_path = sources_file
    if not os.path.isabs(src_path):
        candidate = os.path.join(_BASE_DIR, os.path.basename(src_path))
        if os.path.exists(candidate):
            src_path = candidate
    
    t_fields = time.perf_counter()
    field_ids = get_field_numbers(src_path)
    print(f"\nLoaded {len(field_ids)} fields in {time.perf_counter()-t_fields:.3f}s")
    
    # Limit fields for testing
    max_fields = 5  # Only process 5 fields for benchmarking
    field_ids = field_ids[:max_fields]
    tasks = [(field, i) for field in field_ids for i in range(nf)]
    
    print(f"Processing {len(tasks)} tasks (limited to {max_fields} fields for testing)")
    
    # Single-threaded for clear benchmarking
    print(f"\n--- Running single-threaded for clear timing ---")
    all_timings = []
    t_total = time.perf_counter()
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)}: field={task[0]}, index={task[1]}")
        timings = worker(task)
        all_timings.append(timings)
        print(f"  RNG setup: {timings.get('rng_setup', 0)*1000:.1f}ms")
        print(f"  Generation: {timings.get('generation', 0):.2f}s")
        print(f"  Write: {timings.get('write', 0)*1000:.1f}ms")
        print(f"  Total: {timings.get('total', 0):.2f}s")
    
    elapsed_total = time.perf_counter() - t_total
    
    # Summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_total:.2f}s")
    print(f"Tasks completed: {len(all_timings)}")
    if all_timings:
        avg_gen = np.mean([t.get('generation', 0) for t in all_timings])
        avg_write = np.mean([t.get('write', 0) for t in all_timings])
        print(f"Avg generation time: {avg_gen:.2f}s ({nl/avg_gen:.0f} systems/sec)")
        print(f"Avg write time: {avg_write*1000:.1f}ms")


if __name__ == "__main__":
    main()
