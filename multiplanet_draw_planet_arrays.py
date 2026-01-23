"""Generate multiplanet system arrays for GULLS General lightcurve generator.

This script produces planet files in the new format required by GULLS 3.0.0:
    Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType

Each system consists of 3 rows:
    - OrbitType 1: planet 1
    - OrbitType 2: planet 2  
    - OrbitType 3: moon
Objects with mass=0 indicate no planet/moon in that slot.

Planet properties (mass, semi-major axis) are drawn FROM the Suzuki et al. (2016)
mass ratio function:
    d²N_pl / (d log q d log s) = A × (q/q_br)^n × s^m

The total expected planets per star is computed by integrating this over the
(q, s) bounds. Planet count is drawn from Poisson, capped at 2 (simulation limit).
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

# Precomputed total expected planets per star (computed at module load)
_TOTAL_EXPECTED_PLANETS: float | None = None

# -------------------------------------------------------------------------
# Sampling bounds
# -------------------------------------------------------------------------
LOG10_MASS_MIN = math.log10(1e-7)    # ~0.03 Earth masses
LOG10_MASS_MAX = math.log10(3e-2)    # ~10000 Earth masses
LOG10_A_MIN = math.log10(0.3)        # 0.3 AU
LOG10_A_MAX = math.log10(30.0)       # 30 AU

# Moon bounds
# SMA is relative to planet, range is [0.1 AU, Hill radius]
LOG10_MOON_A_MIN = math.log10(0.1)  # 0.1 AU minimum (smaller = undetectable)
# Moon mass: order of our Moon (~3.7e-8 M☉) to Neptune (~5e-5 M☉)
LOG10_MOON_MASS_MIN = math.log10(1e-8)   # ~3 lunar masses
LOG10_MOON_MASS_MAX = math.log10(1e-4)   # ~3 Neptune masses

# -------------------------------------------------------------------------
# Orbital element parameters
# -------------------------------------------------------------------------
ECCENTRICITY_SIGMA = 0.3
ECCENTRICITY_MAX = 0.95
PERIOD_RATIO_MIN = 1.3
INCLINATION_BASE = 1000.0
INCLINATION_SCATTER_SIGMA = 5.0

# Moon parameters
MOON_PROBABILITY = 0.1          # Probability of moon if planet exists (tunable)
HOST_STAR_MASS = 1.0            # Host star mass in M☉ (for Hill radius calculation)

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
# Suzuki mass function: integration and sampling
# -------------------------------------------------------------------------

def _integrate_power_law(slope: float, x_min: float, x_max: float) -> float:
    """Integrate 10^(slope * x) over [x_min, x_max] in log space.
    
    ∫ 10^(slope*x) dx = 10^(slope*x) / (slope * ln(10))
    """
    if abs(slope) < 1e-10:
        return x_max - x_min  # Flat case
    c = slope * math.log(10)
    return (10.0**(slope * x_max) - 10.0**(slope * x_min)) / c


def compute_total_expected_planets() -> float:
    """Integrate Suzuki over full (q, s) bounds to get planets per star.
    
    The integral is separable:
        N = A × I_q × I_s
    where I_q and I_s are the integrals over log q and log s.
    """
    # Integral over log s: ∫ s^m d(log s) = ∫ 10^(m*log_s) d(log_s)
    I_s = _integrate_power_law(SUZUKI_M, LOG10_A_MIN, LOG10_A_MAX)
    
    # Integral over log q: broken power law
    # Need to split at q_br
    log_q_min = LOG10_MASS_MIN
    log_q_max = LOG10_MASS_MAX
    
    I_q = 0.0
    if log_q_min < LOG10_Q_BREAK:
        # Below break: (q/q_br)^p = 10^(p * (log_q - log_q_br))
        upper = min(LOG10_Q_BREAK, log_q_max)
        I_q += _integrate_power_law(SUZUKI_P, log_q_min - LOG10_Q_BREAK, upper - LOG10_Q_BREAK)
    
    if log_q_max > LOG10_Q_BREAK:
        # Above break: (q/q_br)^n = 10^(n * (log_q - log_q_br))
        lower = max(LOG10_Q_BREAK, log_q_min)
        I_q += _integrate_power_law(SUZUKI_N, lower - LOG10_Q_BREAK, log_q_max - LOG10_Q_BREAK)
    
    return SUZUKI_A * I_q * I_s


def sample_log_s(rng: np.random.Generator) -> float:
    """Sample log10(s) from the Suzuki s distribution: P(log s) ∝ s^m."""
    # CDF: F(x) = ∫_{x_min}^x 10^(m*t) dt / I_s
    # Inverse: x = log10( u * I_s * m * ln(10) + 10^(m*x_min) ) / m
    
    if abs(SUZUKI_M) < 1e-10:
        # Flat: uniform in log s
        return LOG10_A_MIN + (LOG10_A_MAX - LOG10_A_MIN) * rng.random()
    
    c = SUZUKI_M * math.log(10)
    I_s = _integrate_power_law(SUZUKI_M, LOG10_A_MIN, LOG10_A_MAX)
    u = rng.random()
    
    # Inverse CDF
    val = u * I_s * c + 10.0**(SUZUKI_M * LOG10_A_MIN)
    return math.log10(val) / SUZUKI_M


def sample_log_q(rng: np.random.Generator) -> float:
    """Sample log10(q) from the Suzuki broken power law distribution."""
    log_q_min = LOG10_MASS_MIN
    log_q_max = LOG10_MASS_MAX
    
    # Compute probability mass in each region
    I_low = 0.0
    I_high = 0.0
    
    if log_q_min < LOG10_Q_BREAK:
        upper = min(LOG10_Q_BREAK, log_q_max)
        I_low = _integrate_power_law(SUZUKI_P, log_q_min - LOG10_Q_BREAK, upper - LOG10_Q_BREAK)
    
    if log_q_max > LOG10_Q_BREAK:
        lower = max(LOG10_Q_BREAK, log_q_min)
        I_high = _integrate_power_law(SUZUKI_N, lower - LOG10_Q_BREAK, log_q_max - LOG10_Q_BREAK)
    
    I_total = I_low + I_high
    p_low = I_low / I_total
    
    u = rng.random()
    
    if u < p_low:
        # Sample from low-q region (q < q_br)
        slope = SUZUKI_P
        x_min = log_q_min - LOG10_Q_BREAK
        x_max = min(LOG10_Q_BREAK, log_q_max) - LOG10_Q_BREAK
        
        if abs(slope) < 1e-10:
            x = x_min + (x_max - x_min) * (u / p_low)
        else:
            c = slope * math.log(10)
            I_region = I_low
            u_scaled = (u / p_low) * I_region * c + 10.0**(slope * x_min)
            x = math.log10(u_scaled) / slope
        
        return x + LOG10_Q_BREAK
    else:
        # Sample from high-q region (q >= q_br)
        slope = SUZUKI_N
        x_min = max(LOG10_Q_BREAK, log_q_min) - LOG10_Q_BREAK
        x_max = log_q_max - LOG10_Q_BREAK
        
        if abs(slope) < 1e-10:
            x = x_min + (x_max - x_min) * ((u - p_low) / (1 - p_low))
        else:
            c = slope * math.log(10)
            I_region = I_high
            u_scaled = ((u - p_low) / (1 - p_low)) * I_region * c + 10.0**(slope * x_min)
            x = math.log10(u_scaled) / slope
        
        return x + LOG10_Q_BREAK


def sample_suzuki(rng: np.random.Generator) -> tuple[float, float]:
    """Sample (log_q, log_s) from the Suzuki distribution.
    
    Returns
    -------
    tuple[float, float]
        (log10_mass, log10_sma) drawn from Suzuki.
    """
    log_q = sample_log_q(rng)
    log_s = sample_log_s(rng)
    return log_q, log_s


# -------------------------------------------------------------------------
# Hill radius calculation
# -------------------------------------------------------------------------

def compute_hill_radius(a_planet: float, e_planet: float, m_planet: float,
                        m_star: float = HOST_STAR_MASS) -> float:
    """Compute the Hill radius for a planet.
    
    R_H ≈ a(1-e) × (m_planet / (3(m_star + m_planet)))^(1/3)
    
    Parameters
    ----------
    a_planet : float
        Planet semi-major axis (AU).
    e_planet : float
        Planet eccentricity.
    m_planet : float
        Planet mass (M☉).
    m_star : float
        Host star mass (M☉).
    
    Returns
    -------
    float
        Hill radius in AU.
    """
    mass_ratio = m_planet / (3.0 * (m_star + m_planet))
    return a_planet * (1.0 - e_planet) * (mass_ratio ** (1.0 / 3.0))


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

def generate_system(rng: np.random.Generator, expected_planets: float) -> np.ndarray:
    """Generate a single planetary system (3 rows).
    
    Draws planet count from Poisson(expected_planets), then samples each
    planet's (m, a) from the Suzuki distribution.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    expected_planets : float
        Expected planets per star from integrated Suzuki.
    """
    system = np.zeros((3, 7), dtype=float)
    system[0, 6] = 1  # OrbitType
    system[1, 6] = 2
    system[2, 6] = 3
    
    # Draw planet count from Poisson, cap at 2
    n_planets = min(rng.poisson(expected_planets), 2)
    
    if n_planets == 0:
        return system
    
    # Planet 1: sample from Suzuki
    log_m1, log_a1 = sample_suzuki(rng)
    m1 = 10.0 ** log_m1
    a1 = 10.0 ** log_a1
    ecc1 = draw_eccentricity_vec(1, rng)[0]
    inc1 = INCLINATION_BASE + rng.normal(0.0, INCLINATION_SCATTER_SIGMA)
    omega1 = 360.0 * rng.random()
    Omega1 = 360.0 * rng.random()
    system[0, :] = [m1, a1, ecc1, inc1, omega1, Omega1, 1]
    
    if n_planets >= 2:
        # Draw second planet from Suzuki, ensuring period ratio constraint
        for _ in range(50):  # Max attempts
            log_m2, log_a2 = sample_suzuki(rng)
            a2 = 10.0 ** log_a2
            if check_period_ratio(a1, a2):
                m2 = 10.0 ** log_m2
                ecc2 = draw_eccentricity_vec(1, rng)[0]
                inc2 = INCLINATION_BASE + rng.normal(0.0, INCLINATION_SCATTER_SIGMA)
                omega2 = 360.0 * rng.random()
                Omega2 = 360.0 * rng.random()
                system[1, :] = [m2, a2, ecc2, inc2, omega2, Omega2, 2]
                break
    
    # Possibly add moon around planet 1
    # Moon SMA is relative to planet, bounded by [0.1 AU, Hill radius]
    if n_planets > 0 and rng.random() < MOON_PROBABILITY:
        # Use planet 1's properties for Hill radius
        r_hill = compute_hill_radius(a1, ecc1, m1)
        
        # Only add moon if Hill radius > minimum SMA
        a_moon_min = 10.0 ** LOG10_MOON_A_MIN  # 0.1 AU
        if r_hill > a_moon_min:
            log_a_max = math.log10(r_hill)
            moon_m = draw_log_uniform_vec(1, LOG10_MOON_MASS_MIN, LOG10_MOON_MASS_MAX, rng)[0]
            moon_a = draw_log_uniform_vec(1, LOG10_MOON_A_MIN, log_a_max, rng)[0]
            moon_ecc = draw_eccentricity_vec(1, rng, sigma=0.1)[0]
            moon_inc = INCLINATION_BASE + rng.normal(0.0, 10.0)
            moon_omega = 360.0 * rng.random()
            moon_Omega = 360.0 * rng.random()
            system[2, :] = [moon_m, moon_a, moon_ecc, moon_inc, moon_omega, moon_Omega, 3]
    
    return system


def generate_systems_batch(n_systems: int, rng: np.random.Generator,
                           expected_planets: float,
                           benchmark: bool = False) -> np.ndarray:
    """Generate multiple systems with optional benchmarking."""
    t0 = time.perf_counter()
    
    systems = []
    for i in range(n_systems):
        systems.append(generate_system(rng, expected_planets))
        
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


def worker(task: tuple[int, int], expected_planets: float) -> dict:
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
    combined = generate_systems_batch(nl, rng, expected_planets, benchmark=False)
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
    print("MULTIPLANET GENERATOR - SUZUKI SAMPLING")
    print("=" * 60)
    
    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    
    # Compute total expected planets per star
    expected_planets = compute_total_expected_planets()
    
    print(f"\nSuzuki parameters:")
    print(f"  A = {SUZUKI_A} (normalization)")
    print(f"  q_br = {SUZUKI_Q_BREAK:.2e} (break mass ratio)")
    print(f"  n = {SUZUKI_N} (slope q >= q_br)")
    print(f"  p = {SUZUKI_P} (slope q < q_br)")
    print(f"  m = {SUZUKI_M} (separation slope)")
    
    print(f"\nBounds:")
    print(f"  Mass: [{10**LOG10_MASS_MIN:.2e}, {10**LOG10_MASS_MAX:.2e}] M☉")
    print(f"  Semi-major axis: [{10**LOG10_A_MIN:.2f}, {10**LOG10_A_MAX:.2f}] AU")
    
    print(f"\n*** Expected planets per star: {expected_planets:.3f} ***")
    
    print(f"\nConfiguration:")
    print(f"  Systems per file: {nl}")
    print(f"  Files per field: {nf}")
    
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
        timings = worker(task, expected_planets)
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
