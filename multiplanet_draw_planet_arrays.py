"""Generate multiplanet system arrays for GULLS General lightcurve generator.

This script produces planet files in the new format required by GULLS 3.0.0:
    Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType (x2)

Each row is ONE system with 14 columns (2 objects × 7 parameters):
    - Columns 1-7: First object (always a planet, OrbitType=1)
    - Columns 8-14: Second object (either planet OrbitType=1, or moon OrbitType=3)
A system can have a second planet OR a moon, but not both.
Objects with mass=0 indicate no object in that slot (OrbitType=1 as placeholder).

Planet properties (mass, semi-major axis) are drawn FROM the Suzuki et al. (2016)
mass ratio function:
    d²N_pl / (d log q d log s) = A × (q/q_br)^n × s^m

Mass is converted from mass ratio q via: m = q × HOST_STAR_MASS
The total expected planets per star is computed by integrating over the (q, s) bounds.
Planet count is drawn from Poisson, capped by `MAX_PLANETS` (configurable, default 2).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from typing import Optional

import numpy as np
import argparse

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
SUZUKI_P = 0.0              # slope for q < q_break (Neptunes/super-Earths), default 0.6
SUZUKI_M = 0.0              # separation exponent (roughly log-uniform), default 0.49

LOG10_Q_BREAK = math.log10(SUZUKI_Q_BREAK)

# Precomputed total expected planets per star (computed at module load)
TOTAL_EXPECTED_PLANETS: float | None = None # you can replace this with a number to renormalize the suzuki distribution but the numbers don't come out quite right, so I wouldn't trust it.
                                            # default is ~1.81 depending on your bounds
# -------------------------------------------------------------------------
# Sampling bounds
# -------------------------------------------------------------------------
LOG10_MASS_MIN = math.log10(1e-7)    # ~0.03 Earth masses
LOG10_MASS_MAX = math.log10(3e-2)    # ~10000 Earth masses
LOG10_A_MIN = math.log10(0.3)        # default 0.3 AU
LOG10_A_MAX = math.log10(30.0)       # default 30 AU

# -------------------------------------------------------------------------
# Host star mass for q ↔ m conversion
# -------------------------------------------------------------------------
# Mass ratio q = m_planet / m_star, so m_planet = q × HOST_STAR_MASS
# Setting this to 0.5 effectively doubles the mass ratio for a given planet mass
HOST_STAR_MASS = 0.5            # Host star mass in M☉ (tunable)
log10_q_min = LOG10_MASS_MIN - math.log10(HOST_STAR_MASS)
log10_q_max = LOG10_MASS_MAX - math.log10(HOST_STAR_MASS)

# -------------------------------------------------------------------------
# Orbital element parameters
# -------------------------------------------------------------------------
ECCENTRICITY_SIGMA = 1.0
ECCENTRICITY_MAX = 0.9
PERIOD_RATIO_MIN = 1.3

# -------------------------------------------------------------------------
# Moon parameters
# -------------------------------------------------------------------------
MOON_PROBABILITY = 1.0          # Probability of moon if only 1 planet (tunable)
MOON_HILL_FRACTION = 0.1        # Moon SMA upper limit as fraction of Hill radius (tunable)
LOG10_MOON_A_MIN = math.log10(0.005)  # 0.005 AU minimum SMA (tunable)
# Moon mass: order of our Moon (~3.7e-8 M☉) to Neptune (~5e-5 M☉)
LOG10_MOON_MASS_MIN = math.log10(1e-8)   # ~1/4 lunar masses
LOG10_MOON_MASS_MAX = math.log10(1e-4)   # ~2 Neptune masses

# -------------------------------------------------------------------------
# Run configuration
# -------------------------------------------------------------------------
rundes = 'test_multiplanet'
sources_file = './gulls_surot2d_H2023.sources'
file_ext = ''
nl = 1000       # systems per file (reduced for testing)
nf = 1          # files per field
overwrite_existing = True
ALLOW_ZERO_PLANETS = False  # If False, resample until each system has >=1 planet

# Maximum number of planets to allow per system (capped). Set to 1 to
# prevent any 2-planet systems and allow single-planet + moon systems.
MAX_PLANETS = 2


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments and return the namespace."""
    p = argparse.ArgumentParser(description='Generate multiplanet system arrays (Suzuki sampling)')
    p.add_argument('--max-planets', type=int, default=MAX_PLANETS,
                   help='Maximum number of planets allowed per system (default: %(default)s)')
    p.add_argument('--moon-probability', type=float, default=MOON_PROBABILITY,
                   help='Probability of adding a moon to a single-planet system (default: %(default)s)')
    p.add_argument('--allow-zero-planets', action='store_true', default=ALLOW_ZERO_PLANETS,
                   help='Allow systems with zero planets (default: %(default)s)')
    p.add_argument('--systems-per-file', '-n', type=int, default=nl,
                   help='Number of systems per output file (default: %(default)s)')
    p.add_argument('--files-per-field', type=int, default=nf,
                   help='Number of files to produce per field (default: %(default)s)')
    p.add_argument('--seed', type=int, default=None,
                   help='Base RNG seed for reproducible runs (overrides file default)')
    p.add_argument('--total-expected-planets', type=float, default=None,
                   help='Override computed TOTAL_EXPECTED_PLANETS (default: compute from Suzuki)')
    p.add_argument('--rundes', type=str, default=rundes,
                   help='Run identifier used in output filenames')
    p.add_argument('--sources-file', type=str, default=sources_file,
                   help='Path to sources file listing fields')
    p.add_argument('--no-overwrite', action='store_true', default=not overwrite_existing,
                   help='Do not overwrite existing files (default: overwrite)')
    return p.parse_args()

HEADER_LINE = 'Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType Mass SemimajorAxis Eccentricity Inclination LongitudePerihelion LongitudeAscNode OrbitType'
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
    log_q_min = LOG10_MASS_MIN - math.log10(HOST_STAR_MASS)
    log_q_max = LOG10_MASS_MAX - math.log10(HOST_STAR_MASS)
    
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
    log_q_min = LOG10_MASS_MIN - math.log10(HOST_STAR_MASS)
    log_q_max = LOG10_MASS_MAX - math.log10(HOST_STAR_MASS)
    
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

def q_to_mass(log_q: float) -> float:
    """Convert mass ratio q to planet mass using HOST_STAR_MASS.
    
    m_planet = q × HOST_STAR_MASS
    """
    q = 10.0 ** log_q
    return q * HOST_STAR_MASS


def generate_system(rng: np.random.Generator, expected_planets: float) -> np.ndarray:
    """Generate a single planetary system (1 row, 14 columns).
    
    Each row has 2 objects (7 columns each):
        - Object 1 (cols 0-6): planet (OrbitType=1) or empty
        - Object 2 (cols 7-13): planet (OrbitType=1) or moon (OrbitType=3) or empty
    
    Logic:
        - 0 planets: both slots empty (OrbitType=1 as placeholder)
        - 1 planet: first slot = planet, second slot = moon (with prob) or empty
        - 2 planets: first slot = planet1, second slot = planet2 (no moon)
        - If ALLOW_ZERO_PLANETS is False, 0-planet draws are rejected and resampled.
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    expected_planets : float
        Expected planets per star from integrated Suzuki.
    
    Returns
    -------
    np.ndarray
        Shape (14,) with [obj1_params..., obj2_params...]
    """
    # Fixed inclination placeholder (downstream simulator will handle it)
    INC_PLACEHOLDER = 1000.0
    
    # Initialize: zeros for most, but set OrbitType=1 and Inclination=1000
    system = np.zeros(14, dtype=float)
    system[3] = INC_PLACEHOLDER   # Object 1 Inclination
    system[6] = 1                 # Object 1 OrbitType
    system[10] = INC_PLACEHOLDER  # Object 2 Inclination
    system[13] = 1                # Object 2 OrbitType
    
    # Draw planet count from Poisson, cap at MAX_PLANETS (optionally reject 0-planet draws)
    while True:
        n_planets = min(rng.poisson(expected_planets), MAX_PLANETS)
        if n_planets == 0 and not ALLOW_ZERO_PLANETS:
            continue
        break
    
    if n_planets == 0:
        return system
    
    # Planet 1: sample (q, s) from Suzuki, convert q to mass
    log_q1, log_a1 = sample_suzuki(rng)
    m1 = q_to_mass(log_q1)
    a1 = 10.0 ** log_a1
    ecc1 = draw_eccentricity_vec(1, rng)[0]
    omega1 = 360.0 * rng.random()
    Omega1 = 360.0 * rng.random()
    system[0:7] = [m1, a1, ecc1, INC_PLACEHOLDER, omega1, Omega1, 1]
    
    if n_planets >= 2:
        # Draw second planet from Suzuki, ensuring period ratio constraint
        for _ in range(50):  # Max attempts
            log_q2, log_a2 = sample_suzuki(rng)
            a2 = 10.0 ** log_a2
            if check_period_ratio(a1, a2):
                m2 = q_to_mass(log_q2)
                ecc2 = draw_eccentricity_vec(1, rng)[0]
                omega2 = 360.0 * rng.random()
                Omega2 = 360.0 * rng.random()
                system[7:14] = [m2, a2, ecc2, INC_PLACEHOLDER, omega2, Omega2, 1]
                break
        # No moon possible when we have 2 planets
        return system
    
    # n_planets == 1: possibly add moon in second slot
    draw = rng.random()
    if draw < MOON_PROBABILITY:
        # Compute Hill radius for planet 1
        r_hill = compute_hill_radius(a1, ecc1, m1)
        
        # Moon SMA upper limit is MOON_HILL_FRACTION × Hill radius
        a_moon_max = MOON_HILL_FRACTION * r_hill
        a_moon_min = 10.0 ** LOG10_MOON_A_MIN
        
        # Only add moon if there's valid SMA range
        if a_moon_max > a_moon_min:
            log_a_max = math.log10(a_moon_max)
            
            # Try to draw valid moon (mass < host planet)
            for _ in range(10):  # Max attempts
                moon_m = draw_log_uniform_vec(1, LOG10_MOON_MASS_MIN, LOG10_MOON_MASS_MAX, rng)[0]
                if moon_m < m1:  # Moon must be less massive than host planet
                    moon_a = draw_log_uniform_vec(1, LOG10_MOON_A_MIN, log_a_max, rng)[0]
                    moon_ecc = draw_eccentricity_vec(1, rng, sigma=0.1)[0]
                    moon_omega = 360.0 * rng.random()
                    moon_Omega = 360.0 * rng.random()
                    system[7:14] = [moon_m, moon_a, moon_ecc, INC_PLACEHOLDER, moon_omega, moon_Omega, 3]
                    break
    
    return system


def generate_systems_batch(n_systems: int, rng: np.random.Generator,
                           expected_planets: float,
                           benchmark: bool = False) -> np.ndarray:
    """Generate multiple systems with optional benchmarking.
    
    Returns
    -------
    np.ndarray
        Shape (n_systems, 14) - each row is one system with 2 objects.
    """
    t0 = time.perf_counter()
    
    systems = []
    for i in range(n_systems):
        systems.append(generate_system(rng, expected_planets))
        
        # Progress every 10%
        if benchmark and (i + 1) % (n_systems // 10 or 1) == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{n_systems} systems ({rate:.1f}/sec)")
    
    # Stack 1D arrays into (n_systems, 14) array
    combined = np.array(systems)
    
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
        # 14 columns: 2 objects × 7 params each
        fmt_obj = ['%.6e', '%.6e', '%.6f', '%.4f', '%.4f', '%.4f', '%d']
        fmt = fmt_obj + fmt_obj  # Repeat for both objects
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
    
    # Parse CLI args and override globals where requested
    args = parse_cli_args()

    # Apply CLI overrides to module-level config
    global MAX_PLANETS, MOON_PROBABILITY, ALLOW_ZERO_PLANETS, nl, nf, FIXED_BASE_SEED, TOTAL_EXPECTED_PLANETS, rundes, sources_file, overwrite_existing
    MAX_PLANETS = args.max_planets
    MOON_PROBABILITY = args.moon_probability
    ALLOW_ZERO_PLANETS = args.allow_zero_planets
    nl = args.systems_per_file
    nf = args.files_per_field
    if args.seed is not None:
        FIXED_BASE_SEED = args.seed
    if args.total_expected_planets is not None:
        TOTAL_EXPECTED_PLANETS = args.total_expected_planets
    rundes = args.rundes
    sources_file = args.sources_file
    overwrite_existing = not args.no_overwrite

    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    
    # Compute total expected planets per star
    expected_planets = compute_total_expected_planets()
    if TOTAL_EXPECTED_PLANETS is None:
        TOTAL_EXPECTED_PLANETS = expected_planets
    
    print(f"\nSuzuki parameters:")
    print(f"  A = {SUZUKI_A} (normalization)")
    print(f"  q_br = {SUZUKI_Q_BREAK:.2e} (break mass ratio)")
    print(f"  n = {SUZUKI_N} (slope q >= q_br)")
    print(f"  p = {SUZUKI_P} (slope q < q_br)")
    print(f"  m = {SUZUKI_M} (separation slope)")
    
    print(f"\nBounds:")
    print(f"  Mass: [{10**LOG10_MASS_MIN:.2e}, {10**LOG10_MASS_MAX:.2e}] M☉")
    print(f"  (Mass ratio q: [{10**log10_q_min:.2e}, {10**log10_q_max:.2e}])")
    print(f"  Semi-major axis: [{10**LOG10_A_MIN:.2f}, {10**LOG10_A_MAX:.2f}] AU")
    
    print(f"\n*** Expected planets per star: {TOTAL_EXPECTED_PLANETS:.3f} ***")
    
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
    
    tasks = [(field, i) for field in field_ids for i in range(nf)]
    print(f"Processing {len(tasks)} tasks across all fields")
    
    # Single-threaded for clear benchmarking
    print(f"\n--- Running single-threaded for clear timing ---")
    all_timings = []
    t_total = time.perf_counter()
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)}: field={task[0]}, index={task[1]}")
        timings = worker(task, TOTAL_EXPECTED_PLANETS)
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
