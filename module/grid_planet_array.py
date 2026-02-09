"""Generate log-uniform planet property arrays (baseline sampler).

This script is the stylistic companion to ``sumi2023_draw_planet_arrays.py``
but implements a deliberately simple distribution: masses and semi-major
axes are drawn independently, each log-uniform within configured bounds.
Inclinations are isotropic and orbital phases uniform in [0, 360) degrees.

Execution (CLI)
---------------
Run directly::

        python uniform_draw_planet_arrays.py

Configuration
-------------
Set module constants below (``nl``, ``nf``, ``rundes`` etc.). To force
reproducibility across multiprocessing workers, set ``FIXED_BASE_SEED`` to
an integer. Each worker derives a unique seed from the base seed and its
``(field_number, file_index)`` tuple.

Output
------
Each generated file (text or ``.npy``) has four columns:
        1. mass (M_sun)
        2. semi-major axis (au)
        3. inclination (deg, signed; isotropic)
        4. orbital phase (deg)

Notes
-----
* Masses are sampled in Earth masses and converted to solar masses
    using ``_M_EARTH_TO_SOLAR`` prior to output.
* Existing files are removed only when ``overwrite_existing`` is True.
"""

import os
import math
import numpy as np
import multiprocessing as mp
import time
import pandas as pd

# ---------------- Simulation parameter bounds (log10 space where noted) ----------------
mmin = math.log10(0.1)      # log10 Earth masses lower bound
mmax = math.log10(10000)    # log10 Earth masses upper bound
amin = math.log10(0.3)      # log10 au lower bound
amax = math.log10(30)       # log10 au upper bound

_M_EARTH_TO_SOLAR = 3.00348959632e-6  # Earth -> Solar mass conversion

rundes = 'test_grid_draw'
sources_file = '/home/crisp.92/gullsInputFiles/runFields/pad_overguide.coords'  # defines number of fields
file_ext = ''
nl = 100      # planets per file
nf = 1          # files per field
overwrite_existing = True  # if False, existing files are kept

# Text output formatting controls
delineator = ","
header = True
_HEADER_LINE = 'mass (M_Sun), a (au), inc (deg), p (deg)'

# Fixed seeding configuration (mirrors SUMI2023 script). Set to int for reproducible run.
FIXED_BASE_SEED = None


def get_field_numbers(sources_file):
    fields = pd.read_csv(sources_file, sep=',', header=0)
    field_numbers = fields['ID_src'].to_numpy()
    return field_numbers

field_numbers = get_field_numbers(sources_file)

data_dir = './'
if data_dir[-1] != '/':
    data_dir += '/'

def worker(task):
    """Worker process: generate a single uniformly sampled planet file.

    Parameters
    ----------
    task : tuple(int, int)
        ``(field_number, file_index)`` pair specifying which output file to produce.

    Notes
    -----
    Uses a per-task :class:`numpy.random.Generator` for reproducibility when
    ``FIXED_BASE_SEED`` is set; otherwise a fresh unpredictable generator.
    """
    field_number, file_index = task
    base = f"{data_dir}/planets/{rundes}/{rundes}.planets"
    pfile = f"{base}.{field_number}.{file_index}{file_ext}"

    if os.path.exists(pfile):
        if overwrite_existing:
            try:
                os.remove(pfile)
                print(f"Removed existing {pfile} (overwrite enabled).")
            except OSError as e:
                print(f"Could not remove {pfile}: {e}")
        else:
            print(f"File {pfile} exists; skipping (overwrite disabled).")
            return

    # Deterministic per-task generator if requested
    if FIXED_BASE_SEED is not None:
        local_seed = FIXED_BASE_SEED + field_number * 100003 + file_index
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()

    # Semi–major axis: log-uniform in [10^amin, 10^amax]
    a_array = 10 ** (amin + (amax - amin) * rng.random(nl))

    # Mass: log-uniform Earth masses then convert to solar masses
    masses_earth = 10 ** (mmin + (mmax - mmin) * rng.random(nl))
    mass_array = masses_earth * _M_EARTH_TO_SOLAR

    # Isotropic inclination (signed degrees)
    rnd = rng.random(nl)
    arccos_arg = np.where(rnd < 0.5, 2 * rnd, 2 - 2 * rnd)
    safe_arg = np.clip(arccos_arg, -1.0, 1.0)
    angle = np.arccos(safe_arg)
    signed_angle = np.where(rnd < 0.5, angle, -angle)
    inc_array = 180 * signed_angle / np.pi

    # Orbital phase (deg)
    p_array = 360.0 * rng.random(nl)

    combined_array = np.empty((nl, 4))
    combined_array[:, 0] = mass_array
    combined_array[:, 1] = a_array
    combined_array[:, 2] = inc_array
    combined_array[:, 3] = p_array

    if file_ext == ".npy":
        np.save(pfile, combined_array)
    else:
        save_kwargs = {"delimiter": delineator}
        if header:
            save_kwargs["header"] = _HEADER_LINE
            save_kwargs["comments"] = "# "
        np.savetxt(pfile, combined_array, **save_kwargs)

def main():
    """Entry point: orchestrate uniform sampling across all fields.

    Creates the target directory, enumerates tasks, and maps them over a
    process pool sized to the CPU count. Prints a reproducibility notice
    if a fixed seed is configured.
    """
    if FIXED_BASE_SEED is not None:
        print(f"Deterministic run with base seed {FIXED_BASE_SEED}")
    dir_name = f"{data_dir}/planets/{rundes}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    field_numbers = get_field_numbers(sources_file)
    tasks = [(field, i) for field in field_numbers for i in range(nf)]
    print(f"Generated {len(tasks)} unique tasks. Handing them off to workers.")
    with mp.Pool(mp.cpu_count()) as pool:
        pool.map(worker, tasks)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
