"""Quick script to draw 1M planets from Suzuki distribution and count by q range."""

import math
import numpy as np

from suzuki_draw_planet_arrays import draw_s_and_q

# Sampling parameters
N_PLANETS = 1_000_000
LOG10_Q_MIN = math.log10(1e-7)
LOG10_Q_MAX = math.log10(3.0e-2)

# Boundary for counting
Q_BOUNDARY = 2.6e-5


def main():
    print(f"Drawing {N_PLANETS:,} planets from Suzuki distribution...")
    print(f"  q range: [{10**LOG10_Q_MIN:.2e}, {10**LOG10_Q_MAX:.2e}]")
    print()

    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    q_array, s_array = draw_s_and_q(
        N_PLANETS,
        log10_q_min=LOG10_Q_MIN,
        log10_q_max=LOG10_Q_MAX,
        rng=rng
    )

    # Count planets in each range
    low_q_mask = q_array < Q_BOUNDARY
    n_low = np.sum(low_q_mask)
    n_high = np.sum(~low_q_mask)

    print(f"Results:")
    print(f"  Planets with q in [1e-7, 2.6e-5):  {n_low:,} ({100*n_low/N_PLANETS:.2f}%)")
    print(f"  Planets with q in [2.6e-5, 3.0e-2]: {n_high:,} ({100*n_high/N_PLANETS:.2f}%)")
    print()
    print(f"  Total: {n_low + n_high:,}")

    # Some extra stats
    print()
    print(f"Statistics:")
    print(f"  q_min drawn: {q_array.min():.3e}")
    print(f"  q_max drawn: {q_array.max():.3e}")
    print(f"  q_median:    {np.median(q_array):.3e}")


if __name__ == "__main__":
    main()


"""
Drawing 1,000,000 planets from Suzuki distribution...
  q range: [1.00e-07, 3.00e-02]

Results:
  Planets with q in [1e-7, 2.6e-5):  278,254 (27.83%)
  Planets with q in [2.6e-5, 3.0e-2]: 721,746 (72.17%)

  Total: 1,000,000

Statistics:
  q_min drawn: 1.000e-07
  q_max drawn: 3.000e-02
  q_median:    8.996e-05
"""
