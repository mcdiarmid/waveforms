from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pam_unit_pulse(
    phase_pulse: NDArray[np.float64],
    mod_index: float,
) -> NDArray[np.float64]:
    """Generates a PAM unit pulse for a given phase pulse and modulation index.

    Args:
        phase_pulse (NDArray[np.float64]): Phase impulse response filter
        mod_index (float): Modulation index

    Returns:
        NDArray[np.float64]: PAM unit pulse
    """
    response = np.zeros(phase_pulse.size * 2 - 1, dtype=np.float64)
    pih = mod_index * np.pi
    response[1 : phase_pulse.size + 1] = np.sin(2 * pih * phase_pulse) / np.sin(pih)
    response[phase_pulse.size - 1 :] = np.sin(pih - 2 * pih * phase_pulse) / np.sin(pih)
    return response


def rho_pulses(
    pulse_filter: NDArray[np.float64],
    mod_index: float,
    sps: int,
    k_max: int = 2,
) -> list[NDArray[np.float64]]:
    """Generate PAM decomposition rho pulses for CPM.

    From E. Perrins Dissertation Ch. 3
    https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1619&context=etd

    Args:
        pulse_filter: Pulse filter sampled at SPS
        mod_index: Modulation index
        sps: Samples per Symbol
        k_max: Number of rho pulses to generate (first 2 by default)

    Returns:
        list[NDArray[np.float64]]: Rho pulses
    """
    # Construct a_matrix, b_matrix, c_array matricies
    length = int(pulse_filter.size / sps)
    a_matrix = np.array(
        [
            [x + c * (r > 0) for x in range(length) for c in range(2)]
            for r in range(2 * length + 1)
        ],
        dtype=np.uint8,
    )
    b_matrix = np.linspace(0, length, num=length + 1, dtype=np.uint8)
    c_array = np.linspace(1, length + 1, num=length + 1)
    u = pam_unit_pulse(np.cumsum(pulse_filter) / sps, mod_index)

    # Calculate rho_0 and rho_1 - equation 3.21
    rho: list[NDArray[np.float64]] = []
    for k in range(k_max):
        time_shifted_u = np.array(
            [
                np.concatenate((
                    np.zeros(a_matrix[b_matrix[k], col] * sps),
                    u,
                    np.zeros((length - a_matrix[b_matrix[k], col]) * sps),
                ))
                for col in range(2 * length)
            ],
            dtype=np.float64,
        )
        rho_k: NDArray[np.float64] = (
            c_array[k] * reduce(np.multiply, time_shifted_u)
        )[a_matrix[k, -1] * sps : -length * sps]
        rho.append(rho_k)
    return rho
