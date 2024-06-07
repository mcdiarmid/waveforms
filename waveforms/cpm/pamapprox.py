from typing import List
from functools import reduce

import numpy as np
from numpy.typing import NDArray


def pam_unit_pulse(
    phase_pulse: NDArray[np.float64],
    mod_index: float,
) -> NDArray[np.float64]:
    response = np.zeros(phase_pulse.size * 2 - 1, dtype=np.float64)
    pih = mod_index * np.pi
    response[1 : phase_pulse.size + 1] = np.sin(2 * pih * phase_pulse) / np.sin(pih)
    response[phase_pulse.size - 1 :] = np.sin(pih - 2 * pih * phase_pulse) / np.sin(pih)
    return response


def rho_pulses(
    pulse_filter: NDArray[np.float64], mod_index: float, sps: int, k_max: int = 2
) -> List[NDArray[np.float64]]:
    """
    Generate PAM decomposition rho pulses for CPM.

    From E. Perrins Dissertation Ch. 3
    https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1619&context=etd

    :param pulse_filter: Pulse filter sampled at SPS
    :param mod_index: Modulation index
    :param sps: Samples per Symbol
    :param k_max: Number of rho pulses to generate (first 2 by default)
    """
    # Construct A, B, C matricies
    L = int(pulse_filter.size / sps)
    A = np.array([[x + c * (r > 0) for x in range(L) for c in range(2)] for r in range(2 * L + 1)])
    B = np.linspace(0, L, num=L + 1, dtype=np.uint8)
    C = np.linspace(1, L + 1, num=L + 1)
    u = pam_unit_pulse(np.cumsum(pulse_filter) / sps, mod_index)

    # Calculate rho_0 and rho_1 - equation 3.21
    rho: List[NDArray[np.float64]] = []
    for k in range(k_max):
        time_shifted_u = np.array(
            [
                np.concatenate(
                    (np.zeros(A[B[k], col] * sps), u, np.zeros((L - A[B[k], col]) * sps))
                )
                for col in range(0, 2 * L)
            ],
            dtype=np.float64,
        )
        rho_k = C[k] * reduce(np.multiply, time_shifted_u)[A[k, -1] * sps : -L * sps]
        rho.append(rho_k)
    return rho
