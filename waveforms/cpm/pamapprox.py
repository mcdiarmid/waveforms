from typing import List
import numpy as np
from numpy.typing import NDArray


def pam_unit_pulse(
    phase_pulse: NDArray[np.float64],
    mod_index: float,
) -> NDArray[np.float64]:
    response = np.zeros(phase_pulse.size*2, dtype=np.float64)
    pih = mod_index*np.pi
    response[:phase_pulse.size] = (
        np.sin(2*pih*phase_pulse) / np.sin(pih)
    )
    response[phase_pulse.size:] = (
        np.sin(pih -2*pih*phase_pulse) / np.sin(pih)
    )
    return response


def rho_pulses(
    pulse_filter: NDArray[np.float64],
    mod_index: float,
    sps: int,
    k_max: int = 2
) -> List[NDArray[np.float64]]:
    # Construct A, B, C matricies
    L = int(pulse_filter.size/sps)
    A = np.array(
        [
            [x + c*(r>0) for x in range(L) for c in range(2)]
            for r in range(2*L+1)
        ]
    )
    B = np.linspace(0, L, num=L+1, dtype=np.uint8)
    C = np.linspace(1, L+1, num=L+1)
    u = pam_unit_pulse(np.cumsum(pulse_filter)/sps, mod_index)

    # Calculate rho_0 and rho_1
    rho: List[NDArray[np.float64]] = []
    for k in range(k_max):
        rho_k = C[k] * np.cumproduct(
            [
                np.concatenate((
                    np.zeros(A[B[k], col]*sps),
                    u,
                    np.zeros((L-A[B[k], col])*sps)
                ))
                for col in range(0, 2*L)
            ], 
            axis=0
        )[-1, A[k,-1]*sps:-L*sps]
        rho.append(rho_k)
    return rho
