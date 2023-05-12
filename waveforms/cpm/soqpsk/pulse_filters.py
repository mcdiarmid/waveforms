import numpy as np
from numpy.typing import NDArray


SOQPSK_NUMER = 1
SOQPSK_DENOM = 2


def freq_pulse_soqpsk(
    T_1: float = 1.5,
    T_2: float = 0.5,
    rho: float = 0.7,
    b: float = 1.25,
    sps: int = 8,
) -> NDArray[np.float64]:
    tau_max = (T_1 + T_2) * 2
    t_norm = np.linspace(
        -tau_max,
        tau_max,
        num=int(tau_max * sps * 2),
        dtype=np.float64
    )
    g = (
        np.cos(np.pi * rho * b * t_norm / 2) / 
        (1 - np.power(rho * b * t_norm, 2)) * 
        np.sinc(b * t_norm / 2)
    )
    w = np.ones(t_norm.shape, dtype=np.float64)
    if T_2 > 0:
        idx = np.where(
            (np.abs(t_norm) >= 2 * T_1) & 
            (np.abs(t_norm) <= tau_max)
        )
        w[idx] = (1 + np.cos(np.pi * (t_norm[idx] / 2 - T_1) / T_2)) / 2
        w[np.where(np.abs(t_norm) > tau_max)] = 0

    a_scalar = sps / (np.cumsum(g*w)[-1] * 2)
    return a_scalar * g * w


def freq_pulse_soqpsk_a(sps: int = 8) -> NDArray[np.float64]:
    return freq_pulse_soqpsk(b=1.35, T_1=1.4, T_2=0.6, rho=1.0, sps=sps)


def freq_pulse_soqpsk_b(sps: int = 8) -> NDArray[np.float64]:
    return freq_pulse_soqpsk(b=1.45, T_1=2.8, T_2=1.2, rho=0.5, sps=sps)


def freq_pulse_soqpsk_mil(sps: int = 8) -> NDArray[np.float64]:
    return freq_pulse_soqpsk(b=0, T_1=0.25, T_2=0, rho=0, sps=sps)


def freq_pulse_soqpsk_tg(sps: int = 8) -> NDArray[np.float64]:
    return freq_pulse_soqpsk(sps=sps)
