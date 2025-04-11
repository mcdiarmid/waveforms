import numpy as np
from numpy.typing import NDArray


SOQPSK_NUMER = 1
SOQPSK_DENOM = 2


def freq_pulse_soqpsk(
    t1: float = 1.5,
    t2: float = 0.5,
    rho: float = 0.7,
    b: float = 1.25,
    sps: int = 8,
) -> NDArray[np.float64]:
    """Generates a SOQPSK Frequency Pulse given values for equation constants.

    See: notes/cpm/soqpsk.md

    Args:
        t1 (float): Time constant 1 ((t1 + t2) * 2 == pulse duration in symbols)
        t2 (float): Time constant 2 (window transition duration)
        rho (float): Rho constant
        b (float): Beta constant
        sps (int): Samples per symbol time

    Returns:
        NDArray[np.float64]: SOQPSK frequency pulse.
    """
    tau_max = (t1 + t2) * 2
    t_norm = np.linspace(
        -tau_max,
        tau_max,
        num=int(tau_max * sps * 2) + 1,
        dtype=np.float64,
        endpoint=True,
    )
    g = (
        np.cos(np.pi * rho * b * t_norm / 2)
        / (1 - np.power(rho * b * t_norm, 2))
        * np.sinc(b * t_norm / 2)
    )
    w = np.ones(t_norm.shape, dtype=np.float64)
    if t2 > 0:
        idx = np.where((np.abs(t_norm) >= 2 * t1) & (np.abs(t_norm) <= tau_max))
        w[idx] = (1 + np.cos(np.pi * (t_norm[idx] / 2 - t1) / t2)) / 2
        w[np.where(np.abs(t_norm) > tau_max)] = 0

    a_scalar = sps / (np.sum(g * w) * 2)
    return a_scalar * g * w


def freq_pulse_soqpsk_a(sps: int = 8) -> NDArray[np.float64]:
    """Generates a SOQPSK-A frequency pulse.

    Args:
        sps (int): Samples per symbol time

    Returns:
        NDArray[np.float64]: SOQPSK-A frequency pulse.
    """
    return freq_pulse_soqpsk(b=1.35, t1=1.4, t2=0.6, rho=1.0, sps=sps)


def freq_pulse_soqpsk_b(sps: int = 8) -> NDArray[np.float64]:
    """Generates a SOQPSK-B frequency pulse.

    Args:
        sps (int): Samples per symbol time

    Returns:
        NDArray[np.float64]: SOQPSK-B frequency pulse.
    """
    return freq_pulse_soqpsk(b=1.45, t1=2.8, t2=1.2, rho=0.5, sps=sps)


def freq_pulse_soqpsk_mil(sps: int = 8) -> NDArray[np.float64]:
    """Generates a SOQPSK-MIL frequency pulse.

    The below implementation should work, but looks wrong.
    Perhaps it should be generating length sps filter after all.

    freq_pulse_soqpsk(b=0, t1=0.25, t2=0, rho=0, sps=sps)

    On the contrary, SOQPSK-MIL looks wrong in the PAM
    de-composition graphs when using the sps-length filter of
    0.5 points.  If the below is indeed the correct filter
    then clearly something is incorrect in the PAM decomposition
    implementation...

    np.ones(sps, dtype=np.float64) / 2

    For now we will revert to monke as I intend to get this PAM
    de-composition implemented correctly.

    Args:
        sps (int): Samples per symbol time

    Returns:
        NDArray[np.float64]: SOQPSK-MIL frequency pulse.
    """
    g = np.ones(sps + 1, dtype=np.float64) / 2
    g[0] = 0
    return g


def freq_pulse_soqpsk_tg(sps: int = 8) -> NDArray[np.float64]:
    """Generates a SOQPSK-TG frequency pulse.

    Args:
        sps (int): Samples per symbol time

    Returns:
        NDArray[np.float64]: SOQPSK-TG frequency pulse.
    """
    return freq_pulse_soqpsk(sps=sps)
