from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray


j = complex(0, 1)


def phase_modulate(
    phase: NDArray[np.float64],
    sensitivity: float,
) -> NDArray[np.float64]:
    return np.exp(j * sensitivity * phase)


def frequency_modulate(
    freq_pulses: NDArray[np.float64],
    sps: int,
    initial_phase: float = 0,
) -> NDArray[np.float64]:
    """
    Note: since this relies on an integral, appropriate handling of
    quantization error needs to be considered for real-world implementation.

    E.g. GNURadio minimizes the effect of compounding quantization error
    by performing a modulo-2pi after each iteration in the integral/sum.
    The below implementation is similar to the GNURadio implementation,
    since there is no numpy function for "accumulate with modulo".

    :param freq_pulses: Frequency pulses (pulse shaped symbols)
    :param sps: Samples per symbol
    :param initial_phase: Phase of y(t=0)
    :return: Frequency Modulated signal
    """
    phase_array = np.zeros(freq_pulses.shape, dtype=np.float64)
    sensitivity = 2 * np.pi / sps
    revs = 0
    for i, sample in enumerate(freq_pulses):
        revs = (revs + sample) % sps
        phase_array[i] = revs * sensitivity + initial_phase
    return np.exp(j * phase_array)


def cpm_modulate(
    symbols: NDArray[np.int8],
    mod_index: Union[float, NDArray[np.float64]],
    pulse_filter: NDArray[np.float64],
    sps: int = 8,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generic CPM Modulation

    :param symbols: Array of symbols (already mapped from bits)
    :param mod_index: Modulation index, or sequence of modulation indicies
    :param pulse_filter: Pulse shaping filter (must match sps)
    :param sps: Samples per symbol [default=8]
    :return: CPM signal
    """
    # Type conversions
    if isinstance(mod_index, float):
        mod_index = [mod_index]
    if isinstance(mod_index, list):
        mod_index = np.array(mod_index, dtype=np.float64)

    # Normalized time array
    num_points = (symbols.size + 1) * sps + 1
    normalized_time = np.linspace(0, symbols.size + 1, num=num_points, dtype=np.float64)

    # Create an array of alternating mod indicies
    ith_mod_index = np.array(range(symbols.size)) % mod_index.size
    mod_index = mod_index.take(ith_mod_index)

    # Interpolate symbols
    interpolated_soft_symbols = np.zeros(num_points, dtype=np.float64)
    interpolated_soft_symbols[sps:-1:sps] = symbols * mod_index

    # Phase modulate signal
    freq_pulses = np.convolve(interpolated_soft_symbols, pulse_filter, mode="same")
    modulated_signal = frequency_modulate(freq_pulses, sps=sps, initial_phase=np.pi / 4)
    return normalized_time, modulated_signal
