from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray


j = complex(0, 1)


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
    num_points = (len(symbols)+1)*sps+1
    normalized_time = np.linspace(0, symbols.size+1, num=num_points, dtype=np.float64)

    # Create an array of alternating mod indicies
    ith_mod_index = np.array(range(symbols.size)) % mod_index.size
    mod_index = mod_index.take(ith_mod_index)

    # Interpolate symbols
    interpolated_soft_symbols = np.zeros(num_points, dtype=np.float64)
    interpolated_soft_symbols[sps:-1:sps] = symbols * mod_index

    # Phase modulate signal
    freq_pulses = np.convolve(interpolated_soft_symbols, pulse_filter, mode="same")
    phi = 2 * np.pi * np.cumsum(freq_pulses) / sps + np.pi / 4
    return normalized_time, np.exp(j*phi)
