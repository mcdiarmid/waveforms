from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def plot_power_spectral_density(
    signal: NDArray[np.complex128],
    sps: int,
    bps: int = 1,
    nfft: int = 1024,
    axis: Axes | None = None,
) -> Figure:
    """Plots the power spectral density of a signal.

    Args:
        signal: Complex baseband signal
        sps: Samples per Symbol
        bps: Bits per symbol
        nfft: FFT Size
        axis: Axes if a new figure is not desired

    Returns:
        Figure: Plotted figure
    """
    axis = axis or plt.subplots(1)[-1]
    axis.psd(
        signal * np.sqrt(bps),
        NFFT=nfft,
        Fs=sps / bps,
        scale_by_freq=False,
    )

    # Format the PSD plot
    axis.set_ylabel("Amplitude [dBc]")
    axis.set_ylim([-80, 0])
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))

    axis.set_xlim([-2, 2])
    axis.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    axis.set_xlabel("Normalized Frequency [$T_b$ = 1]")

    axis.set_title("Power Spectral Density")
    axis.grid(which="both", linestyle=":")
    return axis.figure
