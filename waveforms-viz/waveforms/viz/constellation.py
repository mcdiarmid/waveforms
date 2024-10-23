from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def constellation(
    signal: NDArray[np.complex64],
    *,
    n: int = 1024,
    color: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plots a constellation (In-phase vs Quadrature) graph.

    Args:
        signal: complex time-domain input signal.
        n: number of samples to plot (default=1024).
        color: Color of the lines and points.
        ax: Optional axis if new figure is not desired.

    Returns:
        Figure of plotted constellation.
    """
    fig, ax = ax or plt.subplots(1)
    ax.plot(signal.real[:n], signal.imag[:n], color=color, alpha=0.7)
    ax.set_title("Constellation")
    ax.set_ylabel("Quadrature [V]")
    ax.set_xlabel("In-phase [V]")
    ax.grid(which="both", linestyle=":")
    return fig
