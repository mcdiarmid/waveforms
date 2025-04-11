from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt


if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def constellation(
    signal: NDArray[np.complex128],
    n: int = 1024,
    color: str | None = None,
    axis: Axes | None = None,
) -> Figure:
    """Plots a constellation (In-phase vs Quadrature) graph.

    Args:
        signal: complex time-domain input signal.
        n: number of samples to plot (default=1024).
        color: Color of the lines and points.
        axis: Optional axis if new figure is not desired.

    Returns:
        Figure of plotted constellation.
    """
    axis = axis or plt.subplots(1)[-1]
    axis.plot(signal.real[:n], signal.imag[:n], color=color, alpha=0.7)
    axis.set_title("Constellation")
    axis.set_ylabel("Quadrature [V]")
    axis.set_xlabel("In-phase [V]")
    axis.grid(which="both", linestyle=":")
    return axis.figure
