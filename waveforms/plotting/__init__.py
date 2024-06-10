from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


def eye_diagram(  # noqa: PLR0913
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    sps: int = 8,
    modulo: int = 4,
    *,
    color: str | None = None,
    eye_ax: tuple[Axes, Axes] | None = None,
) -> Figure:
    """Plots an eye diagram of the provided signal.

    Args:
        time: Array of normalized time values (t/T_b)
        signal: Array of modulated signal samples
        sps: Samples per symbol
        modulo: Eye diagram t_max
        color: Color of the eye diagram
        eye_ax: Axes if a new figure is not desired

    Returns:
        Plotted figure
    """
    real_ax, imag_ax = eye_ax or plt.subplots(2)[-1]

    for i in range((time.size - 1) // (sps * modulo)):
        idx_start = i * sps * modulo
        real_ax.plot(
            (time[idx_start : idx_start + sps * modulo + 1] - time[idx_start]),
            signal.real[idx_start : idx_start + sps * modulo + 1],
            linewidth=0.3,
            color=color,
        )
        imag_ax.plot(
            (time[idx_start : idx_start + sps * modulo + 1] - time[idx_start]),
            signal.imag[idx_start : idx_start + sps * modulo + 1],
            linewidth=0.3,
            color=color,
        )

    # Format Eye diagrams
    real_ax.set_title("Eye Diagram (In-phase)")
    real_ax.set_ylabel("Amplitude")
    real_ax.set_xlabel("Normalized Time [t/T]")

    imag_ax.set_title("Eye Diagram (Quadrature)")
    imag_ax.set_ylabel("Amplitude")
    imag_ax.set_xlabel("Normalized Time [t/T]")
    return real_ax.figure
