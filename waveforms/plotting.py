from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def eye_diagram(
    time: NDArray[np.float64],
    signal: NDArray[np.float64],
    sps: int = 8,
    modulo: int = 4,
    *,
    color: Optional[str] = None,
    eye_ax: Optional[Tuple[Axes, Axes]] = None,
) -> Figure:
    """
    Plots an eye diagram of the provided signal.

    :param time: Array of normalized time values (t/T_b)
    :param signal: Array of modulated signal samples
    :param sps: Samples per symbol
    :param modulo: Eye diagram t_max
    :param color: Color of the eye diagram
    :param figure: Figure for overlaying multiple eye diagrams
    :return: Plotted figure
    """
    real_ax, imag_ax = eye_ax or plt.subplots(2)[-1]

    for i in range((time.size-1)//(sps*modulo)):
        idx_start = i*sps*modulo
        real_ax.plot(
            (time[idx_start:idx_start+sps*modulo+1]-time[idx_start]),
            signal.real[idx_start:idx_start+sps*modulo+1],
            linewidth=0.3,
            color=color,
        )
        imag_ax.plot(
            (time[idx_start:idx_start+sps*modulo+1]-time[idx_start]),
            signal.imag[idx_start:idx_start+sps*modulo+1],
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


def psd() -> Figure:
    pass
