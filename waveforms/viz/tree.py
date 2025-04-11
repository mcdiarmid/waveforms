from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@matplotlib.ticker.FuncFormatter
def pi_fraction_formatter(x: float, _pos: int) -> str:
    """Pi fraction func formatter.

    Args:
        x: Coordinate number to format to string
        _pos: tick index

    Return:
        str: Formatted tick.
    """
    frac = Fraction.from_float(x / np.pi).limit_denominator(16)
    if frac.numerator == 0:
        label = "0"
    elif abs(frac.numerator) == 1 and frac.denominator == 1:
        label = rf"${'' if frac.numerator > 0 else '-'}\pi$"
    elif frac.denominator == 1:
        label = rf"${frac.numerator}\pi$"
    else:
        label = rf"$\frac{{{frac.numerator}\pi}}{{{frac.denominator}}}$"
    return label


def phase_tree(  # noqa: PLR0913
    signal: NDArray[np.complex128],
    sps: int,
    off: float | None = None,
    modulo: int = 4,
    color: str | None = None,
    axis: Axes | None = None,
) -> Figure:
    """Plots the "Phase Tree" of a CPM signal.

    Args:
        signal: Complex time-domain input signal.
        sps: Samples per symbol.
        off: Manual offset for phase tree (default zeros T=0 value).
        modulo: x-axis symbols before wrap.
        color: Optional color
        axis: Optional axis if new figure is not desired.

    Returns:
        Figure of plotted phase tree.
    """
    axis = axis or plt.subplots(1)[-1]
    phase = np.angle(signal)
    t_array = np.linspace(0, modulo, modulo * sps + 1, endpoint=True)
    for chunk in range(phase.size // (sps * modulo)):
        sym = chunk * modulo
        sliced = np.unwrap(phase[sym * sps : (sym + modulo) * sps + 1])
        offset = off if off is not None else sliced[0]
        axis.plot(
            t_array,
            (sliced - offset),
            color=color or "k",
            alpha=0.5,
        )
    axis.set_ylabel("Phase [radians]")
    axis.set_xlabel("Symbol Time [t/T]")
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(np.pi / 4))
    axis.yaxis.set_major_formatter(pi_fraction_formatter)
    axis.grid(which="both", linestyle=":")
    return axis.figure
