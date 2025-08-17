from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt


if TYPE_CHECKING:
    from typing import Annotated, Unpack

    import numpy as np
    from annotated_types import Doc
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.markers import MarkerStyle
    from numpy.typing import NDArray


_CONSTELLATION_DEFAULT_KWARGS = {
    "alpha": 0.2,
    "marker": "s",
    "color": "b",
}


class Line2DKwargs(TypedDict):
    linewidth: float | None = None
    linestyle: str | None = None
    color: str | None = None
    gapcolor: str | None = None
    marker: str | MarkerStyle | None = None
    markersize: float | None = None
    markeredgewidth: str | None = None
    markeredgecolor: str | None = None
    markerfacecolor: str | None = None
    markerfacecoloralt: str = "none"
    fillstyle: str | None = None
    antialiased: bool | None = None
    dash_capstyle: str | None = None
    solid_capstyle: str | None = None
    dash_joinstyle: str | None = None
    solid_joinstyle: str | None = None
    pickradius: int = 5
    drawstyle: str = "default"
    markevery: Annotated[
        int | tuple[int, int] | slice | list[int] | list[bool] | float | tuple[float, float] | None,
        Doc(
            "See: https://matplotlib.org/stable/gallery/lines_bars_and_markers/markevery_demo.html"
        ),
    ] = None


def plot_constellation(
    signal: NDArray[np.complex128],
    n: int = 1024,
    axis: Axes | None = None,
    **plot_kwargs: Unpack[Line2DKwargs],
) -> Figure:
    """Plots a constellation (In-phase vs Quadrature) graph.

    Args:
        signal: complex time-domain input signal.
        n: number of samples to plot (default=1024).
        axis: Optional axis if new figure is not desired.
        plot_kwargs: Additional kw arguments for axis.plot(...).

    Returns:
        Figure of plotted constellation.
    """
    axis = axis or plt.subplots(1)[-1]

    for key, value in _CONSTELLATION_DEFAULT_KWARGS.items():
        plot_kwargs[key] = plot_kwargs.get(key, value)

    axis.plot(signal.real[:n], signal.imag[:n], **plot_kwargs)
    axis.set_title("Constellation")
    axis.set_ylabel("Quadrature [V]")
    axis.set_xlabel("In-phase [V]")
    axis.grid(which="both", linestyle=":")
    return axis.figure
