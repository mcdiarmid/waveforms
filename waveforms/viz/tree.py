from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from waveforms.cpm.modulate import cpm_modulate


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from waveforms.cpm.trellis.encoder import TrellisEncoder


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


def plot_phase_tree(  # noqa: PLR0913
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
    t_array = np.linspace(0, modulo, modulo * sps, endpoint=False)
    for chunk in range(phase.size // (sps * modulo)):
        sym = chunk * modulo
        sliced = np.unwrap(phase[sym * sps : (sym + modulo) * sps])
        offset = off if off is not None else sliced[0]
        axis.plot(
            t_array,
            (sliced - offset),
            color=color or "k",
            alpha=0.3,
            linewidth=0.5,
        )
    axis.set_ylabel("Phase [radians]")
    axis.set_xlabel("Symbol Time [t/T]")
    axis.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(np.pi / 4))
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(np.pi))
    axis.yaxis.set_major_formatter(pi_fraction_formatter)
    axis.grid(which="both", linestyle=":")
    axis.set_title("Phase Tree")
    return axis.figure


def generate_cpm_phase_tree(
    pulse_filter: NDArray[np.float64],
    mod_index: float | NDArray[np.float64],
    encoder: TrellisEncoder,
    sps: int,
    axis: Axes | None = None,
) -> Figure:
    """Generates a phase tree given modulation & sequence gen parameters.

    Args:
        pulse_filter: Pulse filter array.
        mod_index: Modulation index (constant for single-h or array for multi-h).
        encoder: TrellisEncoder object.
        sps: Samples per symbol.
        axis: Axis to plot (optional, will create one if none provided).

    Returns:
        Figure: Plotted figure.

    """
    # Generate all input bit sequences, zero pad at front
    bps = encoder.input_cardinality
    length = int(pulse_filter.size // sps)
    packed_len = bps * length

    symbol_sequences = []
    for i in range(2 ** (bps * length)):
        encoder.state = 0
        seq = [(i >> j) & 1 for j in range(packed_len)]
        padded_seq = np.array(
            [0] * packed_len + seq,
            dtype=np.uint8,
        )
        symbols = tuple(encoder.encode(padded_seq))
        symbol_sequences.append(symbols)

    # Remove duplicate symbol sequences from resulting list
    symbol_sequences = sorted(set(symbol_sequences))

    # Modulate & plot all symbol sequences
    plotted_signal = np.zeros(len(symbol_sequences) * length * sps + 1, dtype=np.complex128)
    for i, symbols in enumerate(symbol_sequences):
        _time, tmp = cpm_modulate(
            np.array(symbols, dtype=np.int8),
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        plotted_signal[i * length * sps : (i + 1) * length * sps] = tmp[
            length * sps - 1 : 2 * length * sps - 1
        ]

    return plot_phase_tree(
        signal=plotted_signal,
        sps=sps,
        modulo=length,
        axis=axis,
    )
