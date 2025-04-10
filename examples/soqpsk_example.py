from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from waveforms.cpm.modulate import cpm_modulate
from waveforms.cpm.soqpsk import (
    SOQPSKPrecoder,
    freq_pulse_soqpsk_a,
    freq_pulse_soqpsk_b,
    freq_pulse_soqpsk_mil,
    freq_pulse_soqpsk_tg,
)
from waveforms.glfsr import PNSequence
from waveforms.viz import eye_diagram

rng = np.random.Generator(np.random.PCG64())

PN_DEGREE = 14
DATA_GEN = PNSequence(PN_DEGREE)
DATA_BUFFER = np.packbits(DATA_GEN.generate_sequence())
j = complex(0, 1)


if __name__ == "__main__":
    # Constants
    sps = 10
    fft_size = 2**10
    mod_index = 1 / 2
    pulse_pad = 0.5

    # Bits of information to transmit
    bit_array = np.unpackbits(DATA_BUFFER)

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, eye_const_axes = plt.subplots(2, 2, figsize=(12, 10), dpi=80)
    eye_real_ax: Axes = eye_const_axes[0, 0]
    eye_imag_ax: Axes = eye_const_axes[1, 0]
    pulse_ax: Axes = eye_const_axes[0, 1]
    psd_ax: Axes = eye_const_axes[1, 1]

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_b(sps=sps), "lightgrey", "B"),
        (freq_pulse_soqpsk_tg(sps=sps), "royalblue", "TG"),
        (freq_pulse_soqpsk_a(sps=sps), "darkorange", "A"),
        (freq_pulse_soqpsk_mil(sps=sps), "crimson", "MIL"),
    )
    signal_dict = {}
    for pulse_filter, color, label in pulses_colors_labels:
        # Modulate the input symbols
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        modulated_signal[:] = modulated_signal * np.exp(-1 * j * np.pi / 4)
        signal_dict[label] = modulated_signal[:]
        normalized_time /= 2  # SOQPSK symbols are spaced at T/2
        eye_diagram(
            normalized_time[: normalized_time.size // 4],
            modulated_signal[: normalized_time.size // 4],
            sps=sps,
            modulo=4,
            axes=(eye_real_ax, eye_imag_ax),
            color=color,
        )
        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
            color=color,
            label=label,
        )

        # Plot frequency and pulses
        pulse_length = pulse_filter.size / sps
        if label == "MIL":
            padded_pulse: NDArray[np.float64] = np.concatenate(
                (
                    np.zeros(int(pulse_pad * sps)),
                    pulse_filter,
                    np.zeros(int(pulse_pad * sps)),
                ),
            )
            t_lim = (
                -(pulse_length / 2 + pulse_pad),
                +(pulse_length / 2 + pulse_pad),
            )
        else:
            padded_pulse = pulse_filter
            t_lim = (
                -(pulse_length / 2),
                +(pulse_length / 2),
            )
        pulse_t = np.linspace(
            *t_lim,
            num=padded_pulse.size,
        )
        pulse_ax.plot(
            pulse_t,
            padded_pulse,
            linestyle="-",
            linewidth=2,
            color=color,
            label=rf"{label} $f(t)$",
        )
        pulse_ax.plot(
            pulse_t,
            np.cumsum(padded_pulse) / sps,
            linestyle="-.",
            linewidth=1,
            color=color,
            label=rf"{label} $q(t)$",
        )

    # Format graphs before showing them
    for ax_row in eye_const_axes:
        for ax in ax_row:
            ax.grid(which="both", linestyle=":")

    # Format pulse diagram
    pulse_ax.set_title("Phase and Frequency Pulses")
    pulse_ax.set_ylabel("Amplitude")
    pulse_ax.set_xlabel("Normalized Time [t/T]")
    pulse_ax.set_ylim(-0.1, 0.7)
    pulse_ax.set_xlim([-8, 8])
    pulse_ax.xaxis.set_major_locator(MultipleLocator(2))
    pulse_ax.legend(loc="upper center", fontsize=8, ncol=4)

    # Format the PSD plot
    psd_ax.set_title("Power Spectral Density")
    psd_ax.set_ylabel("Amplitude [dBc]")
    psd_ax.set_xlabel("Frequency [bit rates]")
    psd_ax.set_ylim([-100, 20])
    psd_ax.yaxis.set_major_locator(MultipleLocator(10))
    psd_ax.set_xlim([-2.5, 2.5])
    psd_ax.legend(loc="upper center", fontsize=8, ncol=4)
    psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_waveforms1.png")
