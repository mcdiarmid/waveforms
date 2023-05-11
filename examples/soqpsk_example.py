import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from waveforms.cpm.soqpsk import (
    SOQPSKPrecoder,
    freq_pulse_soqpsk_a,
    freq_pulse_soqpsk_b,
    freq_pulse_soqpsk_tg,
    freq_pulse_soqpsk_mil,
)
from waveforms.cpm.modulate import cpm_modulate


DATA_HEADER = b"\x1b\x1bHello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(250)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


if __name__ == "__main__":
    # Constants
    sps = 8
    fft_size = 2**9
    mod_index = 1/2
    pulse_pad = 0.5

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, eye_const_axes = plt.subplots(2, 2)
    eye_real_ax: Axes = eye_const_axes[0, 0]
    eye_imag_ax: Axes = eye_const_axes[1, 0]
    pulse_ax: Axes = eye_const_axes[0, 1]
    psd_ax: Axes = eye_const_axes[1, 1]

    fig_constel, constel_ax = plt.subplots(1, 1)

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_b(sps=sps), 'lightgrey', 'B'),
        (freq_pulse_soqpsk_tg(sps=sps), 'royalblue', 'TG'),
        (freq_pulse_soqpsk_a(sps=sps), 'darkorange', 'A'),
        (freq_pulse_soqpsk_mil(sps=sps), 'crimson', 'MIL'),
    )
    for pulse_filter, color, label in pulses_colors_labels:
        # Modulate the input symbols
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        normalized_time /= 2  # SOQPSK symbols are spaced at T/2

        modulo = 4
        for i in range((normalized_time.size-1)//(sps*modulo)):
            idx_start = i*sps*modulo
            eye_real_ax.plot(
                (normalized_time[idx_start:idx_start+sps*modulo+1]-normalized_time[idx_start]),
                modulated_signal.real[idx_start:idx_start+sps*modulo+1],
                linewidth=0.3,
                color=color,
            )
            eye_imag_ax.plot(
                (normalized_time[idx_start:idx_start+sps*modulo+1]-normalized_time[idx_start]),
                modulated_signal.imag[idx_start:idx_start+sps*modulo+1],
                linewidth=0.3,
                color=color,
            )
        constel_ax.plot(
            modulated_signal.real,
            modulated_signal.imag,
            color=color,
            label=label,
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
            padded_pulse: NDArray[np.float64] = np.concatenate((
                np.zeros(int(pulse_pad*sps)),
                pulse_filter,
                np.zeros(int(pulse_pad*sps)),
            ))
            t_lim = (
                -(pulse_length/2 + pulse_pad),
                +(pulse_length/2 + pulse_pad)
            )
        else:
            padded_pulse = pulse_filter
            t_lim = (
                -(pulse_length/2),
                +(pulse_length/2)
            )
        pulse_t = np.linspace(
            *t_lim,
            num=padded_pulse.size
        )
        pulse_ax.plot(
            pulse_t,
            padded_pulse,
            linestyle='-',
            linewidth=2,
            color=color,
            label=rf'{label} $f(t)$',
        )
        pulse_ax.plot(
            pulse_t,
            np.cumsum(padded_pulse) / sps,
            linestyle='-.',
            linewidth=1,
            color=color,
            label=rf'{label} $q(t)$',
        )

    # Format graphs before showing them
    for ax_row in eye_const_axes:
        for ax in ax_row:
            ax.grid(which="both", linestyle=":")

    # Format Eye diagrams
    eye_real_ax.set_title("Eye Diagram (In-phase)")
    eye_real_ax.set_ylabel("Amplitude")
    eye_real_ax.set_xlabel("Normalized Time [t/T]")

    eye_imag_ax.set_title("Eye Diagram (Quadrature)")
    eye_imag_ax.set_ylabel("Amplitude")
    eye_imag_ax.set_xlabel("Normalized Time [t/T]")

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
    psd_ax.set_ylabel('Amplitude [dBc]')
    psd_ax.set_xlabel('Frequency [bit rates]')
    psd_ax.set_ylim([-100, 20])
    psd_ax.yaxis.set_major_locator(MultipleLocator(10))
    psd_ax.set_xlim([-2.5, 2.5])
    psd_ax.legend(loc="upper center", fontsize=8, ncol=4)
    psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))

    fig_eye.tight_layout()
    fig_eye.show()
