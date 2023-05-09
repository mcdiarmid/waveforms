import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from waveforms.cpm.soqpsk import (
    SOQPSKPrecoder,
    freq_pulse_soqpsk_a,
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

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, eye_const_axes = plt.subplots(2, 2)
    eye_real_ax: Axes = eye_const_axes[0, 0]
    eye_imag_ax: Axes = eye_const_axes[1, 0]
    constel_ax: Axes = eye_const_axes[0, 1]
    psd_ax: Axes = eye_const_axes[1, 1]

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_tg, 'royalblue', 'SOQPSK-TG'),
        (freq_pulse_soqpsk_a, 'darkorange', 'SOQPSK-A'),
        (freq_pulse_soqpsk_mil, 'crimson', 'SOQPSK-MIL'),
    )
    for pulse_filter_fn, color, label in pulses_colors_labels:
        # Modulate the input symbols
        pulse_filter = pulse_filter_fn(sps=sps)
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )

        modulo = 4
        q_symbol_offset = 0.0
        for i in range((normalized_time.size-1)//(sps*modulo)):
            idx_start_i = i*sps*modulo
            idx_start_q = idx_start_i + int(sps*q_symbol_offset)
            eye_real_ax.plot(
                (normalized_time[idx_start_i:idx_start_i+sps*modulo]+0.0)%modulo,
                modulated_signal.real[idx_start_i:idx_start_i+sps*modulo],
                linewidth=0.3,
                color=color,
            )
            eye_imag_ax.plot(
                (normalized_time[idx_start_q:idx_start_q+sps*modulo]-q_symbol_offset)%modulo,
                modulated_signal.imag[idx_start_q:idx_start_q+sps*modulo],
                linewidth=0.3,
                color=color,
            )
        constel_ax.plot(
            modulated_signal.real,
            modulated_signal.imag,
            color=color,
        )
        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
            color=color,
        )

    # Format graphs before showing them
    for ax_row in eye_const_axes:
        for ax in ax_row:
            ax.grid(which="both", linestyle=":")
    
    eye_real_ax.set_ylabel("In-phase")
    eye_imag_ax.set_ylabel("Quadrature")
    eye_imag_ax.set_xlabel("Symbol Time [t/T]")

    constel_ax.set_xlabel("In-phase")
    constel_ax.set_ylabel("Quadrature")

    psd_ax.set_ylabel('Amplitude [dBc]')
    psd_ax.set_xlabel('Frequency [bit rates]')
    psd_ax.set_ylim([-100, 20])
    psd_ax.set_yticks([(i-10)*10 for i in range(12)])
    psd_ax.set_xlim([-2.5, 2.5])
    psd_ax.set_xticks(np.linspace(-2.5, 2.5, num=11))
    fig_eye.tight_layout()
    fig_eye.show()
