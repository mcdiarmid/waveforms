import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from waveforms.cpm.soqpsk import (
    SOQPSKPrecoder,
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
    fig_eye, eye_const_axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

    pulse_ax: Axes = eye_const_axes[0, 1]
    iq_ax: Axes = eye_const_axes[1, 1]

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_tg(sps=sps), 'royalblue', 'TG'),
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

    # Format pulse diagram
    pulse_ax.set_title("Phase and Frequency Pulses")
    pulse_ax.set_ylabel("Amplitude")
    pulse_ax.set_xlabel("Normalized Time [t/T]")
    pulse_ax.set_ylim(-0.1, 0.7)
    pulse_ax.set_xlim([-8, 8])
    pulse_ax.xaxis.set_major_locator(MultipleLocator(2))
    pulse_ax.legend(loc="upper center", fontsize=8, ncol=4)
    pulse_ax.grid(which="both", linestyle=":")

    fig_eye.tight_layout()
    # fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    fig_eye.show()
