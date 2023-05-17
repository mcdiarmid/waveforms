import random
from pathlib import Path
from typing import List

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
from waveforms.cpm.pamapprox import rho_pulses


DATA_HEADER = b"\x1b\x1bHello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(250)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


if __name__ == "__main__":
    # Constants
    sps = 40
    fft_size = 2**9
    # mod_index = 1/2
    pulse_pad = 0.5

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, iq_axes = plt.subplots(5, 2, figsize=(12, 10), dpi=100)
    fig_eye, bm_axes = plt.subplots(3, 2, figsize=(12, 4), dpi=100)
    pseudo_symbols = np.array([  # [k, l]
        [-j, 1, j],
        [np.sqrt(2)/2*(1-j), np.sqrt(2)/2, np.sqrt(2)/2*(1+j)],
    ], dtype=np.complex128)

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_mil(sps=sps), 'crimson', 'MIL', 1/4),
        (freq_pulse_soqpsk_tg(sps=sps), 'royalblue', 'TG', 1/4),
    )
    for i, (pulse_filter, color, label, mod_index) in enumerate(pulses_colors_labels):
        # Modulate the input symbols
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols*2,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        # modulated_signal *= np.exp(-j*np.pi/4)
        quad_demod = np.angle(modulated_signal[1:] * modulated_signal.conj()[:-1]) * sps / np.pi
        normalized_time /= 2  # SOQPSK symbols are spaced at T/2

        # PAM De-composition
        rho_0, rho_1 = rho_pulses(
            pulse_filter,
            mod_index,
            sps,
            k_max=2
        )
        y = [
            np.convolve(rho, modulated_signal, mode="same")
            for rho in (rho_0, rho_1)
        ]

        # Plot stuff
        pulse_ax: Axes = iq_axes[0, i]
        phase_ax: Axes = iq_axes[1, i]
        real_ax: Axes = iq_axes[2, i]
        imag_ax: Axes = iq_axes[3, i]
        bm_real_ax: Axes = bm_axes[0, i]
        bm_imag_ax: Axes = bm_axes[1, i]
        bm_ph_ax: Axes = bm_axes[2, i]

        for l in range(3):
            z_pam_l = 0
            for k, yk in enumerate(y):
                z_pam_l += yk * pseudo_symbols[k,l].conj()
            t = np.linspace(0, yk.size/sps, num=yk.size)
            zz = z_pam_l*modulated_signal
            bm_real_ax.plot(
                t,
                zz.real*zz.imag,
                label=fr"SOQPSK-{label} l={l}"
            )
            bm_imag_ax.plot(
                t,
                zz.imag,
                label=fr"SOQPSK-{label} l={l}"
            )
            bm_ph_ax.plot(
                t[:-1],
                quad_demod
            )

        phase_ax.plot(normalized_time[:-1], quad_demod)
        real_ax.plot(normalized_time, modulated_signal.real)
        imag_ax.plot(normalized_time, modulated_signal.imag)
        pulse_ax.plot(
            np.linspace(0, rho_0.size/sps, num=rho_0.size),
            rho_0,
            linestyle="-",
            color="b",
            label=fr"SOQPSK-{label} $\rho_0(t)$"
        )
        pulse_ax.plot(
            np.linspace(0, rho_1.size/sps, num=rho_1.size),
            rho_1,
            linestyle="--",
            color="g",
            label=fr"SOQPSK-{label} $\rho_1(t)$"
        )

    phase_ax.stem(normalized_time[sps:-1:sps], symbols)
    # Format pulse diagram
    for ax_row in (*iq_axes[1:], *bm_axes):
        for ax in ax_row:
            ax: Axes
            ax.set_xlim([10, 20])
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which="both", linestyle=":")

    for pulse_ax in iq_axes[0, :]:
        pulse_ax.grid(which="both", linestyle=":")
        pulse_ax.legend()

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    fig_eye.show()
