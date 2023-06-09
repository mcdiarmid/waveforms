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
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(100)])
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
        noise = np.random.normal(
            loc=0,
            scale=0.05*np.sqrt(2)/2,
            size=(modulated_signal.size, 2)
        ).view(np.complex128).flatten()
        modulated_signal *= np.exp(-j*np.pi/4)
        freq_pulses = np.angle(modulated_signal[1:] * modulated_signal.conj()[:-1]) * sps / np.pi

        # Received signal
        received_signal: NDArray[np.float64] = modulated_signal + noise
        quad_demod = np.angle(received_signal[1:] * received_signal.conj()[:-1]) * sps / np.pi

        # PAM De-composition
        rho = rho_pulses(
            pulse_filter,
            mod_index,
            sps,
            k_max=2
        )
        # truncate to d_k=3
        rho = [rho_k[int((rho_k.size-3*sps)/2):int((rho_k.size+3*sps)/2)] for rho_k in rho]
        d_max = max([rho_k.size for rho_k in rho])

        # Plot stuff
        pulse_ax: Axes = iq_axes[0, i]
        phase_ax: Axes = iq_axes[1, i]
        real_ax: Axes = iq_axes[2, i]
        imag_ax: Axes = iq_axes[3, i]
        bm_real_ax: Axes = bm_axes[0, i]
        bm_imag_ax: Axes = bm_axes[1, i]

        for sym, c in zip(range(3), ("r", "g", "b")):
            for n in normalized_time[:-d_max:sps][:-1]:
                z_ln = np.zeros(d_max, dtype=np.complex128)
                ix = int((n+0)*sps)
                for k, rho_k in enumerate(rho):
                    y_kn = np.cumsum(received_signal[ix:ix+rho_k.size]*rho_k) / sps
                    z_ln[:rho_k.size] += y_kn * pseudo_symbols[k,sym].conj()

                bm_real_ax.plot(
                    normalized_time[ix:ix+d_max],
                    z_ln.real,
                    color=c,
                    linestyle="-",
                    alpha=0.3
                )
                bm_real_ax.plot(
                    normalized_time[ix+d_max],
                    z_ln.real[-1],
                    color=c,
                    linestyle="",
                    marker="o"
                )
                bm_imag_ax.plot(
                    normalized_time[ix:ix+d_max],
                    z_ln.imag,
                    color=c,
                    linestyle="-",
                    alpha=0.3
                )
                bm_imag_ax.plot(
                    normalized_time[ix+d_max],
                    z_ln.imag[-1],
                    color=c,
                    linestyle="",
                    marker="o"
                )

        phase_ax.plot(normalized_time[:-1], quad_demod)
        real_ax.plot(normalized_time, modulated_signal.real)
        imag_ax.plot(normalized_time, modulated_signal.imag)
        for k, rho_k in enumerate(rho):
            pulse_ax.plot(
                np.linspace(0, rho_k.size/sps, num=rho_k.size),
                rho_k,
                linestyle="-",
                color="b",
                label=fr"SOQPSK-{label} $\rho_{k}(t)$"
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
