import random
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator

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
    sps = 20
    fft_size = 2**9
    # mod_index = 1/2
    pulse_pad = 0.5

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, iq_axes = plt.subplots(4, 2, figsize=(12, 10), dpi=100)
    for ax in iq_axes.flatten():
        ax: Axes
        ax.set_xlim([10, 20])
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="both", linestyle=":")

    # Generate pseudo-symbols
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
            scale=0.1*np.sqrt(2)/2,
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
        # fuck why am I doing this again?
        if label == "TG":
            rho: List[NDArray[np.float64]] = [
                rho_k[int((rho_k.size-nt*sps)/2):int((rho_k.size+nt*sps)/2)+1]
                for rho_k, nt in zip(rho, (3, 4))
            ]
        d_max = max([rho_k.size for rho_k in rho])

        # Assign axes
        iq_ax: Axes = iq_axes[0, i]
        pulse_ax: Axes = iq_axes[1, i]
        mf_ax: Axes = iq_axes[2, i]
        rho_ax: Axes = iq_axes[3, i]

        # TODO Implement traceback, viterbi algorithm, and trellis state
        for sym, c in zip(range(3), ("r", "g", "b")):
            z_ln = np.zeros(received_signal.size+d_max-1, dtype=np.complex128)
            t = np.linspace(0, z_ln.size/sps, z_ln.size)
            for k, rho_k in enumerate(rho):
                z_ln[:z_ln.size-(d_max-rho_k.size)] += np.convolve(received_signal, rho_k) * pseudo_symbols[k,sym].conj()

            for n in normalized_time[:-d_max:sps][:-1]:
                ix = int((n+0)*sps)
                mf_ax.plot(
                    normalized_time[ix+d_max],
                    z_ln.real[ix+d_max],
                    color=c,
                    linestyle="",
                    marker="o"
                )
                mf_ax.plot(
                    normalized_time[ix+d_max],
                    z_ln.imag[ix+d_max],
                    color=c,
                    linestyle="",
                    marker="^"
                )
            mf_ax.plot(
                t,
                z_ln.real,
                color=c,
                linestyle="-",
                alpha=0.5
            )
            
            mf_ax.plot(
                t,
                z_ln.imag,
                color=c,
                linestyle="--",
                alpha=0.5
            )

        pulse_ax.plot(normalized_time[:-1], quad_demod)
        pulse_ax.stem(normalized_time[sps:-1:sps], symbols)
        iq_ax.plot(normalized_time, modulated_signal.real)
        iq_ax.plot(normalized_time, modulated_signal.imag)
        for k, rho_k, fmt in zip((0, 1), rho, ("b-", "g--")):
            rho_ax.plot(
                np.linspace(0, (rho_k.size-1)/sps, num=rho_k.size),
                rho_k,
                fmt,
                label=fr"SOQPSK-{label} $\rho_{k}(t)$"
            )

        rho_ax.set_xlim(0, (max(rho, key=np.size).size-1)/sps)

    for rho_ax in iq_axes[3, :]:
        rho_ax.grid(which="both", linestyle=":")
        rho_ax.legend()
        rho_ax.xaxis.set_major_locator(AutoLocator())
        rho_ax.xaxis.set_minor_locator(AutoMinorLocator())

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    fig_eye.show()
