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
    freq_pulse_soqpsk_mil,
    freq_pulse_soqpsk_tg,
)
from waveforms.cpm.helpers import normalize_cpm_filter
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


    fig, pt_axes = plt.subplots(4, 2, figsize=(12, 10), dpi=100)

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

        idx_start = int((pulse_filter.size - sps)/2)
        idx_end = int((pulse_filter.size + sps)/2)
        # truncated_pulse = pulse_filter[idx_start:idx_end]
        truncated_pulse = normalize_cpm_filter(sps=sps, g=pulse_filter[idx_start:idx_end])
        matched_filters = [
            np.exp(-j*2*np.pi*mod_index*a_k*truncated_pulse)
            for a_k in (0, 1)
        ]

        iq_ax: Axes = pt_axes[0, i]
        pulse_ax: Axes = pt_axes[1, i]
        mf_ax: Axes = pt_axes[2, i]
        psd_ax: Axes = pt_axes[3, i]
        mf_rx = np.convolve(received_signal, matched_filters[-2], mode="same") / sps

        iq_ax.plot(normalized_time, modulated_signal.real, color="blue", alpha=1)
        iq_ax.plot(normalized_time, modulated_signal.imag, color="red", alpha=1)
        iq_ax.plot(normalized_time, received_signal.real, color="blue", alpha=0.5)
        iq_ax.plot(normalized_time, received_signal.imag, color="red", alpha=0.5)
        iq_ax.plot(normalized_time, mf_rx.real, color="darkblue", alpha=0.8)
        iq_ax.plot(normalized_time, mf_rx.imag, color="darkred", alpha=0.8)

        pulse_ax.plot(normalized_time[:-1], freq_pulses)
        pulse_ax.plot(normalized_time[:-1], np.convolve(quad_demod, pulse_filter, mode="same")*2/sps)
        pulse_ax.stem(normalized_time[sps:-1:sps], symbols)
        for symbol, color in zip((0, 1), ("g", "b")):
            mf = np.exp(-j*2*np.pi*mod_index*symbol*truncated_pulse)
            for k in normalized_time[:-mf.size:sps][:-1]:
                ix = int((k+0.5)*sps)
                fx = ix + mf.size
                tau = normalized_time[ix:fx]
                z_kn = np.cumsum(received_signal[ix:fx]*mf) / sps
                mf_ax.plot(tau, z_kn.real, color=color, linestyle="-", alpha=0.4)
                mf_ax.plot(normalized_time[(fx+ix)//2], z_kn.real[-1], color=color, marker=">")
                mf_ax.plot(tau, z_kn.imag, color=color, linestyle=":", alpha=0.4)
                mf_ax.plot(normalized_time[(fx+ix)//2], z_kn.imag[-1], color=color, marker="<")

            z_k = np.convolve(received_signal, mf, mode="same") / sps
            mf_ax.plot(
                normalized_time,
                z_k.real,
                color=color,
                linestyle="-",
                alpha=0.7,
            )
            mf_ax.plot(
                normalized_time,
                z_k.imag,
                color=color,
                linestyle=":",
                alpha=0.7,
            )

        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
            label=label,
        )
        psd_ax.psd(
            received_signal,
            NFFT=fft_size,
            Fs=sps,
            label=label,
        )

    for ax_row in pt_axes[:-1]:
        for ax in ax_row:
            ax: Axes
            ax.set_xlim([10, 25])
            ax.set_ylim([-1.5, 1.5])
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            # ax.yaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which="both", linestyle=":")

    # Format the PSD plot
    for psd_ax in pt_axes[-1]:
        psd_ax.set_title("Power Spectral Density")
        psd_ax.set_ylabel('Amplitude [dBc]')
        psd_ax.set_xlabel('Normalized Frequency [$T_b$ = 1]')
        psd_ax.set_ylim([-60, 20])
        psd_ax.yaxis.set_major_locator(MultipleLocator(10))
        psd_ax.set_xlim([-2, 2])
        psd_ax.legend(loc="upper center", fontsize=8, ncol=3)
        psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))
        psd_ax.grid(which="both", linestyle=":")

    fig.tight_layout()
    fig.show()