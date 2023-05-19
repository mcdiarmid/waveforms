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
from waveforms.lpf import hann_window


DATA_HEADER = b"\x1b\x1bHello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(100)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


if __name__ == "__main__":
    # Constants
    sps = 40
    fft_size = 2**9
    pulse_pad = 0.5
    alpha = 1, 0
    noise_variance = 0.1

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)


    fig, pt_axes = plt.subplots(3, 2, figsize=(12, 10), dpi=100)
    lpf = hann_window(sps, 0.9*2) / sps

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
            scale=noise_variance*np.sqrt(2)/2,
            size=(modulated_signal.size, 2)
        ).view(np.complex128).flatten()
        modulated_signal *= np.exp(-j*np.pi/4)
        modulated_signal *= np.exp(-j*np.pi/5)
        freq_pulses = np.angle(modulated_signal[1:] * modulated_signal.conj()[:-1]) * sps / np.pi

        # Received signal
        received_signal: NDArray[np.float64] = modulated_signal + noise
        lpf_rx_signal: NDArray[np.float64] = np.convolve(received_signal, lpf, mode="same")
    
        quad_demod = np.angle(lpf_rx_signal[1:] * lpf_rx_signal.conj()[:-1]) * sps / np.pi

        idx_start = int((pulse_filter.size - sps)/2)
        idx_end = int((pulse_filter.size + sps)/2)
        truncated_pulse = pulse_filter[idx_start:idx_end]
        truncated_pulse = normalize_cpm_filter(sps=sps, g=pulse_filter[idx_start:idx_end])
        matched_filters = [
            np.exp(-j*2*np.pi*mod_index*a_k*truncated_pulse)
            for a_k in alpha
        ]
        mf_rx = np.convolve(lpf_rx_signal, matched_filters[1], mode="same") / sps
        mf_rx_quad_demod = np.angle(mf_rx[1:] * mf_rx.conj()[:-1]) * sps / np.pi

        # Assign axes
        iq_ax: Axes = pt_axes[0, i]
        mf_ax: Axes = pt_axes[1, i]
        mf_theta_ax = mf_ax.twinx()
        psd_ax: Axes = pt_axes[2, i]

        # Plot transmitted, received, and MF received (IQ) signal
        iq_ax.plot(normalized_time, modulated_signal.real, color="blue", alpha=1)
        iq_ax.plot(normalized_time, modulated_signal.imag, color="red", alpha=1)
        iq_ax.plot(normalized_time, lpf_rx_signal.real, color="blue", alpha=0.5)
        iq_ax.plot(normalized_time, lpf_rx_signal.imag, color="red", alpha=0.5)
        iq_ax.plot(normalized_time, mf_rx.real, color="darkblue", alpha=0.8)
        iq_ax.plot(normalized_time, mf_rx.imag, color="darkred", alpha=0.8)

        # pulse_ax
        mf_ax.stem(normalized_time[sps:-1:sps], symbols, linefmt="k", markerfmt="kx", basefmt="k")

        # Create branch metrics
        zeds = []
        
        for symbol, color in zip(alpha, ("b", "g", "r")):
            # Convolving the matched filter and sampling at the correct time
            # is the same as the correctly sampled integral (intergrate and dump)
            mf = np.exp(-j*2*np.pi*mod_index*symbol*truncated_pulse)
            z_k = np.convolve(lpf_rx_signal, mf, mode="same") / sps
            z_k[:] *= np.exp(j*np.pi/5)
            mf_ax.plot(
                normalized_time,
                z_k.real*z_k.imag,
                color=color,
                linestyle="-",
                alpha=0.7,
                label=""
            )
            mf_ax.plot(
                normalized_time[::sps],
                z_k.real[::sps]*z_k.imag[::sps],
                color=color,
                linestyle="",
                marker="o",
                alpha=0.7,
            )
            zeds.append(z_k)
            # mf_ax.plot(
            #     normalized_time,
            #     z_k.imag,
            #     color=color,
            #     linestyle=":",
            #     alpha=0.7,
            # )
            # mf_ax.plot(
            #     normalized_time[::sps],
            #     z_k.imag[::sps],
            #     color=color,
            #     linestyle="",
            #     marker="o",
            #     alpha=0.7,
            # )
        mf_theta_ax.plot(
            normalized_time[:-1],
            np.abs(np.diff(np.angle(
                zeds[0].real*zeds[0].imag * j + 
                zeds[1].real*zeds[1].imag 
            ) % (np.pi)))
        )
        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
            label="$s(t)$",
        )
        psd_ax.psd(
            received_signal,
            NFFT=fft_size,
            Fs=sps,
            label="$s(t) + N(t)$",
        )
        psd_ax.psd(
            lpf_rx_signal,
            NFFT=fft_size,
            Fs=sps,
            label="LPF",
        )

    for ax_row in pt_axes[:-1]:
        for ax in ax_row:
            ax: Axes
            ax.set_xlim([symbols.size/2, symbols.size/2 + 15])
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