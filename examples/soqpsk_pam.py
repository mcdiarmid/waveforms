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
from waveforms.lpf import kaiser_fir_lpf
from waveforms.viterbi.trellis import FiniteStateMachine, SOQPSKTrellis
from waveforms.viterbi.algorithm import viterbi_algorithm


# Set seeds so iterations on implementation can be compared better
random.seed(1)
np.random.seed(1)
DATA_HEADER = b"\x00\x1b\x1b\x00Hello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(2500)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


if __name__ == "__main__":
    # Constants
    sps = 20
    fft_size = 2**9
    pulse_pad = 0.5
    P = 4
    # lpf = np.ones(1)  # Placeholder low pass filter
    lpf = kaiser_fir_lpf(sps, 0.50)

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, iq_axes = plt.subplots(4, 2, figsize=(12, 10), dpi=100)
    for ax in iq_axes.flatten():
        ax.grid(which="both", linestyle=":")

    # Generate pseudo-symbols
    pseudo_symbols = np.array([
        [-j, 1, j],
        [np.sqrt(2)/2*(1-j), np.sqrt(2)/2, np.sqrt(2)/2*(1+j)],
    ], dtype=np.complex128)

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_mil(sps=sps), 'crimson', 'MIL', 1/P),
        (freq_pulse_soqpsk_tg(sps=sps), 'royalblue', 'TG', 1/P),
    )
    for i, (pulse_filter, color, label, mod_index) in enumerate(pulses_colors_labels):
        # Assign axes
        iq_ax: Axes = iq_axes[0, i]
        psd_ax: Axes = iq_axes[1, i]
        rho_ax: Axes = iq_axes[2, i]
        errors_ax: Axes = iq_axes[3, i]

        # Modulate the input symbols
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols*2,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        noise = np.random.normal(
            loc=0,
            scale=1*np.sqrt(2)/2,
            size=(modulated_signal.size, 2)
        ).view(np.complex128).flatten()
        modulated_signal *= np.exp(-j*np.pi/4)
        freq_pulses = np.angle(modulated_signal[1:] * modulated_signal.conj()[:-1]) * sps / np.pi

        # Received signal
        unfiltered_signal: NDArray[np.float64] = modulated_signal + noise
        # received_signal: NDArray[np.float64] = np.convolve(unfiltered_signal, lpf, mode="same")
        received_signal = unfiltered_signal

        # Display transmitted and received signal in the time domain
        iq_ax.plot(normalized_time, modulated_signal.real, "b-", alpha=1.0, label=r"Re[$s(t)]$")
        iq_ax.plot(normalized_time, unfiltered_signal.real, "b-", alpha=0.4, label=r"$Re[s(t)+N]$")
        iq_ax.plot(normalized_time, modulated_signal.imag, "r-", alpha=1.0, label=r"Im[$s(t)]$")
        iq_ax.plot(normalized_time, unfiltered_signal.imag, "r-", alpha=0.4, label=r"$Im[s(t)+N]$")

        pulse_ax = iq_ax.twinx()
        pulse_ax.stem(normalized_time[::sps][1:-1], symbols, markerfmt="ko", linefmt="k-", basefmt=" ", label="Symbol")
        pulse_ax.plot(normalized_time[:-1], freq_pulses, "k-", alpha=0.4, label="Frequency Pulses")
        pulse_ax.set_ylim(-np.pi/2, np.pi/2)

        # Display transmitted and received signal PSD to illustrate SNR
        pxx_tx, freqs = psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
            label="$s(t)$",
            scale_by_freq=False,
        )
        pxx_specan, freqs = psd_ax.psd(
            modulated_signal + noise,
            NFFT=fft_size,
            Fs=sps,
            label="$s(t) + N(t)$",
            scale_by_freq=False,
        )
        psd_ax.psd(
            received_signal,
            NFFT=fft_size,
            Fs=sps,
            label=r"$\hat{s}(t)$",
            scale_by_freq=False,
        )

        # Spectrum Analyzer style Eb/N0 measurement
        marker1 = 10 * np.log10(pxx_specan[fft_size//2])
        marker2 = 10 * np.log10(pxx_specan[-1])
        rbw = sps/fft_size
        c0n0 = marker1 - 10 * np.log10(rbw)
        n0 = marker2 - 10 * np.log10(rbw)
        c0 = 10 * np.log10(np.power(10, c0n0/10) - np.power(10, n0/10))
        ebn0 = c0 - 2.95 - 0 - n0 

        # PAM De-composition
        rho = rho_pulses(
            pulse_filter,
            mod_index,
            sps,
            k_max=2
        )
        d_max = max([rho_k.size for rho_k in rho])
        L = int(pulse_filter.size/sps)

        # Branch metric increment history and cumulative winning phase state
        z_n_history = []
        phase_idx = 0

        # Initialize FSM
        fsm = FiniteStateMachine(trellis=SOQPSKTrellis)

        for n in range(received_signal.size-d_max):
            # Timing recovery (for now we live in an perfectly synchronized world)
            # TODO implement timing and phase recovery, along with introduced imperfections
            if (n + L*sps/2) % sps:
                continue

            # Branch increments - Iterate over hypothetical symbols
            z_n = np.zeros(3)
            y_arr = np.zeros((3, d_max), np.complex128)
            for sym_idx in range(3):
                # Independent of current state
                y_n = np.zeros(d_max, dtype=np.complex128)
                for k, rho_k in enumerate(rho):
                    y_n[:rho_k.size] += received_signal[n:n+rho_k.size] * rho_k * np.conj(pseudo_symbols[k,sym_idx])
                y_arr[sym_idx] = y_n

                # Trellis/phase state dependent
                z_ln = np.real(np.exp(-j * 2 * np.pi * (phase_idx % P) / P) * np.sum(y_n))
                z_n[sym_idx] = z_ln

            # z_ln is trellis/phase state dependent, should execute a short length viterbi algorithm to have a more accurate
            z_n_history.append(z_n)
            phase_idx += np.argmin(z_n*z_n) - 1  # should modulo 4 here, but can just apply it upon phase offset calc

        # Viterbi algorithm with SOQPSK 8x1 state trellis, should execute on increments[n:n+2*L] per n increment
        increments = np.array(z_n_history).T
        recovered_bits, recovered_symbols = viterbi_algorithm(increments, fsm)

        # Display cumulative detection error count
        t = np.linspace(0, symbols.size-1, num=symbols.size)
        # recovered: NDArray[np.int8] = np.argmin(np.array(z_n_history)**2, axis=1) - 1
        recovered = recovered_symbols / 2
        start_offset = int(label == "TG") * 3  # TODO make this a function of L and d_max
        error_idx, = np.where(symbols[start_offset:recovered.size+start_offset] - recovered)
        errors_ax.plot(t[error_idx], np.cumsum(np.ones(error_idx.shape)), color="k", marker="x")

        # Plot Rho pulses used for PAM Approximation
        for k, rho_k, fmt in zip((0, 1), rho, ("b-", "g--")):
            rho_ax.plot(
                np.linspace(0, (rho_k.size-1)/sps, num=rho_k.size),
                rho_k,
                fmt,
                label=fr"SOQPSK-{label} $\rho_{k}(t)$"
            )

        rho_ax.set_xlim(0, (max(rho, key=np.size).size-1)/sps)

    for ax in iq_axes[0, :]:
        ax: Axes
        ax.set_xlim([10, 30])
        ax.legend(loc="upper center", fontsize=8, ncols=4)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    for psd_ax in iq_axes[1, :]:
        psd_ax.set_title("Power Spectral Density")
        psd_ax.set_ylabel('Amplitude [dBc]')
        psd_ax.set_xlabel('Normalized Frequency [$T_b$ = 1]')
        psd_ax.set_ylim([-60, 0])
        psd_ax.yaxis.set_major_locator(MultipleLocator(10))
        psd_ax.set_xlim([-2, 2])
        psd_ax.legend(loc="upper right", fontsize=8, ncol=1)
        psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))
        psd_ax.grid(which="both", linestyle=":")

    for rho_ax in iq_axes[2, :]:
        rho_ax.grid(which="both", linestyle=":")
        rho_ax.legend()

    for ax in iq_axes[3, :]:
        ax.grid(which="both", linestyle=":")
        ax.set_ylabel("Cumulative Errors")
        ax.set_xlabel("Symbol Time [nT]")
        ax.set_ylim(0, None)
        ax.set_xlim(0, symbols.size-1)

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    fig_eye.show()

    """
    for i, c in zip(range(3), "rgb"):
        for ls, yi in zip("-:", (y_arr[i].real, y_arr[i].imag)):
            ax.plot(yi, linestyle=ls, color=c)
    """
