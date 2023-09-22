import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from waveforms.cpm.soqpsk import (
    freq_pulse_soqpsk_tg,
    freq_pulse_soqpsk_mil,
)
from waveforms.cpm.modulate import cpm_modulate
from waveforms.cpm.helpers import normalize_cpm_filter
from waveforms.cpm.pamapprox import rho_pulses
from waveforms.cpm.trellis.model import (
    SOQPSKTrellis4x2,
)
from waveforms.cpm.trellis.encoder import TrellisEncoder
from waveforms.viterbi.algorithm import SOQPSKTrellisDetector


# Set seeds so iterations on implementation can be compared better
random.seed(1)
np.random.seed(1)
DATA_HEADER = b"\x00\x1b\x1b\x00Hello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for _ in range(10000)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


if __name__ == "__main__":
    # Constants
    sps = 20
    fft_size = 2**9
    pulse_pad = 0.5
    P = 4

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = TrellisEncoder(SOQPSKTrellis4x2)
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
        (freq_pulse_soqpsk_mil(sps=sps), 'MIL', 1/P),
        (freq_pulse_soqpsk_tg(sps=sps), 'TG', 1/P),
    )
    for i, (pulse_filter, label, mod_index) in enumerate(pulses_colors_labels):
        # Assign axes
        iq_ax: Axes = iq_axes[0, i]
        psd_ax: Axes = iq_axes[1, i]
        rho_ax: Axes = iq_axes[2, i]
        errors_ax: Axes = iq_axes[3, i]

        # Modulate the input symbols
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        noise = np.random.normal(
            loc=0,
            scale=1.25*np.sqrt(2)/2,
            size=(modulated_signal.size, 2)
        ).view(np.complex128).flatten()
        modulated_signal *= np.exp(-j*1*np.pi/4)
        freq_pulses = np.angle(modulated_signal[1:] * modulated_signal.conj()[:-1]) * sps / np.pi

        # Received signal
        unfiltered_signal: NDArray[np.complex128] = modulated_signal + noise
        received_signal = unfiltered_signal

        # Display transmitted and received signal in the time domain
        iq_ax.plot(normalized_time, modulated_signal.real, "b-", alpha=1.0, label=r"Re[$s(t)]$")
        iq_ax.plot(normalized_time, unfiltered_signal.real, "b-", alpha=0.4, label=r"$Re[s(t)+N]$")
        iq_ax.plot(normalized_time, modulated_signal.imag, "r-", alpha=1.0, label=r"Im[$s(t)]$")
        iq_ax.plot(normalized_time, unfiltered_signal.imag, "r-", alpha=0.4, label=r"$Im[s(t)+N]$")

        pulse_ax = iq_ax.twinx()
        pulse_ax.stem(normalized_time[::sps][1:-1], symbols/2, markerfmt="ko", linefmt="k-", basefmt=" ", label="Symbol")
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
            received_signal,
            NFFT=fft_size,
            Fs=sps,
            label="$r(t)$",
            scale_by_freq=False,
        )

        # Spectrum Analyzer style Eb/N0 measurement
        marker1 = 10 * np.log10(pxx_specan[fft_size//2])
        marker2 = 10 * np.log10(pxx_specan[-1])
        rbw = sps/fft_size
        c0n0 = marker1 - 10 * np.log10(rbw)
        n0 = marker2 - 10 * np.log10(rbw)
        c0 = 10 * np.log10(np.power(10, c0n0/10) - np.power(10, n0/10))
        ebn0 = c0 - 2.95 - n0 

        # Pulse Truncation Filters
        L = int(pulse_filter.size/sps)
        truncated_freq_pulse = pulse_filter[int((L-1)*sps/2):int((L+1)*sps/2)]
        truncated_phase_pulse = np.cumsum(normalize_cpm_filter(sps=sps, g=truncated_freq_pulse)) / sps
        mf_outputs_pt = np.zeros((3, received_signal.size), dtype=np.complex128)
        mf_outputs_pt[0, :] = np.convolve(received_signal, np.exp(-j*2*np.pi*mod_index*-2*truncated_phase_pulse), mode="same")
        mf_outputs_pt[1, :] = np.convolve(received_signal, np.exp(-j*2*np.pi*mod_index*0*truncated_phase_pulse), mode="same")
        mf_outputs_pt[2, :] = np.convolve(received_signal, np.exp(-j*2*np.pi*mod_index*+2*truncated_phase_pulse), mode="same")

        # PAM De-composition rho pulses/matched filters
        rho = rho_pulses(pulse_filter, mod_index, sps, k_max=2)
        d_max = max([rho_k.size for rho_k in rho])

        # Match filter outputs
        k_max, num_symbols = pseudo_symbols.shape
        mf_outputs_pam = np.zeros((num_symbols, received_signal.size), dtype=np.complex128)
        for sym_idx in range(num_symbols):
            for k in range(k_max):
                # Zero-pad all to length d_max for alignment
                rk = np.concatenate((rho[k], np.zeros(d_max-rho[k].size)))
                mf_outputs_pam[sym_idx, :] += (
                    np.convolve(received_signal, rk, mode="same") *
                    np.conj(pseudo_symbols[k, sym_idx])
                )

        # Initialize FSM
        # WIP - ATTEMPTS TO DO A VA TRACEBACK ON EACH SYMBOL
        for mf_outputs, detector_type in zip((mf_outputs_pt, mf_outputs_pam), ("PT", "PAM")):
            det = SOQPSKTrellisDetector(length=2)
            output_symbols = []
            output_bits = []
            delay = 0

            # TODO Magic numbers.  Ideally this gets solved with timing recovery though
            if detector_type == "PT":
                timing_offset = 0 if label == "TG" else -2
            else:
                timing_offset = -2 if label == "TG" else -4

            for n in range(received_signal.size-det.length*sps):
                # Placeholder timing recovery, will replace with Non-data-aided method
                if (n+timing_offset) % sps:
                    continue

                # Perform Fixed Length VA Traceback
                sym_idx = int((n + timing_offset) / sps)
                rbits, rsyms = det.va_iteration(mf_outputs[:, n])
                output_symbols.append(rsyms[0])
                output_bits.append(rbits[0])

            # Calculate number of errors, and visualize
            iter_va_output_symbols = np.array(output_symbols[det.length:], dtype=np.int8)
            iter_va_output_bits = np.array(output_bits, dtype=np.uint8)
            aligned_symbols = symbols[delay:]
            min_size = min(aligned_symbols.size, iter_va_output_symbols.size)
            t = np.linspace(0, min_size-1, num=min_size)
            error_idx, = np.where(iter_va_output_symbols[:min_size]-aligned_symbols[:min_size])
            print(f"SOQPSK-{label} {detector_type} EbN0 = {ebn0:.2f} dB, SER = {len(error_idx)/min_size:.3E}")
            errors_ax.plot(t[error_idx], np.cumsum(np.ones(error_idx.shape)), marker="x", label=detector_type)

        # Plot Rho pulses used for PAM Approximation
        for k, rho_k, fmt in zip(range(k_max), rho, ("b-", "g--")):
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
        ax.set_ylabel("Cumulative Symbol Errors")
        ax.set_xlabel("Symbol Time [nT]")
        ax.set_ylim(0, None)
        ax.set_xlim(0, symbols.size-1)
        ax.legend(loc="upper left", fontsize=8, ncol=1)

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    fig_eye.show()
