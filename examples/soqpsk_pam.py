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
from waveforms.viterbi.trellis import (
    FiniteStateMachine,
    SOQPSKTrellis4x2,
    SOQPSKTrellis8x1,
)
from waveforms.viterbi.algorithm import viterbi_algorithm, SOQPSKTrellisDetector


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
        unfiltered_signal: NDArray[np.complex128] = modulated_signal + noise
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

        # PAM De-composition rho pulses/matched filters
        rho = rho_pulses(
            pulse_filter,
            mod_index,
            sps,
            k_max=2
        )
        d_max = max([rho_k.size for rho_k in rho])
        L = int(pulse_filter.size/sps)

        # Match filter outputs
        k_max, num_symbols = pseudo_symbols.shape
        mf_outputs = np.zeros((num_symbols, received_signal.size), dtype=np.complex128)
        for sym_idx in range(num_symbols):
            for k in range(k_max):
                # Zero-pad all to length d_max for alignment
                rk = np.concatenate((rho[k], np.zeros(d_max-rho[k].size)))
                mf_outputs[sym_idx, :] += (
                    np.convolve(received_signal, rk, mode="same") *
                    np.conj(pseudo_symbols[k, sym_idx])
                )

        # state_history = np.zeros(L)
        # phase_state_index ($I_{n-L}$)
        # for each timed sample n:
        #     1. calculate branch increment for each hypothetical symbol
        #     2. perform viterbi algorithm to determine n-L+1 symbol
        #       a. Inputs Phase state of n-L, Hypothetical symbols n-L+1..n
        #       b. Outputs 
        #     3. Therefore, a GNURadio style block would require history/delay of L and tracking phase state

        # Initialize FSM
        fsm = FiniteStateMachine(trellis=SOQPSKTrellis8x1)
        det = SOQPSKTrellisDetector(length=2)

        # THIS WIP VARIATION ATTEMPTS TO DO A VA TRACEBACK ON EACH SYMBOL
        phase_idx: int = 0  # Phase state from symbols n=(-inf,n-L)
        trace_length = 2*L if label == "TG" else 8  # What does the literature say?
        timing_offset = -1 if label == "TG" else -4  # TODO Magic numbers
        correlative_state_symbols = np.zeros(trace_length, dtype=np.int8)
        correlative_state_increments = np.zeros((num_symbols, trace_length), dtype=np.float64)
        output_symbols = []
        output_bits = []
        output_symbols2 = []
        output_bits2 = []

        for n in range(received_signal.size-trace_length*sps):
            # Placeholder timing recovery, will replace with Non-data-aided method
            if (n+timing_offset) % sps:
                continue

            # Remove oldest correlative symbol
            correlative_state_symbols = np.roll(correlative_state_symbols, -1)

            # Increments need to be re-calculated on each loop since all increments depend on preceding phase state
            correlative_state_increments = np.roll(correlative_state_increments, -1, axis=1)

            # Calculate newest correlative branch metric increment for each hypothetical symbol
            for sym_idx in range(num_symbols):
                hypothetical_idx = phase_idx + correlative_state_symbols[:-1].sum()
                correlative_state_increments[sym_idx, -1] = np.real(
                    np.exp(-j * 2 * np.pi * (hypothetical_idx % P) / P) * mf_outputs[sym_idx, n+sps*trace_length]
                )

            # VA Traceback, update n-L phase state for next iteration, emit n-L symbol and bit
            rbits, rsyms = viterbi_algorithm(
                correlative_state_increments,
                fsm,
                start=phase_idx%P
            )
            correlative_state_symbols[-1] = rsyms[-1] / 2
            phase_idx += correlative_state_symbols[0]
            output_symbols.append(correlative_state_symbols[0])
            output_bits.append(rbits[0])

            # Second implementation
            rbits2, rsyms2 = det.va_iteration(mf_outputs[:, n+sps*trace_length])
            output_symbols2.append(rsyms2[0] / 2)
            output_bits2.append(rbits2[0])

        iter_va_output_symbols = np.array(output_symbols, dtype=np.int8)
        iter_va_output_bits = np.array(output_bits, dtype=np.uint8)
        iter_va_errors, = np.where(symbols[:iter_va_output_symbols.size]-iter_va_output_symbols)

        iter_va_output_symbols2 = np.array(output_symbols2, dtype=np.int8)
        iter_va_output_bits2 = np.array(output_bits2, dtype=np.uint8)

        # Display cumulative detection error count
        # TODO Get rid of magic numbers
        aligned_syms = symbols[8-det.length:] if label == "MIL" else symbols[16-det.length:]
        min_size = min(aligned_syms.size, iter_va_output_symbols2.size)
        t = np.linspace(0, min_size-1, num=min_size)
        error_idx, = np.where(iter_va_output_symbols2[:min_size]-aligned_syms[:min_size])
        errors_ax.plot(t[error_idx], np.cumsum(np.ones(error_idx.shape)), color="k", marker="x")

        print(len(error_idx), len(iter_va_errors))

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

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    fig_eye.show()
