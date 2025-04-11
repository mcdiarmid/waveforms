import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from waveforms.cpm.modulate import cpm_modulate
from waveforms.cpm.pamapprox import rho_pulses
from waveforms.cpm.soqpsk import (
    freq_pulse_soqpsk_mil,
    freq_pulse_soqpsk_tg,
)
from waveforms.cpm.trellis.encoder import TrellisEncoder
from waveforms.cpm.trellis.model import (
    SOQPSKTrellis4x2DiffEncoded,
)
from waveforms.glfsr import PNSequence
from waveforms.noise import generate_complex_awgn
from waveforms.viterbi.algorithm import SOQPSKTrellisDetector


# Set seeds so iterations on implementation can be compared better
rng = np.random.Generator(np.random.PCG64(seed=1))

PN_DEGREE = 15
DATA_GEN = PNSequence(PN_DEGREE)
DATA_BUFFER = np.packbits(DATA_GEN.generate_sequence())

_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Constants
    sps = 10
    fft_size = 2**9
    pulse_pad = 0.5
    sigma = np.sqrt(2) / 2
    P = 4

    # Bits of information to transmit
    bit_array = np.unpackbits(DATA_BUFFER)

    # Convert bits to symbols
    symbol_precoder = TrellisEncoder(SOQPSKTrellis4x2DiffEncoded)
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, iq_axes = plt.subplots(4, 2, figsize=(12, 10), dpi=100)
    for ax in iq_axes.flatten():
        ax.grid(which="both", linestyle=":")

    # Generate pseudo-symbols
    pseudo_symbols = np.array(
        [
            [-1j, 1, 1j],
            [np.sqrt(2) / 2 * (1 - 1j), np.sqrt(2) / 2, np.sqrt(2) / 2 * (1 + 1j)],
        ],
        dtype=np.complex128,
    )

    # Simulate the following SOQPSK Waveforms
    pulses_colors_labels = (
        (freq_pulse_soqpsk_mil(sps=sps), "MIL", 1 / P),
        (freq_pulse_soqpsk_tg(sps=sps), "TG", 1 / P),
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
        noise = generate_complex_awgn(sigma, modulated_signal.size, rng)
        modulated_signal[:] *= np.exp(-1j * np.pi / 4)
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
        pulse_ax.stem(
            normalized_time[sps::sps],
            symbols / 2,
            markerfmt="ko",
            linefmt="k-",
            basefmt=" ",
            label="Symbol",
        )
        pulse_ax.plot(
            normalized_time[:-1],
            freq_pulses,
            "k-",
            alpha=0.4,
            label="Frequency Pulses",
        )
        pulse_ax.set_ylim(-np.pi / 2, np.pi / 2)

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

        ebn0_calc = 10 * np.log10(sps / (2 * sigma ** 2))

        # Pulse Truncation Filters
        L = int(pulse_filter.size / sps)
        truncation = 1
        q = np.cumsum(pulse_filter) / sps
        pt_start = int((L - truncation) * sps / 2)
        pt_end = int((L + truncation) * sps / 2) + 1
        truncated_phase_pulse = q[pt_start:pt_end]
        mf_outputs_pt = np.zeros((3, received_signal.size), dtype=np.complex128)
        mf_outputs_pt[0, :] = np.convolve(
            received_signal,
            np.exp(-2j * np.pi * mod_index * -2 * truncated_phase_pulse),
            mode="same",
        )
        mf_outputs_pt[1, :] = np.convolve(
            received_signal,
            np.exp(-2j * np.pi * mod_index * +0 * truncated_phase_pulse),
            mode="same",
        )
        mf_outputs_pt[2, :] = np.convolve(
            received_signal,
            np.exp(-2j * np.pi * mod_index * +2 * truncated_phase_pulse),
            mode="same",
        )

        # PAM De-composition rho pulses/matched filters
        rho = rho_pulses(pulse_filter, mod_index, sps, k_max=2)
        d_max = max([rho_k.size for rho_k in rho])

        # Match filter outputs
        k_max, num_symbols = pseudo_symbols.shape
        mf_outputs_pam = np.zeros((num_symbols, received_signal.size), dtype=np.complex128)
        for sym_idx in range(num_symbols):
            for k in range(k_max):
                # Zero-pad all to length d_max for alignment
                rk = np.concatenate((rho[k], np.zeros(d_max - rho[k].size)))
                mf_outputs_pam[sym_idx, :] += np.convolve(
                    received_signal,
                    rk,
                    mode="same",
                ) * np.conj(pseudo_symbols[k, sym_idx])

        # Initialize FSM
        # WIP - ATTEMPTS TO DO A VA TRACEBACK ON EACH SYMBOL
        for mf_outputs, detector_type in zip((mf_outputs_pt, mf_outputs_pam), ("PT", "PAM")):
            det = SOQPSKTrellisDetector(length=2, differantial_encoding=True)
            output_symbols = []
            output_bits = []
            delay = 0

            # Should replace magic numbers.  Ideally this gets solved with timing recovery.
            if detector_type == "PT":
                timing_offset = -1 if label == "TG" else -1
            else:
                timing_offset = 0 if label == "TG" else -3

            for n in range(received_signal.size - det.length * sps):
                # Placeholder timing recovery, will replace with Non-data-aided method
                if (n + timing_offset) % sps:
                    continue

                # Perform Fixed Length VA Traceback
                sym_idx = int((n + timing_offset) / sps)
                rbits, rsyms = det.iteration(mf_outputs[:, n])
                output_symbols.append(rsyms[0])
                output_bits.append(rbits[0])

            # Calculate number of errors, and visualize
            iter_va_output_symbols = np.array(output_symbols[det.length :], dtype=np.int8)
            iter_va_output_bits = np.array(output_bits[det.length :], dtype=np.uint8)
            aligned_symbols = symbols[delay:]
            min_size = min(aligned_symbols.size, iter_va_output_symbols.size)
            t = np.linspace(0, min_size - 1, num=min_size)
            sym_err_idx, = np.where(iter_va_output_symbols[:min_size] - aligned_symbols[:min_size])
            bit_err_idx, = np.where(iter_va_output_bits[:min_size] - bit_array[:min_size])
            log_msg = (
                f"SOQPSK-{label} {detector_type}: "
                f"Eb/N0 = {ebn0_calc:.2f} dB, "
                f"SER = {len(sym_err_idx)/min_size:.3E} "
                f"BER = {len(bit_err_idx)/min_size:.3E}"
            )
            _logger.info(log_msg)
            errors_ax.plot(
                t[bit_err_idx],
                np.cumsum(np.ones(bit_err_idx.shape)),
                marker="x",
                label=detector_type,
            )

        # Plot Rho pulses used for PAM Approximation
        for k, rho_k, fmt in zip(range(k_max), rho, ("b-", "g--")):
            t = np.linspace(0, (rho_k.size - 1) / sps, num=rho_k.size)
            rho_ax.plot(
                t,
                rho_k,
                fmt,
                label=rf"SOQPSK-{label} $\rho_{k}(t)$",
            )

        rho_ax.set_xlim(0, (max(rho, key=np.size).size - 1) / sps)

    for ax in iq_axes[0, :]:
        ax: Axes
        ax.set_xlim([100, 130])
        ax.legend(loc="upper center", fontsize=8, ncols=4)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    for psd_ax in iq_axes[1, :]:
        psd_ax.set_title("Power Spectral Density")
        psd_ax.set_ylabel("Amplitude [dBc]")
        psd_ax.set_xlabel("Normalized Frequency [$T_b$ = 1]")
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
        ax.set_ylabel("Cumulative Bit Errors")
        ax.set_xlabel("Symbol Time [nT]")
        ax.set_ylim(0, None)
        ax.set_xlim(0, symbols.size - 1)
        ax.legend(loc="upper left", fontsize=8, ncol=1)

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_pam.png")
    plt.show()
