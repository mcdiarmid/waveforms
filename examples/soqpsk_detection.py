import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, firwin, lfilter

from waveforms.cpm.modulate import cpm_modulate
from waveforms.cpm.pamapprox import rho_pulses
from waveforms.cpm.soqpsk import (
    RecursiveDD,
    RecursiveDE,
    SOQPSKDifferentialDecoder,
    SOQPSKDifferentialEncoder,
    freq_pulse_soqpsk_mil,
    freq_pulse_soqpsk_tg,
)
from waveforms.cpm.trellis.encoder import TrellisEncoder
from waveforms.cpm.trellis.model import (
    SOQPSKTrellis4x2,
    SOQPSKTrellis4x2DiffEncoded,
)
from waveforms.glfsr import PNSequence
from waveforms.noise import generate_complex_awgn
from waveforms.viterbi.algorithm import SOQPSKTrellisDetector
from waveforms.viz import plot_constellation


# Set seeds so iterations on implementation can be compared better
rng = np.random.Generator(np.random.PCG64(seed=1))

PN_DEGREE = 17
DATA_GEN = PNSequence(PN_DEGREE)
DATA_BUFFER = np.packbits(DATA_GEN.generate_sequence())

_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Signal Processing Constants
    sps = 10
    fft_size = 2**9
    ebn0 = 9  # Quasonix RDMS has a 1e-5 BER for SOQPSK-TG @ Eb/N0 = 11.2 dB
    sigma = np.sqrt(sps / np.power(10, ebn0 / 10) / 2)
    filter_type = "hamming"

    # Encoding Constants
    use_irig_diff_encoding = True
    use_recursive_diff_encoding = False
    use_standard_trellis = True

    if use_standard_trellis and not use_irig_diff_encoding and not use_recursive_diff_encoding:
        _logger.warning(
            "No differential encoding utilized. "
            "Trellis detectors susceptible to initial state determining output sequence."
        )

    # Plotting Constants
    t_min, t_max = 100, 120

    # Pre-detector filter
    filter_delay = 0
    filter_method = lfilter

    # IIR Filter options
    if filter_type == "butter":
        order = 5
        lpf = butter(order, 0.45, btype="low", fs=sps, analog=False)
        filter_method = filtfilt
        _logger.warning(
            "Using filtfilt for IIR filter to linearize phase response.  "
            "This filtering method is non-trivial to implement in real-time systems."
        )

    # FIR Filter options - by default generate forward-reverse filter to linearize phase response
    elif filter_type == "hamming":
        fir_taps = firwin(sps * 8 + 1, 0.5, window="hamming", fs=sps)
        forward_reverse_taps = np.convolve(fir_taps, fir_taps[::-1], mode="full")
        lpf = forward_reverse_taps, [1]
        filter_delay = int(len(forward_reverse_taps) / 2)

    elif filter_type == "kaiser":
        fir_taps = firwin(sps * 8 + 1, 0.5, window=("kaiser", 8), fs=sps)
        forward_reverse_taps = np.convolve(fir_taps, fir_taps[::-1], mode="full")
        lpf = forward_reverse_taps, [1]
        filter_delay = int(len(forward_reverse_taps) / 2)

    else:
        lpf = None

    # Bits of information to transmit
    bit_array = np.unpackbits(DATA_BUFFER)
    input_bits = bit_array[:]

    # Map bits of information to ternary symbols
    symbol_precoder = TrellisEncoder(
        SOQPSKTrellis4x2 if use_standard_trellis else SOQPSKTrellis4x2DiffEncoded
    )
    if use_irig_diff_encoding:
        bit_array = SOQPSKDifferentialEncoder()(bit_array)

    if use_recursive_diff_encoding and use_standard_trellis:
        bit_array = RecursiveDE()(bit_array)

    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, iq_axes = plt.subplots(4, 2, figsize=(12, 12), dpi=100)
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
        (freq_pulse_soqpsk_mil(sps=sps), "MIL", 1 / 4),
        (freq_pulse_soqpsk_tg(sps=sps), "TG", 1 / 4),
    )
    for i, (pulse_filter, label, mod_index) in enumerate(pulses_colors_labels):
        # Assign axes
        iq_ax: Axes = iq_axes[0, i]
        mf_ax: Axes = iq_axes[1, i]
        psd_ax: Axes = iq_axes[2, i]
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
        freq_pulses = np.angle(modulated_signal[1:] * modulated_signal.conj()[1:]) * sps / np.pi

        # Received signal
        unfiltered_signal: NDArray[np.complex128] = modulated_signal + noise
        received_signal: NDArray[np.complex128]
        received_signal = filter_method(*lpf, unfiltered_signal) if lpf else unfiltered_signal[:]

        # Display transmitted and received signal in the time domain
        iq_ax.plot(
            normalized_time[t_min * sps : t_max * sps],
            modulated_signal.real[t_min * sps : t_max * sps],
            "b-",
            alpha=1.0,
            label=r"Re[$s(t)]$",
        )
        iq_ax.plot(
            normalized_time[t_min * sps : t_max * sps],
            modulated_signal.imag[t_min * sps : t_max * sps],
            "r-",
            alpha=1.0,
            label=r"Im[$s(t)]$",
        )

        # Unfiltered Signal
        iq_ax.plot(
            normalized_time[t_min * sps : t_max * sps],
            unfiltered_signal.real[t_min * sps : t_max * sps],
            "b-",
            alpha=0.2,
            label=r"$Re[s(t)+N]$",
        )
        iq_ax.plot(
            normalized_time[t_min * sps : t_max * sps],
            unfiltered_signal.imag[t_min * sps : t_max * sps],
            "r-",
            alpha=0.2,
            label=r"$Im[s(t)+N]$",
        )

        # Filtered Signal
        iq_ax.plot(
            normalized_time[: -filter_delay or None][t_min * sps : t_max * sps],
            received_signal.real[filter_delay:][t_min * sps : t_max * sps],
            "b-",
            alpha=0.4,
        )
        iq_ax.plot(
            normalized_time[: -filter_delay or None][t_min * sps : t_max * sps],
            received_signal.imag[filter_delay:][t_min * sps : t_max * sps],
            "r-",
            alpha=0.4,
        )

        pulse_ax = iq_ax.twinx()
        pulse_ax.stem(
            normalized_time[sps::sps][t_min:t_max],
            symbols[t_min:t_max] / 2,
            markerfmt="ko",
            linefmt="k-",
            basefmt=" ",
            label="Symbol",
        )
        pulse_ax.plot(
            normalized_time[:-1][t_min:t_max],
            freq_pulses[t_min:t_max],
            "k-",
            alpha=0.4,
            label="Frequency Pulses",
        )
        pulse_ax.set_ylim(-np.pi / 2, np.pi / 2)

        # Display transmitted and received signal PSD to illustrate SNR
        psd_ax.psd(
            modulated_signal[: fft_size * 100],
            NFFT=fft_size,
            Fs=sps,
            label="$s(t)$",
            scale_by_freq=False,
        )
        psd_ax.psd(
            noise[: fft_size * 100],
            NFFT=fft_size,
            Fs=sps,
            label="$N(t)$",
            scale_by_freq=False,
        )
        psd_ax.psd(
            received_signal[: fft_size * 100],
            NFFT=fft_size,
            Fs=sps,
            label="$r(t)$",
            scale_by_freq=False,
        )

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
            # Generate matched filter for ternary symbol
            matched_filted = np.zeros(d_max, dtype=np.complex128)
            for k in range(k_max):
                # Zero-pad all to length d_max for alignment
                matched_filted[: rho[k].size] += rho[k] * np.conj(pseudo_symbols[k, sym_idx])

            # Convolve received signal with matched filter
            mf_outputs_pam[sym_idx, :] += np.convolve(
                received_signal,
                matched_filted,
                mode="same",
            )
            sym = 2 * (sym_idx - 1)
            (line,) = mf_ax.plot(
                normalized_time[t_min * sps : t_max * sps],
                mf_outputs_pam[sym_idx, filter_delay:].real[t_min * sps : t_max * sps],
                label=f"MF Re[{sym:+}]",
                linestyle="-",
                marker="s",
                markevery=(0 if label == "TG" else sps, sps),
            )
            mf_ax.plot(
                normalized_time[t_min * sps : t_max * sps],
                mf_outputs_pam[sym_idx, filter_delay:].imag[t_min * sps : t_max * sps],
                label=f"MF Im[{sym:+}]",
                color=line.get_color(),
                linestyle="--",
                marker="s",
                markevery=(0 if label == "TG" else sps, sps),
            )

        # DETECTION METHODS
        output_bits_dict = {}

        # SxS Detection
        constellation_signal: NDArray[np.complex128] = np.zeros_like(received_signal)
        constellation_signal[sps:] += (received_signal * np.exp(1j * np.pi / 4)).real[:-sps]
        constellation_signal[:] += (received_signal * np.exp(1j * np.pi / 4)).imag * 1j
        sxs_output = []
        sxs_delay = 2
        id_filter_n = sps
        constellation_out = []

        for n in range(filter_delay + sps, received_signal.size - sps, sps * 2):
            # Integrate & Dump
            soft_symbol: complex = constellation_signal[
                n - int(id_filter_n / 2) :
                n + id_filter_n - int(id_filter_n / 2)
            ].sum()
            constellation_out.append(soft_symbol / id_filter_n)
            sxs_output.append(int(soft_symbol.real > 0))
            sxs_output.append(int(soft_symbol.imag > 0))

        # Decoding Steps
        sxs_output = np.array(sxs_output, dtype=np.uint8)
        if use_recursive_diff_encoding or not use_standard_trellis:
            sxs_output = RecursiveDD()(sxs_output)

        if use_irig_diff_encoding:
            sxs_output = SOQPSKDifferentialDecoder()(sxs_output)

        # Align with input_bits
        output_bits_dict["Single Symbol"] = sxs_output[sxs_delay:]

        # Initialize FSM
        # WIP - ATTEMPTS TO DO A VA TRACEBACK ON EACH SYMBOL
        va_delay = 2
        delay = int(filter_delay / sps)
        for mf_outputs, detector_type in zip(
            (mf_outputs_pt, mf_outputs_pam),
            ("PT Viterbi", "PAM Viterbi"),
        ):
            det = SOQPSKTrellisDetector(length=4, differantial_encoding=not use_standard_trellis)
            output_symbols = []
            output_bits = []

            # Should replace magic numbers.  Ideally this gets solved with timing recovery.
            if detector_type == "PT":
                timing_offset = -1 if label == "TG" else -1
            else:
                timing_offset = 0 if label == "TG" else -3

            for n in range(received_signal.size - det.length * sps):
                # Placeholder timing recovery, will replace with Non-data-aided method
                if (n + timing_offset + filter_delay) % sps:
                    continue

                # Perform Fixed Length VA Traceback
                sym_idx = int((n + timing_offset) / sps)
                rbits, rsyms = det.iteration(mf_outputs[:, n])
                output_symbols.append(rsyms[va_delay])
                output_bits.append(rbits[va_delay])

            # Run decoding steps
            output_bits = np.array(output_bits, dtype=np.uint8)

            if use_recursive_diff_encoding and use_standard_trellis:
                output_bits = RecursiveDD()(output_bits)

            if use_irig_diff_encoding:
                output_bits = SOQPSKDifferentialDecoder()(output_bits)

            # Align with input_bits
            output_bits_dict[detector_type] = output_bits[det.length + delay - va_delay :]

        # Plot cumulative errors over time and log error metrics
        for detector_type, output_bits in output_bits_dict.items():
            # Align input and output bitstreams, count errors, calculate error rate, plot errors
            min_size = min(len(input_bits), len(output_bits))
            t = np.linspace(0, min_size - 1, num=min_size)
            bit_err_idx = np.where(output_bits[:min_size] - input_bits[:min_size])[0]
            ber = len(bit_err_idx) / min_size

            # Handle inverted output (frame sync is responsible for this)
            if ber > (1 - ber):
                ber = 1 - ber
                output_bits[:] = 1 - output_bits[:]
                bit_err_idx = np.where(output_bits[:min_size] - input_bits[:min_size])[0]

            # Log BER and plot cumulative errors
            log_msg = f"SOQPSK-{label} {detector_type}: Eb/N0 = {ebn0:.2f} dB, BER = {ber:.3e}"
            _logger.info(log_msg)
            errors_ax.plot(
                t[bit_err_idx],
                np.cumsum(np.ones(bit_err_idx.shape)),
                marker="x",
                label=f"{detector_type} (BER = {ber:.3e})",
            )

    for ax in iq_axes[0, :]:
        ax: Axes
        ax.grid(which="both", linestyle=":")
        ax.set_xlim([t_min, t_max])
        ax.set_ylim([-4, 4])
        ax.legend(loc="upper center", fontsize=8, ncols=4)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    for ax in iq_axes[1, :]:
        ax: Axes
        ax.set_xlim([t_min, t_max])
        ax.set_ylim([-sps * 2, sps * 2])
        ax.legend(loc="upper center", fontsize=8, ncols=3)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    for psd_ax in iq_axes[2, :]:
        psd_ax.set_title("Power Spectral Density")
        psd_ax.set_ylabel("Amplitude [dBc]")
        psd_ax.set_xlabel("Normalized Frequency [$T_b$ = 1]")
        psd_ax.set_ylim([-60, 0])
        psd_ax.yaxis.set_major_locator(MultipleLocator(10))
        psd_ax.set_xlim([-2, 2])
        psd_ax.legend(loc="upper right", fontsize=8, ncol=1)
        psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))
        psd_ax.grid(which="both", linestyle=":")

    for ax in iq_axes[3, :]:
        ax.grid(which="both", linestyle=":")
        ax.set_ylabel("Cumulative Bit Errors")
        ax.set_xlabel("Symbol Time [nT]")
        ax.set_ylim(0, None)
        ax.set_xlim(0, symbols.size - 1)
        ax.legend(loc="upper left", fontsize=8, ncol=1)

    # Create and format constellation axis
    fig_const, ax_const = plt.subplots(1, figsize=(4, 4), dpi=100)
    ax_const.set_xlim([-1.5, +1.5])
    ax_const.set_ylim([-1.5, +1.5])
    ax_const.xaxis.set_major_locator(MultipleLocator(1))
    ax_const.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax_const.yaxis.set_major_locator(MultipleLocator(1))
    ax_const.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax_const.set_title("SOQPSK SxS I&D Constellation")

    plot_constellation(
        signal=np.array(constellation_out, dtype=np.complex128),
        n=8192,
        axis=ax_const,
        linestyle=" ",
        marker="s",
        markersize=1,
        color="b",
    )

    fig_eye.tight_layout()
    fig_eye.savefig(Path(__file__).parent.parent / "images" / "soqpsk_detection.png")
    plt.show()
