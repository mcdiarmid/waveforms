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
from waveforms.viz import plot_constellation


# Set seeds so iterations on implementation can be compared better
rng = np.random.Generator(np.random.PCG64(seed=1))

PN_DEGREE = 13
DATA_GEN = PNSequence(PN_DEGREE)
DATA_BUFFER = np.packbits(DATA_GEN.generate_sequence())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Constants
    sps = 20

    # Bits of information to transmit
    bit_array = np.unpackbits(DATA_BUFFER)

    # Convert bits to symbols
    symbol_precoder = TrellisEncoder(SOQPSKTrellis4x2DiffEncoded)
    symbols = symbol_precoder(bit_array)

    # Create plots and axes
    fig_eye, iq_axes = plt.subplots(3, 2, figsize=(12, 8), dpi=100)

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
        rho_ax: Axes = iq_axes[1, i]
        mf_ax: Axes = iq_axes[2, i]

        # Modulate the input symbols
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=pulse_filter,
            sps=sps,
        )
        modulated_signal[:] *= np.exp(+1j * np.pi / 4)

        # Display transmitted and received signal in the time domain
        iq_ax.plot(
            normalized_time,
            modulated_signal.real,
            "b-",
            alpha=1.0,
            label=r"CPM-I",
            marker="o",
            markersize=5,
            markevery=(int(sps / 2), sps),
        )
        iq_ax.plot(
            normalized_time,
            modulated_signal.imag,
            "r-",
            alpha=1.0,
            label=r"CPM-Q",
            marker="o",
            markersize=5,
            markevery=(int(sps / 2), sps),
        )

        pulse_ax = iq_ax.twinx()
        pulse_ax.stem(
            normalized_time[sps::sps],
            symbols / 2,
            bottom=0,
            markerfmt="ko",
            linefmt="k-",
            basefmt="k",
            label="Symbol",
        )
        pulse_ax.set_ylim(-np.pi / 2, np.pi / 2)

        # Pulse Truncation Filters
        L = int(pulse_filter.size / sps)
        truncation = 1
        q = np.cumsum(pulse_filter) / sps
        pt_start = int((L - truncation) * sps / 2)
        pt_end = int((L + truncation) * sps / 2) + 1
        truncated_phase_pulse = q[pt_start:pt_end]

        # PAM De-composition rho pulses/matched filters
        rho = rho_pulses(pulse_filter, mod_index, sps, k_max=2)
        d_max = max([rho_k.size for rho_k in rho])
        k_max, num_symbols = pseudo_symbols.shape

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

        # Re-construct signal from PAM pulses following Laurent Decomposition principles
        num_points = (symbols.size + 1) * sps
        pseudo_symbols_interpolated = np.zeros((k_max, num_points), dtype=np.complex128)
        pam_approx = np.zeros(len(symbols)*sps + d_max + 1, dtype=np.complex128)
        symbol_indicies = (symbols / 2 + 1).astype(np.int8)

        # Cumulative phase state from symbol
        phase_index = np.cumsum(np.concatenate([[2], symbols], dtype=np.int32))
        phase_state = np.exp(1j * phase_index * np.pi / 4).round()

        # Plot the matched filters
        matched_filters: list[NDArray[np.complex128]] = []
        t = np.linspace(0, (d_max - 1) / sps, num=d_max)
        for j in range(num_symbols)[::-1]:
            matched_filter = np.zeros(d_max, dtype=np.complex128)
            for k in range(k_max):
                matched_filter[:rho[k].size] += np.conj(pseudo_symbols[k][j]) * rho[k]
            line, = mf_ax.plot(
                t,
                matched_filter.real,
                label=rf"$H_{{{j-1:+}}}(t)$",
                linestyle="-",
                linewidth=1+j,
            )
            mf_ax.plot(
                t,
                matched_filter.imag,
                linestyle="--",
                linewidth=1+j,
                color=line.get_color(),
            )
            matched_filters.append(matched_filter)

        # PAM Approximation of a CPM signal
        for idx, symbol in enumerate(symbols):
            symbol_idx = int(symbol / 2) + 1
            pam_approx[idx*sps:idx*sps + d_max] += phase_state[idx] * matched_filters[symbol_idx]

        # Alternatively, via convolution of rho[k] against an array of pseudo symbols zero-filled to sps
        # for k in range(k_max):
        #     delay = int((d_max - rho[k].size) / 2)
        #     pseudo_symbols_interpolated[k, sps - delay : -1 - delay : sps] = (
        #         np.take(pseudo_symbols[k], symbol_indicies) * phase_state[:-1]
        #     )
        #     pam_approx[:] += np.convolve(pseudo_symbols_interpolated[k, :], rho[k], mode="same")

        pam_approx = pam_approx[sps*L:-2]
        pam_delay = int((d_max + sps) // 2)
        iq_ax.plot(
            normalized_time[pam_delay:],
            pam_approx.real[:-pam_delay],
            "b--",
            alpha=0.5,
            label=rf"PAM-I ($k_{{max}}={k_max}$)",
        )
        iq_ax.plot(
            normalized_time[pam_delay:],
            pam_approx.imag[:-pam_delay],
            "r--",
            alpha=0.5,
            label=rf"PAM-Q ($k_{{max}}={k_max}$)",
        )

    for ax in iq_axes[0, :]:
        ax: Axes
        ax.grid(which="both", linestyle=":")
        ax.set_xlim([100, 130])
        ax.legend(loc="upper center", fontsize=8, ncols=4)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylim([-3, 3])

    for rho_ax in iq_axes[1, :]:
        rho_ax.grid(which="both", linestyle=":")
        rho_ax.legend()

    for mf_ax in iq_axes[2, :]:
        mf_ax.grid(which="both", linestyle=":")
        mf_ax.legend()

    qpsk_esque_signal = np.zeros_like(pam_approx)
    pam_approx[:] *= np.exp(1j * np.pi / 4)
    qpsk_esque_signal[sps:] += pam_approx.real[:-sps]
    qpsk_esque_signal[:] += pam_approx.imag * 1j

    fig_const, ax_const = plt.subplots(1, figsize=(4, 4), dpi=100)
    ax_const.set_xlim([-2, 2])
    ax_const.set_ylim([-2, 2])
    ax_const.set_title("SOQPSK PAM Approximation Constellation")
    fig_const = plot_constellation(
        signal=qpsk_esque_signal[sps :: sps * 2][1:-2],
        n=8192,
        axis=ax_const,
        linestyle=" ",
        marker="s",
    )

    images_dir = Path(__file__).parent.parent / "images"
    fig_eye.tight_layout()
    fig_eye.savefig(images_dir / "soqpsk_laurent_decomp.png")
    fig_const.tight_layout()
    fig_const.savefig(images_dir / "soqpsk_pam_k2_constellation.png")
    plt.show()
