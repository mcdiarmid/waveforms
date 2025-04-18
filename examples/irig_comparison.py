from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from numpy.typing import NDArray

from waveforms.cpm.modulate import cpm_modulate
from waveforms.cpm.multih import (
    MULTIH_IRIG_DENOM,
    MULTIH_IRIG_NUMER,
    freq_pulse_multih_irig,
)
from waveforms.cpm.pcmfm import (
    PCMFM_DENOM,
    PCMFM_NUMER,
    freq_pulse_pcmfm,
)
from waveforms.cpm.soqpsk import (
    SOQPSK_DENOM,
    SOQPSK_NUMER,
    freq_pulse_soqpsk_tg,
)
from waveforms.cpm.trellis.encoder import TrellisEncoder
from waveforms.cpm.trellis.model import (
    SimpleTrellis2,
    SimpleTrellis4,
    SOQPSKTrellis4x2DiffEncoded,
)
from waveforms.glfsr import PNSequence


rng = np.random.Generator(np.random.PCG64())

PN_DEGREE = 15
DATA_GEN = PNSequence(PN_DEGREE)
DATA_BUFFER = np.packbits(DATA_GEN.generate_sequence())


if __name__ == "__main__":
    # Constants
    sps = 20
    fft_size = 2**10
    pulse_pad = 0

    # Bits of information to transmit
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    irig_waveforms = [
        (
            "PCM-FM",
            TrellisEncoder(SimpleTrellis2),
            PCMFM_NUMER / PCMFM_DENOM,
            freq_pulse_pcmfm(sps=sps, order=6),
            1,
        ),
        (
            "SOQPSK-TG",
            TrellisEncoder(SOQPSKTrellis4x2DiffEncoded),
            SOQPSK_NUMER / SOQPSK_DENOM,
            freq_pulse_soqpsk_tg(sps=sps),
            1,
        ),
        (
            "Multi-h CPM",
            TrellisEncoder(SimpleTrellis4),
            MULTIH_IRIG_NUMER / MULTIH_IRIG_DENOM,
            freq_pulse_multih_irig(sps=sps),
            2,
        ),
    ]
    colors = {
        "PCM-FM": "red",
        "SOQPSK-TG": "green",
        "Multi-h CPM": "blue",
    }

    fig_psd, axes = plt.subplots(2, figsize=(6, 8), dpi=100)

    pulse_ax: Axes = axes[0]
    psd_ax: Axes = axes[1]

    for name, mapper, mod_index, freq_pulse, bpsym in irig_waveforms:
        # Modulate the input symbols
        symbols = mapper(bit_array)
        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=freq_pulse,
            sps=sps,
        )

        pulse_length = freq_pulse.size / sps
        padded_pulse: NDArray[np.float64] = np.concatenate(
            (
                np.zeros(int(pulse_pad * sps)),
                freq_pulse,
                np.zeros(int(pulse_pad * sps)),
            ),
        )
        t_lim = (
            -(pulse_length / 2 + pulse_pad),
            +(pulse_length / 2 + pulse_pad),
        )
        pulse_t = np.linspace(
            *t_lim,
            num=padded_pulse.size,
        )
        pulse_ax.plot(
            pulse_t,
            padded_pulse,
            linestyle="-",
            linewidth=2,
            color=colors[name],
            label=rf"{name} $f(t)$",
        )
        pulse_ax.plot(
            pulse_t,
            np.cumsum(padded_pulse) / sps,
            linestyle="-.",
            linewidth=1,
            color=colors[name],
            label=rf"{name} $q(t)$",
        )

        psd_ax.psd(
            modulated_signal * np.sqrt(bpsym),
            NFFT=fft_size,
            Fs=sps / bpsym,
            label=name,
            color=colors[name],
            scale_by_freq=False,
        )

    # Format pulse diagram
    pulse_ax.set_title("Phase and Frequency Pulses")
    pulse_ax.set_ylabel("Amplitude")
    pulse_ax.set_xlabel("Normalized Time [$t/T_b$]")
    pulse_ax.set_ylim(-0.1, 0.7)
    pulse_ax.set_xlim([-4, 4])
    pulse_ax.xaxis.set_major_locator(MultipleLocator(2))
    pulse_ax.legend(loc="upper center", fontsize=8, ncol=3)
    pulse_ax.grid(which="both", linestyle=":")

    # Format the PSD plot
    psd_ax.set_title("Power Spectral Density")
    psd_ax.set_ylabel("Amplitude [dBc]")
    psd_ax.set_xlabel("Normalized Frequency [$T_b$ = 1]")
    psd_ax.set_ylim([-80, 0])
    psd_ax.yaxis.set_major_locator(MultipleLocator(10))
    psd_ax.set_xlim([-2, 2])
    psd_ax.legend(loc="upper center", fontsize=8, ncol=3)
    psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))
    psd_ax.grid(which="both", linestyle=":")

    fig_psd.tight_layout()
    fig_psd.savefig(Path(__file__).parent.parent / "images" / "irig106_waveform_comparison.png")
    plt.show()
