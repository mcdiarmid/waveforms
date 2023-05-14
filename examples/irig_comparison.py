import random
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from waveforms.cpm.pcmfm import (
    PCMFM_NUMER,
    PCMFM_DENOM,
    PCMFMSymbolMapper,
    freq_pulse_pcmfm,
)
from waveforms.cpm.soqpsk import (
    SOQPSK_NUMER,
    SOQPSK_DENOM,
    SOQPSKPrecoder,
    freq_pulse_soqpsk_tg,
)
from waveforms.cpm.multih import (
    MULTIH_IRIG_DENOM,
    MULTIH_IRIG_NUMER,
    MultiHSymbolMapper,
    freq_pulse_multih_irig,
)
from waveforms.cpm.modulate import cpm_modulate


DATA_HEADER = b"\x1b\x1bHello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(2000)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


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
            PCMFMSymbolMapper(),
            PCMFM_NUMER / PCMFM_DENOM,
            freq_pulse_pcmfm(sps=sps, order=6),
            1,
        ),
        (
            "SOQPSK-TG",
            SOQPSKPrecoder(),
            SOQPSK_NUMER / SOQPSK_DENOM,
            freq_pulse_soqpsk_tg(sps=sps),
            1,
        ),
        (
            "Multi-h CPM",
            MultiHSymbolMapper(),
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
        padded_pulse: NDArray[np.float64] = np.concatenate((
            np.zeros(int(pulse_pad*sps)),
            freq_pulse,
            np.zeros(int(pulse_pad*sps)),
        ))
        t_lim = (
            -(pulse_length/2 + pulse_pad),
            +(pulse_length/2 + pulse_pad)
        )
        pulse_t = np.linspace(
            *t_lim,
            num=padded_pulse.size
        )
        pulse_ax.plot(
            pulse_t,
            padded_pulse,
            linestyle='-',
            linewidth=2,
            color=colors[name],
            label=rf'{name} $f(t)$',
        )
        pulse_ax.plot(
            pulse_t,
            np.cumsum(padded_pulse) / sps,
            linestyle='-.',
            linewidth=1,
            color=colors[name],
            label=rf'{name} $q(t)$',
        )

        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps / bpsym,
            label=name,
            color=colors[name],
        )

        # normalized_time /= 2  # SOQPSK symbols are spaced at T/2

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
    psd_ax.set_ylabel('Amplitude [dBc]')
    psd_ax.set_xlabel('Normalized Frequency [$T_b$ = 1]')
    psd_ax.set_ylim([-60, 20])
    psd_ax.yaxis.set_major_locator(MultipleLocator(10))
    psd_ax.set_xlim([-2, 2])
    psd_ax.legend(loc="upper center", fontsize=8, ncol=3)
    psd_ax.xaxis.set_major_locator(MultipleLocator(0.5))
    psd_ax.grid(which="both", linestyle=":")

    fig_psd.tight_layout()
    fig_psd.savefig(Path(__file__).parent.parent / "images" / "irig106_waveform_comparison.png")
    fig_psd.show()