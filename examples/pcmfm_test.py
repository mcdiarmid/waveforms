import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from scipy.signal import besselap, impulse

from waveforms.cpm.pcmfm import (
    PCMFM_NUMER,
    PCMFM_DENOM,
    PCMFMSymbolMapper,
    freq_pulse_pcmfm,
)
from waveforms.cpm.helpers import normalize_cpm_filter
from waveforms.cpm.modulate import cpm_modulate


DATA_HEADER = b"\x1b\x1bHello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(2000)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


if __name__ == "__main__":
    sps = 20
    fft_size = 2**10
    length = 3
    tau = np.linspace(0, length, num=length*sps)
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    irig_waveforms = [
        (
            "PCM-FM",
            PCMFMSymbolMapper(),
            PCMFM_NUMER / PCMFM_DENOM,
            freq_pulse_pcmfm(sps=sps, order=4),
        ),
    ]
    mapper = PCMFMSymbolMapper()
    symbols = mapper(bit_array)
    mod_index = PCMFM_NUMER / PCMFM_DENOM

    fig, axes = plt.subplots(2, figsize=(6, 8), dpi=100)

    pulse_ax: Axes = axes[0]
    psd_ax: Axes = axes[1]


    for order, ls, lw in zip(
        [4, 5, 6, 7, 8],
        ['-', '--', '-', '--', '-.'],
        [3, 3, 2, 2, 2]
    ):
        t, y = impulse(
            besselap(order, norm="mag"),
            T=np.linspace(0, length*2/0.7, num=(length-1)*sps+1)
        )
        freq_pulse = normalize_cpm_filter(sps, np.convolve(y, np.ones(sps)))

        normalized_time, modulated_signal = cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=freq_pulse,
            sps=sps,
        )
        pulse_ax.plot(
            tau,
            freq_pulse,
            color='k',
            linestyle=ls,
            linewidth=lw,
            label=f"{order}-th order",
        )
        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
            label=f"{order}-th order",
        )

    psd_ax.psd(
        cpm_modulate(
            symbols=symbols,
            mod_index=mod_index,
            pulse_filter=normalize_cpm_filter(sps, np.ones(sps)),
            sps=sps
        )[1],
        NFFT=fft_size,
        Fs=sps,
        label=f"No Bessel Filter",
        color='k',
        linestyle=':'
    )
    # Format frequency pulse plot
    pulse_ax.set_title("Impulse Response of $N$-th Order NRZ-Bessel Filter")
    pulse_ax.grid()
    pulse_ax.set_xlabel("$t/T_b$")
    pulse_ax.set_ylabel("Frequency Pulse $f(t)$")
    pulse_ax.set_ylim([-0.1, 0.6])
    pulse_ax.set_xlim([0, 3])
    pulse_ax.legend(loc="upper right")

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

    fig.tight_layout()
    fig.show()
    fig.savefig(Path(__file__).parent.parent / "images" / "pcmfm_bessel_comparison.png")
