import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from waveforms.cpm.pcmfm import freq_pulse_pcmfm
from waveforms.cpm.pcmfm.bessel import bessel, _bessel_b, BESSEL_PARAMS_LUT
from waveforms.cpm.helpers import normalize_cpm_filter


if __name__ == "__main__":
    fig, ax = plt.subplots(1)
    sps = 20
    length = 3
    t = np.linspace(0, length, num=length*sps)
    freq_pulse_nrz = np.ones(sps)/(sps)
    for order in [4, 5, 6, 7, 8]:
        # bessel_filter = normalize_cpm_filter(sps, bessel(sps=sps, order=order))
        normalized_filter = normalize_cpm_filter(sps, bessel(sps=sps, order=order))
        ax.plot(
            # t,
            freq_pulse_pcmfm(sps, order),
            # bessel(sps=sps, order=order),
            # np.cumsum(np.convolve(freq_pulse_nrz, normalized_filter))/sps,
            label=f"{order}-th order"
        )

    ax.grid()
    ax.set_xlabel("$t/T_b$")
    ax.set_ylabel("Frequency Pulse $f(t)$")
    ax.xaxis.set_major_formatter(lambda x,*_: f'{x/sps:.1f}')
    ax.legend(loc="upper right")
    fig.show()
