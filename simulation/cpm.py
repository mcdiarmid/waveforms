import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


DATA_BUFFER = b"\x1b\x1bHello!"


def freq_pulse_soqpsk_mil(
    sps: int = 8
) -> NDArray[np.float64]:
    return np.ones(sps)/(sps*2)


def freq_pulse_soqpsk_tg(
    t_1: float = 1.5,
    t_2: float = 0.5,
    rho: float = 0.7,
    b: float = 1.25,
    sps: int = 8,
) -> NDArray[np.float64]:
    pass


class SOQPSKPrecoder:
    def __init__(self) -> None:
        self.i = 0
        self.mem = 0, 0
    
    def __call__(
        self,
        bits: NDArray[np.uint8],
    ) -> NDArray[np.int8]:
        a = np.concatenate((self.mem, bits), dtype=np.int8)
        i_arr = np.ones(bits.shape, dtype=np.int8)
        i_arr[self.i::2] = -1
        self.i = (self.i + len(bits)) % 2
        self.mem = a[-2:]
        return i_arr*(2*a[1:-1] - 1)*(a[2:] - a[:-2])


if __name__ == "__main__":
    # Bits of information to transmit
    j = complex(0, 1)
    sps = 8
    mod_index = 1/2
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # zero pad symbols to sps
    interpolated_symbols = np.zeros(sps*(symbols.size+1), dtype=np.int8)
    interpolated_symbols[sps::sps] = symbols

    # Generate pulse filter
    pulse_filter = freq_pulse_soqpsk_mil(sps=sps)  # Placeholder
    freq_pulses = np.convolve(interpolated_symbols, pulse_filter, mode="same")
    phi = 2 * np.pi * mod_index * np.cumsum(freq_pulses)
    modulated_signal = np.exp(j*phi)

    # Display things
    t = np.linspace(0, symbols.size, num=(symbols.size+1)*sps)
    # t = range(phi.size)
    f, (iq_ax, trajectory_ax) = plt.subplots(2, 1)
    # filter_ax.plot(pulse_filter)
    # filter_ax.plot(np.cumsum(pulse_filter))
    # filter_ax.set_ylim(0, 0.5)
    # filter_ax.grid(which="both", linestyle=":")

    iq_ax.plot(t, modulated_signal.real, 'b-')
    iq_ax.plot(t, modulated_signal.imag, 'r-')
    iq_ax.set_ylabel("Amplitude [V]")
    iq_ax.grid(which="both", linestyle=":")

    # trajectory_ax.plot(t, phi, 'b-.', markersize=2)
    trajectory_ax.plot(t, phi % (2*np.pi), 'ro', markersize=2)
    trajectory_ax.set_xlabel("Symbol time [t/T]")
    trajectory_ax.set_ylabel("Phase [rad]")
    trajectory_ax.grid(which="both", linestyle=":")
    
    pulse_ax = trajectory_ax.twinx()
    pulse_ax.plot(t, freq_pulses, 'g-', markersize=2)
    pulse_ax.set_ylabel("Frequency pulse")
    f.show()

    f2, eye_ax = plt.subplots(1)
    eye_ax.plot(modulated_signal.real, modulated_signal.imag, 'bo--')
    eye_ax.set_ylabel("Imaginary")
    eye_ax.set_xlabel("Real")
    eye_ax.grid(which="both", linestyle=":")
    f2.show()
