import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


DATA_BUFFER = b"\x1b\x1bHello!"


def freq_pulse_soqpsk_tg(
    T_1: float = 1.5,
    T_2: float = 0.5,
    rho: float = 0.7,
    b: float = 1.25,
    sps: int = 8,
) -> NDArray[np.float64]:
    # Calculate w.r.t. normalized time
    tau_max = (T_1 + T_2)*2
    t_norm = np.linspace(-tau_max, tau_max, num=int(tau_max*sps*2))
    # Frequency pulse
    g = np.cos(np.pi*rho*b*t_norm/2)/(1-np.power(rho*b*t_norm, 2)) * np.sinc(b*t_norm/2)
    # Windowing function
    if T_2 > 0:
        w = (1+np.cos(np.pi*(t_norm/2-T_1)/T_2))/2
    else:
        w = np.ones(t_norm.shape)
    w[np.where(np.abs(t_norm) < 2*T_1)] = 1
    w[np.where(np.abs(t_norm) > 2*(T_1+T_2))] = 0
    # Calculate A that yeilds frequency pulse with integral=1/2
    a_scalar = sps / (np.cumsum(g*w)[-1] * 2)
    # Scaled & windowed frequency pulse
    return a_scalar * g * w


def freq_pulse_soqpsk_mil(
    sps: int = 8
) -> NDArray[np.float64]:
    # return np.ones(sps)/2
    return freq_pulse_soqpsk_tg(T_1=0.25, T_2=0, b=0, rho=0, sps=sps)


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
    symbol_rate = 1e3
    sps = 20
    mod_index = 1/2
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)
    t = np.linspace(0, symbols.size+1, num=(symbols.size+1)*sps)

    # zero pad symbols to sps
    interpolated_symbols = np.zeros(sps*(symbols.size+1), dtype=np.int8)
    interpolated_symbols[sps::sps] = symbols

    # Generate pulse filter
    fig_eye, eye_ax = plt.subplots(1)
    eye_ax.set_ylabel("Imaginary")
    eye_ax.set_xlabel("Real")

    # Set-up some plots
    fig_time, (symbol_ax, trajectory_ax, iq_ax) = plt.subplots(3, 1)
    symbol_ax.set_ylabel("Symbols")
    symbol_ax.stem(range(symbols.size+2), [0, *symbols, 0], linefmt='ko')

    iq_ax.set_ylabel("Amplitude [V]")
    trajectory_ax.set_xlabel("Symbol time [t/T]")
    trajectory_ax.set_ylabel("Phase [rad]")

    # Generate and plot SOQPSK-MIL and SOQPSK-TG modulated data
    for pulse_filter_fn in (freq_pulse_soqpsk_mil, freq_pulse_soqpsk_tg):
        pulse_filter = pulse_filter_fn(sps=sps)
        freq_pulses = np.convolve(interpolated_symbols, pulse_filter, mode="same")
        phi = 2 * np.pi * mod_index * np.cumsum(freq_pulses) / sps
        modulated_signal = np.exp(j*phi)

        # Display things
        iq_ax.plot(t, modulated_signal.real, linestyle='-')
        iq_ax.plot(t, modulated_signal.imag, linestyle='-')
        symbol_ax.plot(t, freq_pulses, linestyle='-')
        trajectory_ax.plot(t, phi % (2*np.pi), marker='o', markersize=2)
        trajectory_ax.plot(t, phi, markersize=2)
        eye_ax.plot(modulated_signal.real, modulated_signal.imag, marker='o', linestyle='--')

    # Show plots
    symbol_ax.grid(which="both", linestyle=":")
    iq_ax.grid(which="both", linestyle=":")
    trajectory_ax.grid(which="both", linestyle=":")
    fig_time.show()

    eye_ax.grid(which="both", linestyle=":")
    fig_eye.show()
