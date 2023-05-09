import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


DATA_HEADER = b"\x1b\x1bHello World!"
DATA_EXTRA = bytes([random.randint(0,0xff) for i in range(250)])
DATA_BUFFER = DATA_HEADER + DATA_EXTRA
j = complex(0, 1)


def freq_pulse_soqpsk_tg(
    T_1: float = 1.5,
    T_2: float = 0.5,
    rho: float = 0.7,
    b: float = 1.25,
    sps: int = 8,
) -> NDArray[np.float64]:
    # Calculate w.r.t. normalized time
    tau_max = (T_1 + T_2)*2
    t_norm = np.linspace(-tau_max, tau_max, num=int(tau_max*sps*2), dtype=np.float64)
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


def freq_pulse_soqpsk_a(
    sps: int = 8
) -> NDArray[np.float64]:
    return freq_pulse_soqpsk_tg(b=1.35, T_1=1.4, T_2=0.6, rho=1, sps=sps)


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
    sps = 10
    fft_size = 2**11
    mod_index = 1/2
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    points = (symbols.size+1)*sps+1
    t = np.linspace(0, symbols.size+1, num=points, dtype=np.float64)

    # zero pad symbols to sps
    interpolated_symbols = np.zeros(points, dtype=np.int8)
    interpolated_symbols[sps:-1:sps] = symbols

    # Create plots and axes
    fig_eye, eye_const_axes = plt.subplots(2, 2)
    eye_real_ax: Axes = eye_const_axes[0, 0]
    eye_imag_ax: Axes = eye_const_axes[1, 0]
    constel_ax: Axes = eye_const_axes[0, 1]
    psd_ax: Axes = eye_const_axes[1, 1]

    fig_time, (symbol_ax, trajectory_ax, iq_ax) = plt.subplots(3, 1)

    # Plot the ternary symbols
    symbol_ax.stem(
        t[::sps],
        [0, *symbols, 0],
        linefmt='k',
        basefmt='',
        markerfmt='k.'
    )

    # Generate and plot SOQPSK-MIL and SOQPSK-TG modulated data
    pulses_colors_labels = (
        (freq_pulse_soqpsk_tg, 'royalblue', 'SOQPSK-TG'),
        (freq_pulse_soqpsk_a, 'darkorange', 'SOQPSK-A'),
        (freq_pulse_soqpsk_mil, 'crimson', 'SOQPSK-MIL'),
    )

    for pulse_filter_fn, color, label in pulses_colors_labels:
        # Modulate the input symbols
        pulse_filter = pulse_filter_fn(sps=sps)
        freq_pulses = np.convolve(interpolated_symbols, pulse_filter, mode="same")
        phi = 2 * np.pi * mod_index * np.cumsum(freq_pulses) / sps + np.pi / 4
        modulated_signal = np.exp(j*phi)

        # Display things
        iq_ax.plot(t, modulated_signal.real, linestyle='-.', color=color)
        iq_ax.plot(t, modulated_signal.imag, linestyle=':', color=color)
        symbol_ax.plot(t, freq_pulses, linestyle='-', color=color)

        modulo = 16
        for i in range((points-1)//(sps*modulo)):
            idx_start = i*sps*modulo
            idx_end = (i+1)*sps*modulo
            trajectory_ax.plot(
                t[idx_start:idx_end] % modulo,
                phi[idx_start:idx_end] % (2*np.pi),
                linestyle='-',
                linewidth=0.2,
                # marker='.',
                # markersize=1,
                color=color,
            )
        # trajectory_ax.plot(t, phi, markersize=2)
        modulo = 4
        q_symbol_offset = 0.5
        for i in range((points-1)//(sps*modulo)):
            idx_start_i = i*sps*modulo
            idx_start_q = idx_start_i + int(sps*q_symbol_offset)
            eye_real_ax.plot(
                (t[idx_start_i:idx_start_i+sps*modulo]+0.0)%modulo,
                modulated_signal.real[idx_start_i:idx_start_i+sps*modulo],
                linewidth=0.3,
                color=color,
            )
            eye_imag_ax.plot(
                (t[idx_start_q:idx_start_q+sps*modulo]-q_symbol_offset)%modulo,
                modulated_signal.imag[idx_start_q:idx_start_q+sps*modulo],
                linewidth=0.3,
                color=color,
            )
        constel_ax.plot(
            modulated_signal.real,
            modulated_signal.imag,
            color=color,
        )
        psd_ax.psd(
            modulated_signal,
            NFFT=fft_size,
            Fs=sps,
        )

    # Format plots
    iq_ax.set_ylabel("Amplitude [V]")
    iq_ax.set_xlabel("Symbol time [t/T]")
    iq_ax.grid(which="both", linestyle=":")
    iq_ax.set_xlim([0, len(DATA_HEADER)*8+1])

    trajectory_ax.set_ylabel("Phase [rad]")
    trajectory_ax.set_yticks([(2*i+1)*np.pi/4 for i in range(4)])
    trajectory_ax.set_yticklabels([rf"$\frac{{{2*i+1}\pi}}{{4}}$" for i in range(4)])
    trajectory_ax.grid(which="both", linestyle=":")

    symbol_ax.set_ylabel("Symbols")
    symbol_ax.set_xlim([-1, len(DATA_HEADER)*8+2])
    symbol_ax.grid(which="both", linestyle=":")

    fig_time.show()

    for ax_row in eye_const_axes:
        for ax in ax_row:
            ax.grid(which="both", linestyle=":")
    
    eye_real_ax.set_ylabel("In-phase")
    eye_imag_ax.set_ylabel("Quadrature")
    eye_imag_ax.set_xlabel("Symbol Time [t/T]")

    constel_ax.set_xlabel("In-phase")
    constel_ax.set_ylabel("Quadrature")

    psd_ax.set_ylabel('Amplitude [dBc]')
    psd_ax.set_xlabel('Frequency [bit rates]')
    psd_ax.set_ylim([-100, 20])
    psd_ax.set_yticks([(i-10)*10 for i in range(12)])
    psd_ax.set_xlim([-2.5, 2.5])
    fig_eye.show()
