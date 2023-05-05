import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


DATA_BUFFER = b"Test CPM Payload!"


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
    bit_array = np.unpackbits(np.frombuffer(DATA_BUFFER, dtype=np.uint8))

    # Convert bits to symbols
    symbol_precoder = SOQPSKPrecoder()
    symbols = symbol_precoder(bit_array)

    # Generate pulse filter
    pulse_filter = np.ones(1)  # Placeholder
