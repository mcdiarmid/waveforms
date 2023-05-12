import numpy as np
from numpy.typing import NDArray


MULTIH_IRIG_NUMER = np.array([4, 5])
MULTIH_IRIG_DENOM = 16


class MultiHSymbolMapper:
    def __init__(self) -> None:
        self.i = 0
    
    def __call__(
        self,
        bits: NDArray[np.uint8],
    ) -> NDArray[np.int8]:
        if bits.size % 2:
            raise ValueError("Odd length bit array passed into quaternary mapper.")
        self.i = (self.i + len(bits)) % 2
        return 2*(2*bits[self.i::2] + bits[(self.i+1)%2::2]).astype(np.int8) - 3


def freq_pulse_multih_irig(
    sps: int = 8,
    length: float = 3
) -> NDArray[np.float64]:
    t_norm = np.linspace(0, length, num=length*sps)
    g = (1 - np.cos(2 * np.pi * t_norm / length)) / (2 * length)
    a_scalar = sps / (np.cumsum(g)[-1] * 2)
    return a_scalar * g
