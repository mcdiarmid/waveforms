import numpy as np
from numpy.typing import NDArray


class SOQPSKPrecoder:
    def __init__(self) -> None:
        self.i = 0
        self.mem = 0, 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """Encodes bits to symbols when instance is called.

        Args:
            bits (NDArray[np.uint8]): Input bit sequence.

        Returns:
            NDArray[np.int8]: Output symbols.
        """
        a = np.concatenate((self.mem, bits), dtype=np.int8)
        i_arr = np.ones(bits.shape, dtype=np.int8)
        i_arr[self.i :: 2] = -1
        self.i = (self.i + len(bits)) % 2
        self.mem = a[-2:]
        return i_arr * (2 * a[1:-1] - 1) * (a[:-2] - a[2:])
