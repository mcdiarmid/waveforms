import numpy as np
from numpy.typing import NDArray


class MultiHSymbolMapper:
    def __init__(self) -> None:
        self.i = 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """Encoding process execution when the object instance is called.

        Args:
            bits (NDArray[np.uint8]): Bits to be encoded/mapped

        Returns:
            NDArray[np.int8]: Symbols
        """
        if bits.size % 2:
            msg = "Odd length bit array passed into quaternary mapper."
            raise ValueError(msg)

        self.i = (self.i + len(bits)) % 2
        return 2 * (2 * bits[self.i :: 2] + bits[(self.i + 1) % 2 :: 2]).astype(np.int8) - 3
