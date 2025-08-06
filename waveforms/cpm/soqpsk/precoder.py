import numpy as np
from numpy.typing import NDArray


class SOQPSKDifferentialEncoder:
    def __init__(self) -> None:
        self.out_mem = 0, 0
        self.in_mem = 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """SOQPSK Differential Encoder.

        From IRIG 106 Chapter 2 2-22:
        out[2i] = in[2i] ^ out[2i-1] ^ 1
        out[2i+1] = in[2i+1] ^ out[2i]

        Simplified to either:
        out[i] = in[i] ^ in[i-1] ^ out[i-2] ^ 1

        or:
        out[i] = (in[i] + in[i-1] + 1 - out[i-2]) & 1

        Args:
            bits (NDArray[np.uint8]): Input bit sequence.

        Returns:
            NDArray[np.int8]: Output bits.

        """
        out = np.zeros(bits.size + 2, dtype=np.int8)
        out[:2] = self.out_mem
        inp = np.concatenate(([self.in_mem], bits), dtype=np.int8)
        for i in range(2, bits.size + 1):
            out[i] = inp[i - 1] ^ inp[i - 2] ^ out[i - 2] ^ 1
        self.in_mem = bits[-1]
        self.out_mem = out[-2:]
        return out[2:]


class SOQPSKStandardPrecoder:
    def __init__(self) -> None:
        self.i = 1
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
        return i_arr * (2 * a[1:-1] - 1) * (a[:-2] - a[2:]) * 2


class SOQPSKRecursivePrecoder:
    def __init__(self) -> None:
        self.i = 0
        self.sign = 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """Encodes bits to symbols when instance is called.

        Args:
            bits (NDArray[np.uint8]): Input bit sequence.

        Returns:
            NDArray[np.int8]: Output symbols.

        """
        out = np.zeros_like(bits, dtype=np.int8)
        for i, bit in enumerate(bits):
            out[i] = (-1) ** self.sign * bit * 2
            self.sign = (self.sign + out[i] / 2 + 1) % 2

        self.i = (self.i + len(bits)) % 2
        return out
