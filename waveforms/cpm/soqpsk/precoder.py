import numpy as np
from numpy.typing import NDArray


class RecursiveDE:
    def __init__(self) -> None:
        self.memory = 0, 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Recursive diff encoder."""
        out = np.zeros(len(bits) + 2, dtype=np.uint8)
        out[:2] = self.memory
        for i in range(2, len(bits) + 1):
            out[i] = bits[i - 2] ^ out[i - 2]
        self.memory = out[-2:]
        return out[2:]


class RecursiveDD:
    def __init__(self) -> None:
        self.memory = 1, 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Recursive diff encoder."""
        out = np.zeros(len(bits), dtype=np.uint8)
        inp = np.concatenate((self.memory, bits), dtype=np.int8)
        for i in range(len(bits)):
            out[i] = inp[i + 2] ^ inp[i]
        self.memory = inp[-2:]
        return out


class SOQPSKDifferentialEncoder:
    def __init__(self) -> None:
        self.out_mem = 1, 1
        self.in_mem = 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """SOQPSK Differential Encoder.

        From IRIG 106 Chapter 2 2-22:
        𝛿̅[2i] = b[2i] ⊕ 𝛿̅[2i-1] ⊕ 1
        𝛿̅[2i+1] = b[2i+1] ⊕ 𝛿̅[2i]

        Simplified:
        𝛿̅[i] = b[i] ⊕ b[i-1] ⊕ 𝛿̅[i-2] ⊕ 1

        Args:
            bits (NDArray[np.uint8]): Input bit sequence.

        Returns:
            NDArray[np.unt8]: Output bits.

        """
        out = np.zeros(len(bits) + 2, dtype=np.int8)
        out[:2] = self.out_mem
        inp = np.concatenate(([self.in_mem], bits), dtype=np.int8)
        for i in range(2, len(bits) + 1):
            out[i] = inp[i - 1] ^ inp[i - 2] ^ out[i - 2] ^ 1
        self.in_mem = bits[-1]
        self.out_mem = out[-2:]
        return out[2:]


class SOQPSKDifferentialDecoder:
    def __init__(self) -> None:
        self.out_mem = 1
        self.in_mem = 0, 0

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """SOQPSK Differential Decoder.

        From IRIG 106 Chapter 2 2-23:
        b[2i] = δ[2i] ⊕ 𝛿̅[2i-1] ⊕ 1
        b[2i+1] = δ[2i+1] ⊕ 𝛿̅[2i]

        Simplified:
        b[i] = b[i-1] ⊕ 𝛿[i] ⊕ 𝛿̅[i-2] ⊕ 1

        Args:
            bits (NDArray[np.uint8]): Input bit sequence.

        Returns:
            NDArray[np.int8]: Output bits.

        """
        out = np.zeros(len(bits) + 1, dtype=np.uint8)
        out[0] = self.out_mem
        inp = np.concatenate((self.in_mem, bits), dtype=np.int8)
        for i in range(1, len(bits) + 1):
            out[i] = out[i - 1] ^ inp[i + 1] ^ inp[i - 1] ^ 1
        self.in_mem = bits[-2:]
        self.out_mem = out[-1]
        return out[1:]


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
