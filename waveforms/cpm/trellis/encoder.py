import numpy as np
from numpy.typing import NDArray

from waveforms.cpm.trellis.model import Trellis, forward_map


class TrellisEncoder:
    def __init__(self, trellis: Trellis) -> None:
        self.trellis = trellis
        self.input_cardinality = trellis.input_cardinality
        self.i = 0
        self.state = 0
        self.forward_branch_mapping = [
            [forward_map(st, column) for st in range(trellis.states)] for column in trellis.branches
        ]

    def encode(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """Encodes a sequence of input bits to symbols.

        This assumes inputs are either 0 or 1.

        Args:
            bits (NDArray[np.uint8]): Input bit sequence

        Returns:
            NDArray[np.int8]: Symbols
        """
        if bits.size % self.input_cardinality:
            msg = "Input length must be a multiple of FSM cardinality."
            raise ValueError(msg)

        output = np.zeros(bits.size // self.input_cardinality, dtype=np.int8)
        for n in range(bits.size // self.input_cardinality):
            idx = n * self.input_cardinality
            inp = sum(
                [
                    bits[idx + x] << (self.input_cardinality - x - 1)
                    for x in range(self.input_cardinality)
                ],
            )
            branch = self.forward_branch_mapping[self.i % self.trellis.columns][self.state][
                inp
            ]  # Branch with matching start and input
            self.state = branch.end
            output[n] = branch.out
            self.i += 1

        return output

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """Invokes the encode method when the instance is called.

        Args:
            bits (NDArray[np.uint8]): Input bit sequence

        Returns:
            NDArray[np.int8]: Symbols
        """
        return self.encode(bits)
