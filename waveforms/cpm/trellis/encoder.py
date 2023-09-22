from numpy.typing import NDArray
import numpy as np

from waveforms.cpm.trellis.model import (
    Trellis,
    SOQPSKTrellis4x2,
    forward_map
)


class TrellisEncoder:
    def __init__(self, trellis: Trellis) -> None:
        self.trellis = trellis
        self.input_cardinality = trellis.input_cardinality
        self. i = 0
        self.state = 0
        self.forward_branch_mapping = [
            [
                forward_map(st, column)
                for st in range(trellis.states)
            ]
            for column in trellis.branches
        ]

    def encode(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        """
        This assumes inputs
        """
        if bits.size % self.input_cardinality:
            raise ValueError(f"Input length must be a multiple of FSM cardinality.")

        output = np.zeros(bits.size//self.input_cardinality, dtype=np.int8)
        for n in range(bits.size//self.input_cardinality):
            idx = n*self.input_cardinality
            inp = sum([
                bits[idx+x] << (self.input_cardinality - x - 1)
                for x in range(self.input_cardinality)
            ])
            branch = self.forward_branch_mapping[self.i%self.trellis.columns][self.state][inp]  # Branch with matching start and input
            self.state = branch.end
            output[n] = branch.out
            self.i += 1

        return output

    def __call__(self, bits: NDArray[np.uint8]) -> NDArray[np.int8]:
        return self.encode(bits)
