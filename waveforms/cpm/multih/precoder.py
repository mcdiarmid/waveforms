import numpy as np
from numpy.typing import NDArray


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
