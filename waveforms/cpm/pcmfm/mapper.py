import numpy as np
from numpy.typing import NDArray


class PCMFMSymbolMapper:    
    def __call__(
        self,
        bits: NDArray[np.uint8],
    ) -> NDArray[np.int8]:
        return 2 * bits.astype(np.int8) - 1
