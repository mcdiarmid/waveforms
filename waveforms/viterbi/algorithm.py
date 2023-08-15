import numpy as np
from numpy.typing import NDArray

from waveforms.viterbi.trellis import (
    FiniteStateMachine,
    Trellis,
    Branch,
)

def viterbi_algorithm(
    increments: NDArray[np.float64],
    fsm: FiniteStateMachine,
    start: NDArray[np.float64] = None
) -> NDArray[np.uint8]:
    """
    Viterbi Algorithm
    
    :param increments: Branch metric increments for each hypothetical symbol
    :param fsm: Finite State Machine object
    :param start: Starting symbol weights
    :return: Array of hard decisions for the given input increments and fsm
    """
    # Initialize arrays
    N, O = increments.shape
    S = len(fsm.trellis.branches)
    output = np.zeros(N)

    # Standard Viterbi Algorithm
    T_1 = np.zeros((S, N))
    T_2 = np.zeros((S, N))

    # Assign initial state likelihood based on previous viterbi iteration
    T_1[:,0] = start or np.zeros(S)

    for j in range(2, N):
        for i in range(S):
            min_k = 0

            # Iterate branches and input to determine most likely k
            for k in range(S):
                # TODO figure out min_k
                pass

            T_1[i,j] = T_1[min_k,j-1]
            T_2[i,j] = min_k

    # TODO will need to modify FSM class for convenient use here
    z = np.argmin(T_1[:, -1])
    for j in reversed(range(N)):
        output[j] = ...
        z = T_2[z, j]

    return output
