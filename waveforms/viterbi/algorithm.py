from typing import Tuple

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
) -> Tuple[NDArray[np.uint8], NDArray[np.int8]]:
    """
    Viterbi Algorithm
    
    :param increments: Branch metric increments for each hypothetical symbol
    :param fsm: Finite State Machine object
    :param start: Starting symbol weights
    :return: Array of hard decisions for the given input increments and fsm
    """
    # Initialize arrays
    O,N = increments.shape
    S = fsm.states
    recovered_symbols = np.zeros(N)
    output = np.zeros(N)

    # Standard Viterbi Algorithm
    branch_metric = np.zeros((S, 2), dtype=np.float64)
    path = np.zeros((S, N), dtype=np.uint8)

    # Assign initial state likelihood based on previous viterbi iteration
    branch_metric[:,0] = start or np.zeros(S)

    for j in range(N):
        # Iterate all viable states
        for i in range(S):
            # Iterate branches to determine most likely prior state
            min_k = 0
            min_m = np.inf
            for branch in fsm.reverse_branch_mapping[i].values():
                mm = branch_metric[branch.start,j%2] + increments[fsm.symbols.index(branch.out)][j]
                if mm < min_m:
                    min_m = mm
                    min_k = branch.start

            branch_metric[i,(j+1)%2] = min_m
            path[i,j] = min_k

        branch_metric[:,(j+1)%2] -= np.min(branch_metric[:,(j+1)%2])

    # Traceback
    state = np.argmin(branch_metric[:, -1])
    for j in reversed(range(N)):
        branch = fsm.reverse_transitions[state][path[state,j]]
        output[j] = branch.inp
        recovered_symbols[j] = branch.out
        state = path[state, j]

    return output, recovered_symbols
