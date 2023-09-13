from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from waveforms.viterbi.trellis import FiniteStateMachine


def viterbi_algorithm(
    increments: NDArray[np.float64],
    fsm: FiniteStateMachine,
    start: Union[NDArray[np.float64], int] = None
) -> Tuple[NDArray[np.uint8], NDArray[np.int8]]:
    """
    Viterbi Algorithm
    
    :param increments: Branch metric increments for each hypothetical symbol
    :param fsm: Finite State Machine object
    :param start: Starting symbol weights
    :return: Array of hard decisions for the given input increments and fsm
    """
    # Initialize arrays
    _output_cardinality, n_transitions = increments.shape
    n_states = fsm.states
    increments = np.abs(increments)
    recovered_symbols = np.zeros(n_transitions)
    output = np.zeros(n_transitions)

    # Standard Viterbi Algorithm
    branch_metric = np.zeros((n_states, 2), dtype=np.float64)
    path = np.zeros((n_states, n_transitions), dtype=np.uint8)

    # Assign initial state likelihood based on previous viterbi iteration
    if isinstance(start, int):
        start = [np.inf if i != start else 0 for i in range(n_states)]

    branch_metric[:, 0] = start or np.zeros(n_states)

    # Calculate branch metrics and winning paths going forward
    for j in range(n_transitions):
        # Iterate all viable states
        for st in range(n_states):
            # Iterate branches to determine most likely prior state
            min_k = 0
            min_m = np.inf

            for branch in fsm.reverse_branch_mapping[st].values():
                symbol_index = fsm.symbols.index(branch.out)
                mm = branch_metric[branch.start, j % 2] + increments[symbol_index, j]

                if mm < min_m:
                    min_m = mm
                    min_k = branch.start

            branch_metric[st, (j + 1) % 2] = min_m
            path[st, j] = min_k

        branch_metric[:, (j + 1) % 2] -= np.min(branch_metric[:, (j + 1) % 2])

    # Traceback along winning path
    state = np.argmin(branch_metric[:, n_transitions % 2])
    for j in reversed(range(n_transitions)):
        branch = fsm.reverse_transitions[state][path[state, j]]
        output[j] = branch.inp
        recovered_symbols[j] = branch.out
        state = path[state, j]

    return output, recovered_symbols
