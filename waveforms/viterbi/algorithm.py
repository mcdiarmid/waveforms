from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from waveforms.viterbi.trellis import FiniteStateMachine, SOQPSKTrellis4x2


j = complex(0,1)


class SOQPSKTrellisDetector:
    def __init__(self, length: int = 2) -> None:
        self.i = 0
        self.length = length
        self.cumulative_metrics = np.zeros(4, dtype=np.float64)
        self.fsm = FiniteStateMachine(trellis=SOQPSKTrellis4x2)
        self.state_exp_term = [+j, -1, +1, -j]
        self.bi_history= np.zeros((self.fsm.branches_per_column, length), dtype=np.float64)

    def va_iteration(
        self,
        mf_outputs: NDArray[np.complex128]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.int8]]:
        """

        :param mf_outputs: One row per symbol, assuming column length already sliced correctly
        :return: Recovered bits and recovered symbols
        """
        # Latest Branch increments
        self.bi_history: NDArray[np.float64] = np.roll(self.bi_history, -1, axis=1)
        self.bi_history[:, -1] = [
            np.abs(np.real(
                self.state_exp_term[branch.start] *
                mf_outputs[self.fsm.symbol_idx_map[branch.out]]
            ))
            for branch in self.fsm.trellis.branches[self.i%self.fsm.columns]
        ]
        n_transitions = self.length
        branch_metric = np.zeros((self.fsm.states, self.length), dtype=np.float64)
        branch_metric[:, -1] = self.cumulative_metrics
        path = np.zeros((self.fsm.states, n_transitions), dtype=np.uint8)

        for j in range(n_transitions):
            branches = self.fsm.trellis.branches[(self.i+j-1)%self.fsm.columns]
            for st in range(self.fsm.states):
                # Find most likely state state for ending state
                min_k = 0
                min_m = np.inf

                for branch, increment in zip(branches, self.bi_history[:, j]):
                    if branch.end != st:
                        continue
                    mm = branch_metric[branch.start, (j-1) % n_transitions] + increment
                    if mm < min_m:
                        min_m = mm
                        min_k = branch.start

                branch_metric[st, j] = min_m
                path[st, j] = min_k

            branch_metric[:, j] -= np.min(branch_metric[:, j])

        # Traceback along winning path
        recovered_symbols = np.zeros(n_transitions)
        output = np.zeros(n_transitions)
        state = np.argmin(branch_metric[:, -1])
        for j in reversed(range(n_transitions)):
            column = (self.i + j-1) % self.fsm.columns
            branch = self.fsm.reverse_transitions[column][state][path[state, j]]
            output[j] = branch.inp
            recovered_symbols[j] = branch.out
            state = path[state, j]

        self.i += 1
        self.cumulative_metrics[:] = branch_metric[:, 0]
        return output, recovered_symbols


def viterbi_algorithm(
    increments: NDArray[np.float64],
    fsm: FiniteStateMachine,
    start: Union[NDArray[np.float64], int] = None,
    column: int = 0
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

            for branch in fsm.reverse_branch_mapping[(j+column)%fsm.columns][st].values():
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
        branch = fsm.reverse_transitions[(j+column)%fsm.columns][state][path[state, j]]
        output[j] = branch.inp
        recovered_symbols[j] = branch.out
        state = path[state, j]

    return output, recovered_symbols
