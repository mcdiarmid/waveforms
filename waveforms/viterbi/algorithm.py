from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from waveforms.cpm.trellis.model import (
    FiniteStateMachine,
    SOQPSKTrellis4x2,
    SOQPSKTrellis4x2DiffEncoded,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray


class SOQPSKTrellisDetector:
    def __init__(
        self,
        length: int = 2,
        *,
        differantial_encoding: bool = True,
    ) -> None:
        self.i = 0
        self.length = length
        self.fsm = FiniteStateMachine(
            trellis=(SOQPSKTrellis4x2DiffEncoded if differantial_encoding else SOQPSKTrellis4x2),
        )
        self.state_exp_term = [+1j, -1, +1, -1j]
        self.bi_history = np.zeros(
            (self.fsm.branches_per_column, length),
            dtype=np.float64,
        )
        self.metrics = np.zeros(
            (self.fsm.states, self.length),
            dtype=np.float64,
        )
        self.path = np.zeros(
            (self.fsm.states, self.length),
            dtype=np.uint8,
        )

    def iteration(
        self,
        mf_outputs: NDArray[np.complex128],
    ) -> tuple[NDArray[np.uint8], NDArray[np.int8]]:
        """Performs a single iteration of the viterbi algorithm.

        Args:
            mf_outputs: One row per symbol, assuming column length already sliced correctly

        Returns:
            tuple[NDArray[np.uint8], NDArray[np.int8]]: Recovered bits, recovered symbols
        """
        # Latest Branch increments
        self.bi_history[:, :] = np.roll(self.bi_history, -1, axis=1)
        self.bi_history[:, -1] = [
            np.real(
                self.state_exp_term[branch.start] * mf_outputs[self.fsm.symbol_idx_map[branch.out]],
            )
            for branch in self.fsm.trellis.branches[self.i % self.fsm.columns]
        ]
        n_transitions = self.length
        self.metrics[:, -1] = self.metrics[:, 0] - self.metrics[:, 0].min()
        self.metrics[:, :-1] = 0
        self.path[:, :] = 0

        for j in range(n_transitions):
            # Iterating branches that lead to the end state st
            branches = self.fsm.trellis.branches[(self.i + j - 1) % self.fsm.columns]
            for st in range(self.fsm.states):
                # Find most likely state state for ending state
                min_k = 0
                min_m = np.inf

                for branch, increment in zip(branches, self.bi_history[:, j]):
                    if branch.end != st:
                        continue

                    mm = self.metrics[branch.start, (j - 1) % n_transitions] + increment
                    if mm < min_m:
                        min_m = mm
                        min_k = branch.start

                self.metrics[st, j] = min_m
                self.path[st, j] = min_k

        # Traceback along winning path
        recovered_symbols = np.zeros(n_transitions)
        output = np.zeros(n_transitions)
        state = np.argmin(self.metrics[:, -1])
        for j in reversed(range(n_transitions)):
            column = (self.i + j - 1) % self.fsm.columns
            branch = self.fsm.reverse_transitions[column][state][self.path[state, j]]
            output[j] = branch.inp
            recovered_symbols[j] = branch.out
            state = self.path[state, j]

        self.i += 1
        return output, recovered_symbols
