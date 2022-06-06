import numpy as np


class ReplayBuffer:
    """Base class for replay buffers"""

    def __init__(self, max_size: int):
        self._max_size = max_size

    def add(self, item: object):
        raise NotImplementedError("Abstract class ReplayBuffer is used")

    def sample_batch(self, batch_size: int):
        raise NotImplementedError("Abstract class ReplayBuffer is used")


class NLineBuffer:
    """Buffer, which supports several lines of items
    Lines are forced to have the same length
    """

    def __init__(self, n_lines: int):
        self._n_lines = n_lines
        self.buffer = [[] for i in range(n_lines)]

    def __len__(self):
        return len(self.buffer[0])

    def append(self, nline_item):
        assert (
            len(nline_item) == self._n_lines
        ), "Item length should be equal to the buffer n lines"
        for idx, single_item in enumerate(nline_item):
            self.buffer[idx].append(single_item)

    def pop_0th(self):
        for idx in range(self._n_lines):
            self.buffer[idx].pop(0)

    def sample_batch(self, batch_size: int):
        buffer_size = len(self)
        sample_indeces = np.random.choice(
            list(range(buffer_size)),
            size=batch_size,
            replace=False,
        )
        nline_batch = []
        for line in self.buffer:
            nline_batch.append(np.array(line)[sample_indeces])
        return nline_batch


class SARSReplayBuffer(ReplayBuffer):
    def __init__(self, max_size: int = 200):
        super().__init__(max_size)
        self._buffer = NLineBuffer(n_lines=4)

    def add(self, item):
        """Adds tuple (state, action, reward, new_state) to buffer"""
        self._buffer.append(item)
        if len(self._buffer) > self._max_size:
            self._buffer.pop_0th()

    def sample_batch(self, batch_size=32):
        buffer_sample = self._buffer.sample_batch(batch_size)
        buffer_sample_numpied = list(map(np.array, buffer_sample))
        states, actions, rewards, new_states = buffer_sample_numpied
        return states, actions, rewards, new_states
