"""
@project: THPMaster
@File   : memory.py
@Desc   :
@Author : gql
@Date   : 2024/7/15 21:03
"""
import numpy as np

Transition_dtype = np.dtype([('timestep', np.int32),
                             ('state', np.int32, (100, 100)),
                             ('action', np.int32, (3, 13)),
                             ('reward', np.float32),
                             ('terminal', np.bool_)])

blank_trans = (0, np.zeros((84, 84), dtype=np.int32), 0, 0.0, False)


class ArrayMemory:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)

    def append(self, sarst_data):
        self.data[self.index] = sarst_data
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True

    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        if self.full:
            return self.size
        else:
            return self.index


class Replay:
    def __init__(self, args):
        self.transitions = ArrayMemory(args.memory_capacity)
        self.t = 0  # 时间步计数器
        self.n = 1
        self.history_length = args.history_length
        self.discount = args.discount
        self.capacity = args.memory_capacity
        self.reward_n_step_scaling = np.array([self.discount ** i for i in range(self.n)])

    def append(self, fram_data, action, reward, terminal):
        self.transitions.append((self.t, fram_data, action, reward, not terminal))
        if terminal:
            self.t = 0
        else:
            self.t += 1

    def _get_transitions(self, idx):
        transition_idx = np.arange(-self.history_length + 1, self.n + 1) + np.expand_dims(idx, axis=1)
        transitions = self.transitions.get(transition_idx)
        transitions_first = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_first, dtype=np.bool_)
        for t in range(self.history_length - 2, -1, -1):
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1], transitions_first[:, t + 1])

        for t in range(self.history_length, self.history_length + self.n):
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1], transitions_first[:, t])
        transitions[blank_mask] = blank_trans
        return transitions

    def _get_samples(self, batch_size, n_total):
        idxs = []
        while len(idxs) < batch_size:
            idx = np.random.randint(0, n_total - 1)  # 均匀采样
            if (self.transitions.index - idx) % self.capacity >= self.n and \
                    (idx - self.transitions.index) % self.capacity >= self.history_length - 1:
                idxs.append(idx)
        # 检索所有需要转换的数据（from t-h to t+n）
        transitions = self._get_transitions(idxs)
        # 创建未分散的状态和第n个下一状态
        all_states = transitions['state']
        states = all_states[:, :self.history_length]
        next_states = all_states[:, self.n:self.n + self.history_length]
        # Discrete actions to be used as index
        actions = transitions['action'][:, self.history_length - 1]
        # Calculate truncated n-step discounted returns
        rewards = transitions['reward'][:, self.history_length - 1: -1]
        ret = np.matmul(rewards, self.reward_n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminals = transitions['terminal'][:, self.history_length + self.n - 1]
        return states, actions, ret, next_states, nonterminals

    def sample(self, batch_size):
        n_total = self.transitions.total()
        states, actions, returns, next_states, nonterminals = self._get_samples(batch_size, n_total)
        # (np.uint8, (84, 84)), np.int32, np.float32, (np.uint8, (84, 84)), np.uint8
        # s,a,r,s_next,non_terminal
        return np.asarray(states, np.uint8), \
            np.asarray(actions, np.int32), \
            np.asarray(returns, np.float32), \
            np.asarray(next_states, np.uint8), \
            np.asarray(nonterminals, np.uint8)


if __name__ == '__main__':
    pass
