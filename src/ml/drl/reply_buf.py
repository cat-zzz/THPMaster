"""
@project: THPMaster
@File   : reply_buf.py
@Desc   :
@Author : gql
@Date   : 2024/7/16 16:03
"""
import numpy as np
from typing import Tuple, Any


class ArrayMemory:
    def __init__(self, size: int, card_features_shape: Tuple[int, int, int], action_seq_shape: Tuple[int, int, int],
                 env_features_shape: int):
        """
        初始化ArrayMemory类
        :param size: int 缓冲区大小
        :param card_features_shape: Tuple[int, int, int] 卡片特征的形状
        :param action_seq_shape: Tuple[int, int] 动作序列的形状
        :param env_features_shape: int 环境特征的长度
        :return: None
        """
        self.index = 0  # 当前插入位置索引，也用于表示当前缓冲区大小
        self.size = size  # 缓冲区的大小
        self.full = False  # 缓冲区是否已满

        # 定义transition的数据类型
        self.Transition_dtype = np.dtype([
            ('timestep', np.int32),  # 时间步
            ('card_features', np.float32, card_features_shape),  # 状态张量
            ('action_seq', np.float32, action_seq_shape),  # 状态中的动作序列
            ('env_features', np.float32, (env_features_shape,)),  # 状态向量
            ('action', np.float32, ()),  # 动作
            ('reward', np.float32),  # 奖励
            ('terminal', np.bool_)  # 终止标志
        ])

        # 定义空的transition
        self.blank_trans = (
            0, np.zeros(card_features_shape, dtype=np.float32), np.zeros(action_seq_shape, dtype=np.float32),
            np.zeros((env_features_shape,), dtype=np.float32), 0.0, 0.0, False)
        # 初始化数据缓冲区
        self.data = np.array([self.blank_trans] * size, dtype=self.Transition_dtype)

    def append(self, sarst_data: Tuple[int, Any, Any, Any, float, float, bool]):
        """
        添加新的转换数据到缓冲区
        :param sarst_data: Tuple[int, Any, Any, Any, float, float, bool] 包含时间步、卡片特征、动作序列、环境特征、动作、奖励和终止标志的元组
        :return: None
        """
        self.data[self.index] = sarst_data  # 将新数据插入到当前索引位置
        self.index = (self.index + 1) % self.size  # 更新索引位置，如果到达缓冲区末尾则回到开头
        if self.index == 0:
            self.full = True  # 如果回到开头，表示缓冲区已满

    def get(self, data_index: int) -> np.ndarray:
        """
        根据索引获取数据
        """
        return self.data[data_index % self.size]

    def total(self) -> int:
        """
        获取缓冲区中的有效数据总数
        """
        return self.size if self.full else self.index


class Replay:
    def __init__(self, memory_capacity: int, discount: float, card_features_shape: Tuple[int, int, int],
                 action_seq_shape: Tuple[int, int, int], env_features_shape: int):
        """
        初始化Replay类，用于管理ArrayMemory，提供添加数据和采样数据功能
        :param memory_capacity: int 经验回放池的容量
        :param discount: float 折扣因子
        :param card_features_shape: Tuple[int, int, int] 卡片特征的形状
        :param action_seq_shape: Tuple[int, int] 动作序列的形状
        :param env_features_shape: int 环境特征的长度
        :return: None
        """
        self.transitions = ArrayMemory(memory_capacity, card_features_shape, action_seq_shape, env_features_shape)
        self.t = 0  # 时间步计数器
        self.n = 1  # n步折扣奖励的步数
        self.discount = discount  # 折扣因子
        self.capacity = memory_capacity  # 经验回放池的容量
        self.reward_n_step_scaling = np.array([self.discount ** i for i in range(self.n)])  # 计算n步折扣奖励的缩放因子

    def append(self, card_features: np.ndarray, action_seq: np.ndarray, env_features: np.ndarray,
               action: float, reward: float, terminal: bool):
        """
        添加新的transition数据到经验回放池
        :param card_features: np.ndarray 当前卡片特征
        :param action_seq: np.ndarray 当前动作序列
        :param env_features: np.ndarray 当前环境特征
        :param action: float 当前动作
        :param reward: float 当前奖励
        :param terminal: bool 当前是否为终止状态
        :return: None
        """
        self.transitions.append(
            (self.t, card_features, action_seq, env_features, action, reward, terminal))  # 添加转换数据
        self.t = 0 if terminal else self.t + 1  # 如果是终止状态，重置时间步计数器，否则增加计数器

    def _get_samples(self, batch_size: int, n_total: int):
        """
        获取一批样本数据
        :param batch_size: 样本批量大小
        :param n_total: 缓冲区中有效数据的总数
        :return: 样本数据，包括状态张量、状态中的动作序列、状态向量、动作、奖励、下一状态张量、下一状态中的动作序列、下一状态向量和非终止标志
        """
        if n_total < batch_size:
            raise ValueError("Not enough data to sample. Increase the memory capacity or reduce the batch size.")

        idxes = np.random.randint(0, n_total, size=batch_size)  # 生成随机索引

        # 获取抽样的转换数据
        transitions = self.transitions.get(idxes)
        state_tensors = transitions['state_tensor']
        state_action_sequences = transitions['state_action_sequence']
        state_vectors = transitions['state_vector']
        actions = transitions['action']
        rewards = transitions['reward']
        next_state_tensors = self.transitions.get((idxes + 1) % self.capacity)['state_tensor']
        next_state_action_sequences = self.transitions.get((idxes + 1) % self.capacity)['state_action_sequence']
        next_state_vectors = self.transitions.get((idxes + 1) % self.capacity)['state_vector']
        terminals = transitions['terminal']

        return (state_tensors, state_action_sequences, state_vectors, actions, rewards, next_state_tensors,
                next_state_action_sequences, next_state_vectors, terminals)

    def sample(self, batch_size: int):
        """
        采样一批数据(对外的接口)
        :param batch_size: int 采样数据的数量
        :return:
        """
        n_total = self.transitions.total()  # 获取有效数据总数
        return self._get_samples(batch_size, n_total)  # 返回样本数据


def test():
    # 示例使用
    memory_capacity = 10000
    discount = 0.99
    state_tensor_shape = (5, 4, 13)
    action_sequence_shape = (8, 3, 13)
    vector_length = 10

    replay_memory = Replay(memory_capacity, discount, state_tensor_shape, action_sequence_shape, vector_length)

    for _ in range(100):
        # 插入一个示例转换
        state_tensor = np.random.rand(*state_tensor_shape)
        state_action_sequence = np.random.rand(*action_sequence_shape)
        state_vector = np.random.rand(vector_length)
        action = np.random.rand()
        reward = 1.0
        terminal = True

        replay_memory.append(state_tensor, state_action_sequence, state_vector, action, reward, terminal)

    # 抽样一批数据
    batch_size = 32
    samples = replay_memory.sample(batch_size)

    print(samples)


if __name__ == '__main__':
    test()
