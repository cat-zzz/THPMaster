"""
@project: machine_game
@File   : nolimitholdem2.py
@Desc   :
@Author : gql
@Date   : 2024/3/29 9:10
"""
from abc import ABC, abstractmethod

import numpy as np

from src.env.game import NoLimitHoldemGame

default_env_config = {
    'num_players': 2,
    # 以下是game.py下的配置参数
    # 'seed': 0,  # 随机数种子，为0表示随机
    # 'show_all_hands': True,  # 默认一旦进行比牌，则向所有玩家都展示手牌
    # 'one_game_reset': True,  # todo 筹码默认为一局一复位（未实现）
    # 'store_all_game_history': False,  # game_history是否保存全部对局的历史序列，还是只保存一局的序列
}


class BasePokerEnv(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass


'''
动作编码
[
  [action],
  [position],
  [pot],
]
[action]: 
'''


class NoLimitHoldemEnv(BasePokerEnv):
    def __init__(self, env_config=None):
        super().__init__()
        if env_config is None:
            env_config = default_env_config
        self.name = 'No Limit Holdem'
        self.default_game_config = env_config
        self.game = NoLimitHoldemGame()
        self.action_space = Action
        self.obs_shape = 56 + len(Action)  # 状态空间维度大小，此处是一维数组，长度为56 + len(Action)
        self.num_actions = len(Action)

    def reset(self):
        self.game.reset()
        obs = self.get_obs(0)
        return obs

    def step(self, action: Action):
        # 检查动作是否合法
        legal_actions = self.get_legal_actions()
        player_idx = self.game.cur_player_idx
        is_legal = True
        if action not in legal_actions:
            # todo game_config可配置参数：动作不合法时，用fold代替此动作，还是重新输入动作
            print('illegal action:', action)
            action = Action.FOLD
            is_legal = False
        raw_action = convert_to_raw_action(action)
        state, action_is_legal, game_state_flag, down, info = self.game.step(raw_action)
        obs = self.get_obs(player_idx)
        reward = self.get_reward(player_idx, action_is_legal, down, info)
        # if not is_legal:
        #     reward = int(-constants.total_chip / 2)
        # elif down:
        #     reward = int(info['payoffs'][player_idx])
        # else:
        #     reward = 0
        return obs, reward, down, info

    def get_legal_actions(self):
        """
        获取当前合法动作（抽象后）
        :return:
        """
        raw_legal_actions = self.game.get_legal_actions()  # 原始合法动作
        legal_actions = []
        # 处理check, call, fold, allin
        for i, action in enumerate(raw_legal_actions[:-1]):
            if action == 1:
                legal_actions.append(Action(i))
        # 处理raise
        # 分阶段的固定筹码量间隔
        legal_actions += [Action(i + 4) for i, x in enumerate(action_mapping) if x >= raw_legal_actions[-1]]
        # one_hot_legal_actions = to_one_hot(legal_actions, len(Action))
        return legal_actions

    def get_obs(self, player_idx):
        """
        获取某位玩家的可观测状态
        :param player_idx: 玩家索引
        :return:
        """
        '''
        [0-51]表示手牌和公共牌，不用具体区分手牌和公共牌
        [52]表示己方下注筹码量
        [53]表示己方当前阶段下注量
        [54]表示当前阶段总下注量
        [55]表示所有玩家总下注量
        [56...]表示合法动作
        牌面信息表示：[花色, 点数]，花色：0-3，点数：0-12
        '''
        hands = self.game.get_hand_cards(player_idx)
        public_cards = self.game.get_public_cards()
        cards = np.concatenate((hands, public_cards), axis=0)
        indices = cards[:, 0] * 13 + cards[:, 1]
        obs = np.zeros(self.obs_shape, dtype=int)
        obs[indices] = 1
        obs[52] = self.game.players[player_idx].game_total_chip  # 玩家本局下注量
        obs[53] = self.game.players[player_idx].cur_stage_chip  # 玩家当前阶段下注量
        obs[54] = self.game.cur_stage_chip  # 当前阶段总下注量
        obs[55] = self.game.pot_chip  # 所有玩家总下注量
        legal_actions = [item.value for item in self.get_legal_actions()]  # 整数列表
        one_hot_legal_actions = to_one_hot(legal_actions, len(Action))
        legal_actions = np.array(one_hot_legal_actions)
        obs[-len(legal_actions):] = legal_actions
        return obs

    def get_reward(self, player_idx, action_is_legal, down, info):
        if down:
            # reward = int(info['payoffs'][player_idx])
            if int(info['payoffs'][player_idx]) > 0:
                reward = 100
            else:
                reward = -100
        else:
            reward = 1
        return reward


def test_action(env: NoLimitHoldemEnv, action: Action):
    obs, reward, down, info = env.step(action)
    print('obs:', obs)
    print('reward:', reward)
    print('down:', down)
    print('info', info)


def test():
    env = NoLimitHoldemEnv()
    env.game.reset()
    test_action(env, Action.CALL)
    test_action(env, Action.CHECK)
    test_action(env, Action.RAISE_100)


if __name__ == '__main__':
    test()
