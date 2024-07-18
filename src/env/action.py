"""
@project: THPMaster
@File   : action.py
@Desc   :
@Author : gql
@Date   : 2024/7/11 17:21
"""
from enum import Enum

import numpy as np

from src.env import constants


class Action(Enum):
    # 想法1：把总筹码20000分成若干个动作，动作抽象依据总筹码进行
    # 想法2：动作不再是one-hot类型，每个动作都（可能）对应一个连续值（0~1），类似底池筹码的编码方式。下注动作就变成连续的了
    # 同样，最后输出的动作也可以结合每种动作的概率计算出一个具体的筹码值
    RAISE_1_BET = 0
    RAISE_1_25_BET = 1
    RAISE_1_75_BET = 2
    RAISE_2_BET = 3
    RAISE_2_5_BET = 4
    RAISE_3_BET = 5
    RAISE_4_BET = 6
    RAISE_5_BET = 7
    RAISE_6_5_BET = 8
    RAISE_8_BET = 9
    ALLIN = 10
    CALL = 11
    FOLD = 12


raise_action_mapping = [2, 2.25, 2.75, 3, 3.5, 4, 5, 6, 7.5, 9]


def action_to_one_hot(action: Action):
    """
    将抽象后的动作
    :param action:
    :return:
    """
    length = len(Action)
    one_hot = np.zeros(length, dtype=int)
    one_hot[action.value] = 1
    print(one_hot)
    return one_hot


def action_to_raw(action: Action, last_chip):
    """
    将离散化后的动作转为具体的筹码量
    :param action:
    :param last_chip:
    :return:
    """
    raw_action = np.zeros(2, dtype=int)
    not_raise_action = {
        Action.CALL,
        Action.FOLD,
        Action.ALLIN
    }
    if action in not_raise_action:
        # todo call和check共用一个动作位，需要做出区分
        if action.value == Action.CALL:
            raw_action[0] = constants.CALL_ACTION
        elif action.value == Action.FOLD:
            raw_action[0] = constants.FOLD_ACTION
        elif action.value == Action.ALLIN:
            raw_action[0] = constants.ALLIN_ACTION
    else:
        raise_ratio = raise_action_mapping[action.value]
        raw_action[0] = constants.RAISE_ACTION
        raw_action[1] = int(last_chip * raise_ratio)
    return raw_action


if __name__ == '__main__':
    action_to_one_hot(Action.CALL)
    action_to_raw(Action.RAISE_1_25_BET, 1000)
