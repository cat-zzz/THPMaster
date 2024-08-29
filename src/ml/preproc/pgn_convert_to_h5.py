"""
@project: THPMaster
@File   : pgn_convert_to_h5.py
@Desc   :
@Author : gql
@Date   : 2024/8/20 15:37
"""
import logging
import re

import h5py
import numpy as np

from src.util.logger import get_logger

CARDS = ['2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc', 'Ac',
         '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd', 'Ad',
         '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh', 'Ah',
         '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks', 'As']


def read_one_pgn_file(file_path):
    """
    读取一个棋谱文件，剔除掉不需要的列和行，返回一个列表，列表的每个元素是单局比赛的棋谱
    """
    with open(file_path, 'r') as file:
        games = file.readlines()
    games = games[4:]  # 前4行是一些说明信息，不需要
    drop_list = [0, 1]  # 原数据集中不需要的列
    games = [np.delete(_.split(':'), drop_list).tolist() for _ in games]
    return games


def read_pgn_files(file_paths: list[str]):
    """
    读取多个棋谱txt文件并返回所有棋谱列表(list[str])，每个元素是单局比赛的棋谱
    """
    games = []
    for file_path in file_paths:
        games.extend(read_one_pgn_file(file_path))  # extend()只接收list，且把该list中的每个元素添加到原list中
    return games


def convert_one_game_to_numpy(game_txt: str):
    pass


def convert_one_cards_game(one_cards_game: str, pos: int):
    """
    将pos位置的手牌和公共牌转为numpy形式
    :param one_cards_game: 一局比赛的手牌和公共牌（由于会有玩家弃牌，所以公共牌可能并不完全）
    :param pos: 待提取位置玩家的手牌，取值为0或1
    :return: ndarray[5,4,13]
    """
    '''
    示例
        输入：one_cards_game='Td4h|6cAd/5c8hKh/Qh/5h',pos=0
        输出：格式为[5, 4, 13]，其中第一个[4, 13]的矩阵是手牌（Td4h），
        第二个[4, 13]是flop阶段公共牌，以此类推，最后一个[4, 13]是当前所有已发出的公共牌和手牌
    '''
    cards = np.zeros((5, 4, 13), dtype=int)
    one_cards_history_splits = one_cards_game.split('/')
    # 提取手牌
    hands_history = one_cards_history_splits[0]  # 两位玩家的手牌都在这里，根据pos的值确定提取哪个玩家的手牌
    pos = pos * 5
    hand1 = CARDS.index(hands_history[pos:pos + 2])
    suit, num = hand1 // 13, hand1 % 13
    cards[0][suit][num] = 1
    hand2 = CARDS.index(hands_history[pos + 2:pos + 4])
    suit, num = hand2 // 13, hand2 % 13
    cards[0][suit][num] = 1
    # 提取公共牌
    public_cards = ''
    for _ in one_cards_history_splits[1:]:
        public_cards += _
    # 判断是否有公共牌
    if public_cards:
        # 每两个字符表示一张牌，进行遍历
        for i in range(0, len(public_cards), 2):
            suit, num = CARDS.index(public_cards[i:i + 2]) // 13, CARDS.index(public_cards[i:i + 2]) % 13
            # i的可能取值为0，2，4，6，8
            if i <= 4:  # flop公共牌
                cards[1][suit][num] = 1
            elif i == 6:  # turn公共牌
                cards[2][suit][num] = 1
            elif i == 8:  # river公共牌
                cards[3][suit][num] = 1
            cards[4][suit][num] = 1
    return cards


def convert_one_actions_game(one_actions_game: str):
    """
    将动作序列转为numpy张量形式
    :param one_actions_game: 动作序列
    """
    '''
    示例
        动作序列：r200r459c/r776c/r1313c/r2369r8100f
        单个动作表示
    '''
    one_actions_seq = np.zeros((2, 14), dtype=np.float32)


def specific_to_abstract_chip(specific_chip, last_chip, ratio_flag=1):
    """
    将具体筹码抽象为相较于最近一次下注或底池筹码（通过ratio_flag区分）的倍率
    :return: 不符合德扑规则的下注筹码返回-1
    """
    closest_index = -1
    if ratio_flag:
        ratio_seq = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]  # bet_ratio_seq，阶段内非第一次下注
    else:
        ratio_seq = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]  # pot_ratio_seq，阶段内第一次下注
    if last_chip == 0:
        logger.error('分母last_chip不能为0', exc_info=True)
        raise ValueError('分母last_chip不能为0')
    # 确保当为bet_ratio_seq时，specific_chip必须大于等于last_chip的两倍
    if ratio_flag and specific_chip < 2 * last_chip:
        logger.error(
            f'当前为bet_ratio_seq，specific_chip(值为{specific_chip})必须大于等于last_chip(值为{last_chip})的两倍')
        return closest_index

    ratio = (specific_chip - last_chip) / last_chip
    min_diff = float('inf')
    for index, value in enumerate(ratio_seq):
        diff = abs(ratio - value)
        if diff < min_diff:
            min_diff = diff
            closest_index = index
    return closest_index


if __name__ == '__main__':
    logger = get_logger()
    # 测试specific_to_abstract_chip()函数
    closest_index_1 = specific_to_abstract_chip(110, 100, 1)
    print(closest_index_1)
    pass
