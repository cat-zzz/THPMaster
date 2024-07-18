"""
@project: THPMaster
@File   : pgn_to_h5_converter.py
@Desc   :
@Author : gql
@Date   : 2024/7/17 14:01
"""
import re

import h5py
import numpy as np

CARDS = ['2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc', 'Ac',
         '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd', 'Ad',
         '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh', 'Ah',
         '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks', 'As']


def read_pgn_file(file_path):
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
        games.extend(read_pgn_file(file_path))  # extend()只接收list，且把该list中的每个元素添加到原list中
    return games


def convert_game_to_numpy(game_txt: str):
    """
    将单局比赛的棋谱转换为numpy数组
    :param game_txt: (str)单局比赛的棋谱
    """
    # todo
    pass


def convert_one_cards_game(one_cards_game: str, pos: int):
    """
    将pos位置的手牌和公共牌转为numpy形式
    :param one_cards_game: 手牌和公共牌，非一局完整的比赛棋谱
    :param pos: 待提取位置玩家的手牌，取值为0或1
    :return: ndarray[5,4,13]
    """
    '''
    示例
        输入：one_cards_history='Td4h|6cAd/5c8hKh/Qh/5h',pos=0
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


num_active_players = 2  # 只适用双人德扑，即此处值只能为2


def convert_one_actions_game(one_actions_game):
    """
    提取一局比赛的动作序列，转为numpy张量形式
    :param one_actions_game:
    :return:
    """
    '''
    示例
        r200r459c/r776c/r1313c/r2369r8100f
    '''
    # todo 对于每个动作都拆分一个动作序列
    one_actions_seq = np.zeros((2, 14), dtype=np.float32)
    all_actions_seq = []
    one_actions_game_splits = one_actions_game.split('/')
    players_chip = [50, 100, 0]  # 前两个元素表示大小盲注在当前阶段内的下注筹码量，最后一个元素表示除当前阶段下注筹码之外的底池大小
    # 遍历每个阶段
    for i, one_split in enumerate(one_actions_game_splits):
        one_split = re.findall(r'\d+|f|c', one_split)  # 提取出数字或f或c
        if i == 0:  # flop阶段，小盲注先行动
            j = 0  # j用来表示当前动作的玩家索引（大小盲注索引）
            last_chip = 100
        else:  # 其他阶段，大盲注先行动
            j = 1
            last_chip = 0
        action_idx = 0
        k = 0  # k用来表示是否为每个阶段内的第一个raise动作
        m = 0  # 区分每个阶段内的第二个c
        # 遍历一个阶段内的所有动作
        for raw_action in one_split:
            if raw_action.isdigit():  # 表示此raw_action是raise动作
                raise_chip = int(raw_action)
                if raise_chip >= 20000:  # 原数据集里的allin用r20000表示，对方此时只能call或fold
                    l = -4  # l对应当前动作在动作列表中的索引
                    k = 1
                elif k == 0 and i != 0:  # 非flop阶段内第一次下注。flop阶段没有第一次下注（因为已经下了大小盲注）
                    l = specific_to_abstract_chip(raise_chip, players_chip[-1], k)
                    k = 1
                else:
                    k = 1  # flop阶段没有第一次下注，k需要先置1
                    l = specific_to_abstract_chip(raise_chip, last_chip, k)
                last_chip = raise_chip
                players_chip[j] = raise_chip
                one_actions_seq[i][action_idx][l] = 1
            elif raw_action == 'f':  # fold
                one_actions_seq[i][action_idx][-1] = 1
            elif raw_action == 'c':  # check或call
                one_actions_seq[i][action_idx][-2] = 1
                if players_chip[0] == players_chip[1] and m == 0:  # check
                    one_actions_seq[i][action_idx][-2] = 1
                else:  # call
                    one_actions_seq[i][action_idx][-3] = 1
                players_chip[j] = players_chip[(j + 1) % num_active_players]
                m = 1
            else:
                raise ValueError(f'未知的动作:{raw_action}')
            one_actions_seq[i][action_idx + 2] = j + 1  # 更新位置信息，位置信息用[1, 2]表示大小盲注，而j的取值范围为[0,1]
            j = (j + 1) % num_active_players
            action_idx += 3
        players_chip[-1] += players_chip[0] + players_chip[1]
        players_chip[0] = 0  # 非flop阶段，两位玩家的初始筹码量都置为0
        players_chip[1] = 0
        return one_actions_seq


def specific_to_abstract_chip(specific_chip, last_chip, ratio_flag=1):
    if ratio_flag:
        ratio_seq = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]  # bet_ratio_seq
    else:
        ratio_seq = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4]  # pot_ratio_seq
    if last_chip == 0:
        raise ValueError("分母last_chip不能为0")
    ratio = specific_chip / last_chip
    min_diff = float('inf')
    closest_index = -1
    for index, value in enumerate(ratio_seq):
        diff = abs(ratio - value)
        if diff < min_diff:
            min_diff = diff
            closest_index = index
    return closest_index


def save_numpy_to_h5(numpy_arrays, output_file: str):
    """
    将多个numpy数组保存为h5文件001.
    """
    with h5py.File(output_file, 'w') as h5f:
        for i, array in enumerate(numpy_arrays):
            h5f.create_dataset(f'game_{i}', data=array)


def save_pgn_files_to_h5(input_files: list[str], output_file):
    """
    将多个棋谱文件（txt）转换为一个h5文件
    :param input_files:
    :param output_file:
    :return:
    """
    pass


if __name__ == '__main__':
    pass
