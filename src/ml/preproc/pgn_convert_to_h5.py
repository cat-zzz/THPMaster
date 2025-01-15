"""
@project: THPMaster
@File   : pgn_convert_to_h5.py
@Desc   :
@Author : gql
@Date   : 2024/8/20 15:37
"""
import re
import time

import numpy as np

from src.ml.preproc.encode_pot_binning import chip_encoding_with_distance
from src.util.logger import logger

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
    # 转换为四元组（状态、动作、奖励、下一状态）
    pass


def convert_one_game_cards(one_cards_game: str, pos: int):
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


num_active_players = 2  # 只适用于双人德扑，此变量的值一直为2


def convert_one_game_actions(one_actions_game: str):
    """
    将动作序列转为numpy形式
    :param one_actions_game: 动作序列，格式为[动作序列长度, 动作特征维度（动作编码(8维+5维)+位置信息(1维)+底池编码(8维)）]
    """
    '''
    示例
        动作序列：r200r459c/r776c/r1313c/r2369r8100f
        单个动作用21个元素表示，[位置信息(1维), 底池编码(8维), allin, fold, check/call, raise, 下注筹码编码(8维，仅对raise动作有效)]
    '''
    player_chip = np.array([100, 50, 0])  # 前两个元素分别表示大小盲注在当前阶段内的下注筹码量，最后一个元素表示当前阶段之前的底池大小，所有元素相加才是整个底池大小
    one_actions_seq = np.zeros((24, 22), dtype=np.float32)
    one_actions_game_splits = one_actions_game.split('/')
    action_idx = 0  # one_actions_seq的索引，表示当前进行到第几个动作
    for i, one_stage in enumerate(one_actions_game_splits):  # 遍历每个阶段
        one_stage = re.findall(r'\d+|f|c', one_stage)
        if i == 0:
            player_idx = 1  # 当前行动的玩家索引
            last_player_idx = 0  # 上一位行动的玩家索引，初始置为1表示大盲注，若是多人德扑需考虑有玩家弃牌的情况
        else:
            player_idx = 0
            last_player_idx = 1
        k = 0  # k用来表示是否为一个阶段内的第一个raise动作，将用于specific_to_abstract_chip()函数中的ratio_flag
        for raw_action in one_stage:  # 遍历一个阶段内的所有动作
            # 底池编码，需要先进行底池编码再进行动作编码，因为底池编码是对当前动作未执行的底池进行编码
            pot_weight = chip_encoding_with_distance(np.sum(player_chip))
            one_actions_seq[action_idx][1:9] = pot_weight
            # 动作编码
            if raw_action.isdigit():  # raise动作
                raise_chip = int(raw_action)
                if raise_chip >= 20000:  # 原数据集里的allin用r20000表示，对方此时只能call或fold
                    one_actions_seq[action_idx][9] = 1
                elif k == 0 and i != 0:  # 第一次下注，只存在非preflop阶段，preflop阶段没有第一次下注（因为已经下了大小盲注）
                    ratio = raise_chip / player_chip[-1]  # 相对于底池的倍率，由于是阶段内第一次下注，所以player_chip其他值均为0
                    chip_encoding = specific_to_abstract_chip(ratio, ratio_flag=k)
                    one_actions_seq[action_idx][-8:] = chip_encoding
                    one_actions_seq[action_idx][12] = 1
                    k = 1
                else:  # 一般情况的下注动作（不是本阶段内的第一次下注）
                    ratio = raise_chip / np.sum(player_chip[last_player_idx])
                    k = 1  # preflop阶段没有第一次下注，k需要先置1
                    chip_encoding = specific_to_abstract_chip(ratio, ratio_flag=k)
                    one_actions_seq[action_idx][-8:] = chip_encoding
                    one_actions_seq[action_idx][12] = 1
                player_chip[player_idx] = raise_chip
            elif raw_action == 'f':  # fold
                one_actions_seq[action_idx][10] = 1
            elif raw_action == 'c':  # check或call
                one_actions_seq[action_idx][11] = 1
                player_chip[player_idx] = player_chip[last_player_idx]
            else:
                raise ValueError(f'未知的动作:{raw_action}')
            # 位置编码
            one_actions_seq[action_idx][0] = player_idx
            # 更新变量信息
            player_idx = (player_idx + 1) % num_active_players
            last_player_idx = (last_player_idx + 1) % num_active_players
            action_idx += 1
        # 一个阶段结束，更新筹码信息
        player_chip[-1] = np.sum(player_chip[0:-1])
        player_chip[:-1] = 0
    return one_actions_seq


# 使用此方法
def specific_to_abstract_chip(current_ratio, ratio_flag=1, threshold=1e-3):
    if ratio_flag:  # 相对于上一位下注玩家的倍率
        ratio_seq = np.array([2, 2.5, 3, 3.75, 4.5, 5.75, 7.5, 9])
        if current_ratio <= 3:
            sigma = 0.15
        elif current_ratio <= 4:
            sigma = 0.25
        elif current_ratio <= 5:
            sigma = 0.4
        elif current_ratio <= 7:
            sigma = 0.53
        else:
            sigma = 0.7
    else:  # 相对于底池的倍率
        # ratio_seq = np.array([0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4])
        ratio_seq = np.array([0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4])
        if current_ratio <= 1.5:
            sigma = 0.1
        elif current_ratio <= 2.75:
            sigma = 0.2
        else:
            sigma = 0.3
    # 计算当前下注金额与各个基准倍数的差异，并使用高斯核进行加权
    differences = current_ratio - ratio_seq  # 后续计算会对differences进行平方，所以此处无需计算绝对值
    similarities = np.exp(-differences ** 2 / (2 * sigma ** 2))
    normalized_similarities = similarities / np.sum(similarities)
    # 识别和处理小于阈值的元素，并将其权重重新分配
    mask = normalized_similarities >= threshold
    small_elements_sum = np.sum(normalized_similarities[~mask])
    normalized_similarities[~mask] = 0
    remaining_sum = np.sum(normalized_similarities)  # 将小于阈值的元素的总和平均分配给剩余的元素
    if remaining_sum > 0:
        normalized_similarities += normalized_similarities * (small_elements_sum / remaining_sum)
    return normalized_similarities


# 不推荐此方法
def specific_to_abstract_chip_2(specific_chip, last_chip, ratio_flag=1):
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


def test_specific_to_abstract_chip():
    # logger = get_logger()
    # 测试specific_to_abstract_chip()函数
    # closest_index_1 = specific_to_abstract_chip(110, 100, 1)
    # print(closest_index_1)
    # result = specific_to_abstract_chip_2(220, 100, 10)
    # print(result)
    # result = specific_to_abstract_chip_2(1000, 400, 1, 40)
    # print(result)
    # result = specific_to_abstract_chip_3(3, 1)
    # print(result)
    # result = specific_to_abstract_chip_3(3, 1)
    # print(result)
    # result = specific_to_abstract_chip_3(3.15, 1)
    # print(result)
    start_time = time.time()
    cr = 0.1
    incr = 0.05
    for i in range(100):
        cr = round(cr, 2)
        result = specific_to_abstract_chip(cr, 1)
        result[result < 1e-3] = 0
        print('current ratio:', cr, 'result:', result)
        cr += incr
    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间为: {runtime} 秒")

    # start_time = time.time()
    # cr = 0.1
    # incr = 0.05
    # for i in range(1000000):
    #     cr = round(cr, 2)
    #     result = specific_to_abstract_chip_2(cr, 0)
    #     result[result < 1e-3] = 0
    #     # print('current ratio:', cr, 'result:', result)
    #     cr += incr
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f"程序运行时间为: {runtime} 秒")


def test_convert_one_game():
    actions = 'r200r459c/r776c/r1313c/r2369r8100f'
    result = convert_one_game_actions(actions)
    print(result)


if __name__ == '__main__':
    test_specific_to_abstract_chip()
    # test_convert_one_game()
    pass
