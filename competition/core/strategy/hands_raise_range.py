"""
@project: THPMaster
@File   : hands_raise_range.py
@Desc   :
@Author : gql
@Date   : 2024/7/25 11:39
"""
import os
import re

import numpy as np
import pandas as pd

CARDS = ['2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', 'Tc', 'Jc', 'Qc', 'Kc', 'Ac',
         '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', 'Td', 'Jd', 'Qd', 'Kd', 'Ad',
         '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', 'Th', 'Jh', 'Qh', 'Kh', 'Ah',
         '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', 'Ts', 'Js', 'Qs', 'Ks', 'As']


def read_pgn_files(file_paths: list[str]):
    """
    读取多个棋谱txt文件并返回所有棋谱列表(list[str])，每个元素是单局比赛的棋谱
    """
    games = []
    for file_path in file_paths:
        games.extend(read_pgn_file(file_path))  # extend()只接收list，且把该list中的每个元素添加到原list中
    return games


def read_pgn_file(file_path):
    with open(file_path, 'r') as file:
        games = file.readlines()
    games = games[4:]  # 前4行是一些说明信息，不需要
    drop_list = [0, 1, 4, 5]  # 原数据集中不需要的列
    games = [np.delete(_.split(':'), drop_list).tolist() for _ in games]
    return games


# 所有手牌组合（考虑花色）的初始下注筹码
cards_raise_chip = np.zeros((52, 52), dtype=float)
cards_raise_count = np.zeros((52, 52), dtype=int)
num_active_players = 2


def convert_one_game(one_game: str):
    pos1 = 0
    pos2 = 5
    one_cards_game_splits = one_game[1].split('/')
    one_actions_game = one_game[0]
    hands_history = one_cards_game_splits[0]
    # 大盲注手牌
    hand1_1_idx = CARDS.index(hands_history[pos1:pos1 + 2])
    hand1_2_idx = CARDS.index(hands_history[pos1 + 2:pos1 + 4])
    # 小盲注手牌
    hand2_1_idx = CARDS.index(hands_history[pos2:pos2 + 2])
    hand2_2_idx = CARDS.index(hands_history[pos2 + 2:pos2 + 4])
    one_actions_game_splits = one_actions_game.split('/')
    players_chip = [50, 100]
    # m = 0  # 区分每个阶段内的第二个c
    j = 0
    # print(one_actions_game_splits)
    first_fold = False
    one_split = re.findall(r'\d+|f|c', one_actions_game_splits[0])  # 提取出数字或f或c
    for i, raw_action in enumerate(one_split):
        if i == 0 and raw_action == 'f':
            first_fold = True
        if raw_action.isdigit():  # 表示此raw_action是raise动作
            raise_chip = int(raw_action)
            players_chip[j] = raise_chip
        elif raw_action == 'f':  # fold
            pass
        elif raw_action == 'c':  # check或call
            players_chip[j] = players_chip[(j + 1) % num_active_players]
        else:
            raise ValueError(f'未知的动作:{raw_action}')
        j = (j + 1) % num_active_players
    return hand1_1_idx, hand1_2_idx, hand2_1_idx, hand2_2_idx, players_chip, first_fold


def analyse_hands_range_from_file(file_path):
    pgn = read_pgn_file(file_path)
    for e in pgn:
        hand1_1_idx, hand1_2_idx, hand2_1_idx, hand2_2_idx, players_chip, first_fold = convert_one_game(e)
        if first_fold is True:  # 小盲注首次动作即弃牌
            cards_raise_count[hand2_1_idx][hand2_2_idx] += 2.5  # 加大弃牌的比重
        elif players_chip[0] == players_chip[1]:
            cards_raise_chip[hand1_1_idx][hand1_2_idx] += players_chip[0]
            cards_raise_chip[hand2_1_idx][hand2_2_idx] += players_chip[1]
            cards_raise_count[hand1_1_idx][hand1_2_idx] += 1
            cards_raise_count[hand2_1_idx][hand2_2_idx] += 1
        else:
            if players_chip[0] > players_chip[1]:
                cards_raise_chip[hand1_1_idx][hand1_2_idx] += players_chip[0]
                cards_raise_count[hand1_1_idx][hand1_2_idx] += 1
            else:
                cards_raise_chip[hand2_1_idx][hand2_2_idx] += players_chip[1]
                cards_raise_count[hand2_1_idx][hand2_2_idx] += 1


def analyse_hands_range_from_dir(source_dir):
    print(f'共{len(os.listdir(source_dir))}个文件')
    for i, filename in enumerate(os.listdir(source_dir)):
        analyse_hands_range_from_file(os.path.join(source_dir, filename))
        if i % 100 == 0:
            np.save('output\\cards_raise_chip_' + str(i) + '.npy', cards_raise_chip)
            np.save('output\\cards_raise_count_' + str(i) + '.npy', cards_raise_count)
            print(f'已保存前{i}个文件的结果，{filename}')
    np.save('output\\cards_raise_chip_final.npy', cards_raise_chip)
    np.save('output\\cards_raise_count_final.npy', cards_raise_count)
    print(f'已保存所有文件的结果')


def load_cards_raise_chip_to_result(cards_raise_chip_path, cards_raise_count_path, save_result_path):
    """
    根据保存的cards_raise_chip.npy和cards_raise_count.npy计算出最后的下注筹码分布
    """
    cards_raise_chip_1 = np.load(cards_raise_chip_path)
    cards_raise_count_1 = np.load(cards_raise_count_path)
    n, m = cards_raise_chip_1.shape
    i_upper, j_upper = np.triu_indices(n, k=1)  # 获取上三角部分的索引（不包括主对角线）
    cards_raise_chip_1[i_upper, j_upper] += cards_raise_chip_1[j_upper, i_upper]  # 将下三角部分的元素加到上三角部分
    cards_raise_chip_1[j_upper, i_upper] = cards_raise_chip_1[i_upper, j_upper]  # 将下三角部分的元素设置为上三角对应位置的元素
    i_upper, j_upper = np.triu_indices(n, k=1)
    cards_raise_count_1[i_upper, j_upper] += cards_raise_count_1[j_upper, i_upper]
    cards_raise_count_1[j_upper, i_upper] = cards_raise_count_1[i_upper, j_upper]
    result = cards_raise_chip_1 / cards_raise_count_1
    result[result < 150] = 0  # 将下注小于200的手牌置为0
    np.save(save_result_path, result)
    print("对角元素的和：", cards_raise_chip_1, cards_raise_count_1)
    print('result', result)


def result_convert_to_csv(file_path):
    result = np.load(file_path)
    np.savetxt('hands_raise_range_result_6.csv', result, delimiter=',', fmt='%.2f')


if __name__ == '__main__':
    dir1 = 'D:\\Desktop\\thp_dataset\\acpc_dataset_original'
    save_path = 'output\\result5.npy'
    # 1 提取数据集
    # analyse_hands_range_from_dir(dir1)
    # 2 分析结果
    # load_cards_raise_chip_to_result('output\\cards_raise_chip_final.npy',
    #                                 'output\\cards_raise_count_final.npy',
    #                                 save_path)
    # 3 转为csv文件
    # result_convert_to_csv(save_path)

    hand_cards_raw = "As8c"
    # hand_cards_raw = "2c8c"
    hands_raise_range_result = pd.read_csv(
        'D:\\Development\\pythonProjects\\THPMaster\\competition\\core\\strategy\\hands_raise_range_result_6.csv',
        header=None)
    card1_idx = CARDS.index(hand_cards_raw[0:2])
    card2_idx = CARDS.index(hand_cards_raw[2:4])
    print(card1_idx, card2_idx)
    hands_raise_chip = hands_raise_range_result.iloc[card1_idx, card2_idx]
    print(hands_raise_chip)
