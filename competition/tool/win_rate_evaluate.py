"""
@File   : win_rate_evaluate.py
@Desc   : 胜率评估
@Author : gql
@Date   : 2023/4/24 15:46
"""
import multiprocessing
import os
import time

import numpy as np
import pandas as pd

from competition.util.poker_util import deal_cards, compare_hands_by_random_cards
from competition.util.poker_value import best_cards
from competition.util.poker_util import per_win_score, per_draw_score, per_lose_score


def evaluate_hands_win_rate_by_single(my_hand_cards, oppo_hand_cards, total_count):
    """
    单进程手牌胜率模拟（2000次以下速度更快）
    """
    score = 0
    for _ in range(total_count):
        temp = compare_hands_by_random_cards(my_hand_cards, oppo_hand_cards)
        score += temp
    return score / (total_count * per_win_score)


def evaluate_hands_win_rate_by_multiprocess(my_hand_cards, oppo_hand_cards, total_count):
    """
    评估手牌胜率
    :param my_hand_cards: 我方手牌
    :param oppo_hand_cards: 对方手牌
    :param total_count: 总模拟次数
    :return: 我方手牌胜率
    """
    num_processes = 4
    pool = multiprocessing.Pool(num_processes)
    num_episodes = total_count // num_processes
    win_score = 0
    args = []
    for _ in range(num_processes):
        arg = [num_episodes, my_hand_cards, oppo_hand_cards]
        args.append(arg)
    result = pool.map(sub_evaluate_hands, tuple(args))
    for _ in range(len(result)):
        win_score += result[_]
    win_rate = win_score / (total_count * per_win_score)
    return win_rate


def sub_evaluate_hands(args):
    """
    用于手牌胜率模拟的子任务
    :return 返回经过num_episodes次模拟后，我方手牌获得的分数
    """
    num_episodes = args[0]
    my_hand_cards = args[1]
    oppo_hand_cards = args[2]
    score = 0
    for _ in range(num_episodes):
        temp = compare_hands_by_random_cards(my_hand_cards, oppo_hand_cards)
        score += temp
    return score


def evaluate_my_hands_win_rate(my_hand_cards, oppo_hand_cards, total_count):
    """
    模拟手牌胜率的主入口，根据模拟次数分为单、多进程两种方式模拟
    :param my_hand_cards: 我方手牌
    :param oppo_hand_cards: 对方手牌
    :param total_count: 模拟次数
    :return: 本次模拟我方手牌获得的分数（0，1，2分）
    """
    # if total_count > 2000:
    #     win_rate = evaluate_hands_win_rate_by_multiprocess(my_hand_cards, oppo_hand_cards, total_count)
    # else:
    #     win_rate = evaluate_hands_win_rate_by_single(my_hand_cards, oppo_hand_cards, total_count)
    win_rate = evaluate_hands_win_rate_by_single(my_hand_cards, oppo_hand_cards, total_count)
    return win_rate


def evaluate_hands_win_rate(my_hand_cards, total_count):
    if total_count <= 2000:
        win_rate = evaluate_hands_win_rate_by_single(my_hand_cards, None, total_count)
    else:
        win_rate = evaluate_hands_win_rate_by_multiprocess(my_hand_cards, None, total_count)
    # win_rate = evaluate_hands_win_rate_by_single(hands_cards, None, total_count)
    return win_rate


def evaluate_hands_win_rate_1(my_hand_cards, current_public_cards, total_count=1314):
    score = 0
    current_public_cards_count = current_public_cards.shape[0]
    win_count = 0
    tie_count = 0
    lose_count = 0

    for _ in range(total_count):
        # 随机发牌
        # 随机发出剩余公共牌
        remain_public = deal_cards(5 - current_public_cards_count,
                                   np.vstack((current_public_cards, my_hand_cards)))
        public_cards = np.vstack((current_public_cards, remain_public))
        # 随机发对手手牌
        oppo_hand_cards = deal_cards(2, np.vstack((public_cards, my_hand_cards)))
        # 比较大小
        my_poker_value = best_cards(my_hand_cards, public_cards)
        oppo_poker_value = best_cards(oppo_hand_cards, public_cards)
        if my_poker_value > oppo_poker_value:
            score += per_win_score  # 我方牌型比对方大，加4分
            win_count += 1
        elif my_poker_value == oppo_poker_value:
            score += per_draw_score  # 平局
            tie_count += 1
        else:
            score += per_lose_score  # 我方牌型比对方小，加0分
            lose_count += 1
    return score / (total_count * per_win_score)


def save_hands_win_rate():
    """
    生成并保存手牌胜率表
    """
    hands_win_rate = np.zeros((13, 13))
    for i in range(13):
        start = time.time()
        for j in range(i, 13):
            hand_cards_1 = np.array([[1, i], [0, j]])
            hand_cards_2 = np.array([[1, i], [1, j]])
            no_flush_win_rate = evaluate_hands_win_rate(hand_cards_1, 13140)
            hands_win_rate[i, j] = no_flush_win_rate  # 非同花，位于对角线上方
            flush_win_rate = 0
            if i != j:
                flush_win_rate = evaluate_hands_win_rate(hand_cards_2, 13140)
                hands_win_rate[j, i] = flush_win_rate  # 同花，位于对角线下方（其值一般比非同花的值大）
            print("{}, {}同花胜率:{}, 非同花胜率:{}".format(i, j, flush_win_rate, no_flush_win_rate))
        end = time.time()
        print("开始时间:", start)
        print("结束时间:", end)
        print("耗时:", str(end - start), end="\n\n")
    df = pd.DataFrame(hands_win_rate)
    df.to_csv('hands_win_rate.csv')
    return hands_win_rate


def save_hands_win_rate_ranking(hands_win_rate: np.ndarray):
    """
    保存手牌胜率排名
    :param hands_win_rate: numpy二维数组
    :return data: 手牌胜率排名(numpy二维数组)
    """
    data = np.argsort(np.argsort(-hands_win_rate, axis=None)).reshape(hands_win_rate.shape)
    print("手牌胜率排名:")
    print(data)
    df = pd.DataFrame(data)
    df.to_csv('hands_win_rate_rank.csv')
    hands_prob_sum = np.zeros([data.shape[0] * data.shape[1], 4], dtype=float)  # hands_prob_sum是二维数组
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # 如果i>j表示原二维数组中的上三角部分（非同花部分），i<j表示原二维数组中的下三角部分（同花部分）
            hands_prob_sum[data[i, j], 0] = i  # 第0列对应原数组中的行
            hands_prob_sum[data[i, j], 1] = j  # 第1列对应原数组中的列
            hands_prob_sum[data[i, j], 3] = hands_win_rate[i, j]
    prob_sum = 0  # 前i名手牌的概率之和
    for i in range(hands_prob_sum.shape[0]):
        if int(hands_prob_sum[i, 0]) == int(hands_prob_sum[i, 1]):  # 对子
            prob_sum += 0.004525
        elif int(hands_prob_sum[i, 0]) > int(hands_prob_sum[i, 1]):  # 同花
            prob_sum += 0.003017
        else:  # 非同花
            prob_sum += 0.009049
        hands_prob_sum[i, 2] = prob_sum
    hands_prob_sum = np.round(hands_prob_sum, 6)
    print("手牌概率之和:")
    print(hands_prob_sum)
    df = pd.DataFrame(hands_prob_sum)
    df.to_csv("hands_prob_sum.csv")
    return data


def get_hands_by_high_win_rate(min_win_tate=0.5, max_win_rate=1.0):
    """
    选出基础胜率大于basic_win_rate的所有手牌
    """
    # 读取文件需要用绝对路径
    hands_prob_sum = pd.read_csv(os.path.dirname(__file__) + '/hands_prob_sum.csv', header=0,
                                 usecols=range(1, 5)).values
    # high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1])]
    high_win_hand_cards = hands_prob_sum[np.where(high_win_hand_cards[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = high_win_hand_cards.astype(np.int32)
    # 将手牌转成不同的花色的手牌
    # hands_prob_sum.csv文件只考虑同花和非同花，没有考虑具体的花色
    suit_hand_cards = np.zeros((0, 2, 2), dtype=int)
    # suit_hand_cards = []
    for first, second in high_win_hand_cards:
        if first <= second:  # 对子(first==second)和非同花(first<second)
            for i in range(4):  # 确定第一张牌的花色
                for j in range(i + 1, 4):  # 确定第二张牌的花色
                    if i != j:
                        temp = np.array([[[i, first], [j, second]]], dtype=int)
                        suit_hand_cards = np.append(suit_hand_cards, temp, axis=0)
        else:  # 同花
            for i in range(4):  # 两张牌花色相同
                temp = np.array([[[i, first], [i, second]]], dtype=int)
                suit_hand_cards = np.append(suit_hand_cards, temp, axis=0)
    return suit_hand_cards


def win_rate_by_oppo_hands(my_hands: np.ndarray, oppo_hands: np.ndarray, count=1314):
    """
    我方手牌与对方可能的手牌oppo_hands随机对弈，计算胜率。
    :param my_hands: 我方手牌（只能是一副手牌）
    :param oppo_hands: 对方可能的手牌集合（若干副手牌）
    :param count: 随机模拟总次数
    :return: 我方手牌的胜率
    """
    # 随机遍历1000次手牌
    score = 0
    for _ in range(count):
        oppo_hand = oppo_hands[np.random.choice(oppo_hands.shape[0])]  # 从可能的对手手牌中随机选择一副
        if not (oppo_hand == my_hands).all():  # 双方手牌不能相同
            score += compare_hands_by_random_cards(my_hands, oppo_hand)
    return score / (count * per_win_score)


def win_rate_by_basic_oppo_hands(my_hands, basic_win_rate=0.5):
    oppo_hands = get_hands_by_high_win_rate(basic_win_rate)
    return win_rate_by_oppo_hands(my_hands, oppo_hands)


if __name__ == '__main__':
    # 测试evaluate_hands_win_rate_1()函数
    # hand_cards = np.array([[2, 6], [3, 3]])
    # public = np.array([[0, 12], [2, 1], [1, 1]])
    # oppo_cards = np.array([[0, 5], [1, 3]])
    # # oppo_cards = None
    # rate = evaluate_hands_win_rate_1(hand_cards, public, oppo_cards, 1314)
    # print(rate)

    # 测试save_hands_win_rate()函数
    hands = save_hands_win_rate()
    hands = pd.read_csv('hands_win_rate.csv', header=0, usecols=range(1, 14)).values
    save_hands_win_rate_ranking(hands)

    # 测试get_hands_by_high_win_rate()函数
    # get_hands_by_high_win_rate()

    # 测试win_rate_by_oppo_hands()函数
    # hands = np.array([[1, 10], [1, 1]], dtype=int)
