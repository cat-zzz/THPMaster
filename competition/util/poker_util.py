"""
@File   : poker_util.py
@Desc   : 判断牌型
@Author : gql
@Date   : 2023/2/4 22:59
"""
import re

import numpy as np

from competition.core import constants
# from competition.tool.win_rate_evaluate import per_win_score, per_draw_score, per_lose_score
from competition.util.common_util import print_exception
from competition.util.poker_value import best_cards

per_lose_score = 0
per_draw_score = 2
per_win_score = 4


def deal_one_card(excludes=None):
    """
    随机发一张不在excludes里的牌

    :return: [花色, 点数] numpy数组类型
    """
    # 无需递归调用，只执行一次就能随机产生一张不在excludes里的牌
    if excludes is None:
        num = np.random.randint(0, 52)
        card = np.array((num % 4, num // 4))
        card = card.reshape((1, 2))
        return card
    else:
        count = excludes.shape[0]
        # 先按花色再按点数排序
        # 先按第一列排序（excludes[:,0]）再按第二列排序（excludes[:,1]），返回值是排序之后的索引值（不是数组）
        index = np.lexsort((excludes[:, 1], excludes[:, 0]))
        excludes = excludes[index]  # 把index里的数据当作excludes的索引值重新排序
        num = np.random.randint(0, 52 - count)
        for i in range(count):
            temp = excludes[i][0] * 13 + excludes[i][1]
            if temp <= num:
                num += 1
        card = np.array([[num // 13, num % 13]])
        card = card.reshape((1, 2))
        return card


def deal_cards(num=1, excludes=None):
    """
    随机发若干张牌

    :param num: 发牌数量
    :param excludes: 排除在外的牌
    :return: [花色, 点数] numpy数组类型
    """
    if excludes is None:
        excludes = np.zeros((0, 2), dtype=int)
    cards = np.zeros((0, 2), dtype=int)
    for i in range(num):
        card = deal_one_card(excludes)  # 随机生成一张牌
        excludes = np.vstack((excludes, card))  # 将这张牌添加到排除列表中
        cards = np.vstack((cards, card))
    return cards


def extract_cards_from_reply(reply):
    """
    从服务端返回的原始信息里提取手牌或公共牌信息

    :param reply: 服务端返回的原始信息
    :return: nparray
    """
    patten = r"<(.+?)>"  # 正则表达式
    # reply = r"enter_flop|<0,9><1,6><2,7>"
    str_list = re.split(patten, reply)
    # str_list的结果类似['1,2', '3,9']
    str_list = str_list[1::2]  # 步长为2
    cards = []
    card_raw = ''
    for i in range(len(str_list)):
        line = str_list[i].split(",")
        int_list = list(map(int, line))
        cards.append(int_list)
        card_raw += new_func1_card_encode(int_list)  # 2024-07-24新增代码
    cards = np.array(cards)
    print(f'extract_cards_from_reply func()->原始手牌: {reply}, 解析手牌: {card_raw}')
    return cards


def new_func1_card_encode(int_list):
    # 2024-07-24新增代码，仅用在extract_cards_from_reply函数中，将手牌编码为3d格式，只能编码一张牌
    if int_list[1] == 10:
        card_raw = 'T'
    elif int_list[1] == 11:
        card_raw = 'J'
    elif int_list[1] == 12:
        card_raw = 'Q'
    elif int_list[1] == 13:
        card_raw = 'K'
    elif int_list[1] == 14:
        card_raw = 'A'
    else:
        card_raw = str(int_list[1])
    if int_list[0] == 0:
        card_raw += 's'
    elif int_list[0] == 1:
        card_raw += 'h'
    elif int_list[0] == 2:
        card_raw += 'd'
    elif int_list[0] == 3:
        card_raw += 'c'
    return card_raw


def trans_to_server_msg(operation: np.ndarray):
    action = operation[0, 0]
    if action == constants.player_call_action:
        return 'call'
    elif action == constants.player_fold_action:
        return 'fold'
    elif action == constants.player_allin_action:
        return 'allin'
    elif action == constants.player_check_action:
        return 'check'
    elif action == constants.player_raise_action:
        return 'raise ' + str(int(operation[0, 1]))
    else:
        print_exception(trans_to_server_msg, "未知的玩家行为")
        return 'fold'


def test():
    print_exception(test, "未知身份")


def compare_hands_by_random_cards(my_hand_cards, oppo_hand_cards=None):
    """
    随机发公共牌牌，比较双方牌型大小
    :param my_hand_cards: 我方手牌
    :param oppo_hand_cards: 对方手牌（若为空，则随机生成对方手牌）
    :return: 本次模拟我方手牌获得的分数
    """
    if oppo_hand_cards is None:  # 为对手随机发手牌
        oppo_hand_cards = deal_cards(2, my_hand_cards)
    all_hand_cards = np.vstack((my_hand_cards, oppo_hand_cards))
    public_cards = deal_cards(5, all_hand_cards)
    my_poker_value = best_cards(my_hand_cards, public_cards)
    oppo_poker_value = best_cards(oppo_hand_cards, public_cards)
    if my_poker_value > oppo_poker_value:
        return per_win_score  # 我方牌型比对方大，加4分
    elif my_poker_value == oppo_poker_value:
        return per_draw_score  # 平局，加1分
    else:
        return per_lose_score  # 我方牌型比对方小，加0分


def compare_hands_by_partial_public_cards(my_hand_cards, current_public_cards, oppo_hand_cards=None):
    pass


if __name__ == '__main__':
    cs = np.array([[3, 9], [1, 9], [2, 11], [2, 10], [2, 12], [2, 8], [0, 3]])
    deal_one_card(cs)
    for _ in range(100):
        c = deal_cards(5, cs)
        print(c)
