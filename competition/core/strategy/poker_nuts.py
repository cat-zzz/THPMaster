"""
@project: THPMaster
@File   : poker_nuts.py
@Desc   :
@Author : gql
@Date   : 2024/7/24 20:16
"""
import numpy as np
import pandas as pd
from treys import Card, Evaluator, Deck

from competition.util.poker_util import deal_cards
from competition.util.poker_value import best_cards


def is_nuts(hole_cards, community_cards):
    evaluator = Evaluator()
    deck = Deck()

    # Convert cards to treys format
    hole_cards = [Card.new(card) for card in hole_cards]
    community_cards = [Card.new(card) for card in community_cards]

    # Calculate the rank of the current hand
    my_hand_rank = evaluator.evaluate(community_cards, hole_cards)

    # Generate all possible opponents' hands
    best_opponent_hand_rank = 7463  # worse possible rank in Texas Hold'em
    for card1 in deck.cards:
        for card2 in deck.cards:
            if card1 >= card2 or card1 in community_cards or card2 in community_cards or card1 in hole_cards or card2 in hole_cards:
                continue

            opponent_hand = [card1, card2]
            opponent_hand_rank = evaluator.evaluate(community_cards, opponent_hand)

            if opponent_hand_rank < best_opponent_hand_rank:
                best_opponent_hand_rank = opponent_hand_rank

    return my_hand_rank < best_opponent_hand_rank


per_lose_score = 0
per_draw_score = 2
per_win_score = 4


def get_hands_with_win_rate_section(min_win_tate=0.5, max_win_rate=1.0, exclude=None):
    """
    选出基础胜率在[min_win_rate, max_win_rate]区间下的所有手牌(区分手牌花色)
    """
    # 读取文件需要用绝对路径
    # hands_prob_sum = pd.read_csv('../../tool/hands_prob_sum.csv', header=0,
    #                              usecols=range(1, 5)).values
    # 从main.py启动时，运行下面的代码
    hands_prob_sum = pd.read_csv('../../tool/hands_prob_sum.csv', header=0, usecols=range(1, 5)).values
    # high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = hands_prob_sum[np.where(min_win_tate <= hands_prob_sum[:, -1])]
    high_win_hand_cards = hands_prob_sum[np.where(high_win_hand_cards[:, -1] <= max_win_rate)][:, :-2]
    high_win_hand_cards = high_win_hand_cards.astype(np.int32)
    # 将手牌转成不同的花色的手牌，hands_prob_sum.csv文件只考虑同花和非同花，没有考虑具体的花色
    suit_hand_cards = np.zeros((0, 2, 2), dtype=int)
    for first, second in high_win_hand_cards:
        if first <= second:  # 对子(first==second)和非同花(first<second)
            for i in range(4):  # 确定第一张牌的花色
                for j in range(i + 1, 4):  # 确定第二张牌的花色
                    if i != j:
                        temp = np.array([[[i, first], [j, second]]], dtype=int)
                        if exclude is None or \
                                (exclude is not None and temp[0, 0] not in exclude and temp[0, 1] not in exclude):
                            suit_hand_cards = np.append(suit_hand_cards, temp, axis=0)
        else:  # 同花
            for i in range(4):  # 两张牌花色相同
                temp = np.array([[[i, first], [i, second]]], dtype=int)
                suit_hand_cards = np.append(suit_hand_cards, temp, axis=0)
    return suit_hand_cards


def my_hands_win_rate(my_hands: np.ndarray, current_public_cards: np.ndarray,
                      oppo_hands=get_hands_with_win_rate_section(), total_count=1314):
    """
    随机模拟若干次后，我方手牌的胜率

    :param my_hands: 我方手牌（二维数组）
    :param current_public_cards: 当前公共牌（二维数组）
    :param oppo_hands: 对方可能的手牌集合（三维数组）
    :param total_count: 总模拟次数
    :return: 我方手牌胜率
    """
    win_count = 0
    tie_count = 0
    lose_count = 0
    score = 0
    current_public_cards_count = current_public_cards.shape[0]
    for _ in range(total_count):
        # 随机发出剩下的公共牌
        remain_public = deal_cards(5 - current_public_cards_count, np.vstack((current_public_cards, my_hands)))
        public_cards = np.vstack((current_public_cards, remain_public))
        oppo_hand = oppo_hands[np.random.choice(oppo_hands.shape[0])]  # 从可能的对手手牌中随机选择一副
        # 比较大小
        my_poker_value = best_cards(my_hands, public_cards)
        oppo_poker_value = best_cards(oppo_hand, public_cards)
        if my_poker_value > oppo_poker_value:
            score += per_win_score  # 我方牌型比对方大，加4分
            win_count += 1
        elif my_poker_value == oppo_poker_value:
            tie_count += 1
            score += per_draw_score  # 平局
        else:
            score += per_lose_score  # 我方牌型比对方小，加0分
            lose_count += 1
    print("win:", win_count, "tie:", tie_count, "lose:", lose_count, "total:", total_count)
    return score / (total_count * per_win_score)


def example():
    hole_cards = ['As', 'Ks']
    hole_cards = ['As', 'Ks']
    hole_cards = ['3s', 'Ks']
    community_cards = ['Qs', 'Js', 'Ts', '5h', '2d']

    if is_nuts(hole_cards, community_cards):
        print("You have the nuts!")
    else:
        print("You don't have the nuts.")

    my = np.array([[0, 12], [0, 11]])
    # public = np.array([[1, 9], [2, 3], [1, 4], [2, 4], [1, 12]])
    public = np.array([[0, 10], [0, 9], [0, 8], [1, 3], [2, 0]])
    oppo = get_hands_with_win_rate_section(0.5, 1, np.vstack((public, my)))
    print(np.vstack((public, my)).shape)
    print(my_hands_win_rate(my, public, oppo))


if __name__ == '__main__':
    example()
