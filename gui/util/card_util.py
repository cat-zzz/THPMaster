"""
@File   : card_util.py
@Desc   : 通用的工具类函数
@Author : gql
@Date   : 2023/9/15 9:34
"""
import numpy as np


def print_exception(sender, msg):
    print(f'\033[0;31merror: func {sender.__name__}()-> {msg}\033[0m')


def deal_cards(num=1, excludes=None, seed=None):
    """
    随机发若干张牌
    :param num: 发牌数量
    :param excludes: 排除在外的牌
    :param seed: 随机数种子
    :return: [花色, 点数] numpy数组类型
    """
    if excludes is None:
        excludes = np.zeros((0, 2), dtype=int)
    cards = np.zeros((0, 2), dtype=int)
    for i in range(num):
        card = deal_one_card(excludes, seed)  # 随机生成一张牌
        excludes = np.vstack((excludes, card))  # 将这张牌添加到排除列表中
        cards = np.vstack((cards, card))
    return cards


def deal_one_card(excludes=None, seed=None):
    """
    随机发一张不在excludes里的牌

    :return: [花色, 点数] numpy数组类型
    """
    # 此函数优点：无需递归调用，只执行一次就能随机产生一张不在excludes里的牌
    if seed is not None:
        np.random.seed(seed)
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


def card_to_one_hot(card):
    """
    ndarray形式的card转为one hot形式
    :param card: ndarray [花色, 点数]
    :return: ndarray 一维数组
    """
    a = np.zeros(52, dtype=int)
    i = card[0] * 13 + card[1]
    a[i] = 1
    return a


def card_to_ndarray(card):
    """
    :param card: ndarray 一维数组
    :return: ndarray [花色, 点数]
    """
    num = np.where(card == 1)[0][0]
    card = np.array([num // 13, num % 13])
    return card


def card_to_num(card):
    """
    ndarray的card转为数字(0~51)
    :param card: ndarray [[花色, 点数]]
    :return: int
    """
    num = card[0] * 13 + card[1]
    return num


if __name__ == '__main__':
    c = deal_one_card()
    c = c[0]
    c1 = card_to_one_hot(c)
    c2 = card_to_ndarray(c1)
    c3 = card_to_num(c)
    print(c)
    print(c1)
    print(c2)
    print(c3)
