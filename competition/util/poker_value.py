"""
@File   : poker_value.py
@Desc   : 评估牌值大小
@Author : gql
@Date   : 2023/3/27 17:16
"""
import numpy as np
import competition.util.poker_util as poker_util


def best_cards(hand_cards, public_cards):
    """
    返回最大牌力值，并判断最大牌型是否为公共牌
    """
    all_cards = np.vstack((hand_cards, public_cards))
    flag, value = is_royal_straight_flush(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 9)
    flag, value = is_straight_flush(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 8)
    flag, value = is_four_kind(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 7)
    flag, value = is_full_house(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 6)
    flag, value = is_flush(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 5)
    flag, value = is_straight(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 4)
    flag, value = is_three_kind(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 3)
    flag, value = is_two_pairs(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 2)
    flag, value = is_one_pair(all_cards)
    if flag:
        return value, max_is_public_cards(value, public_cards, 1)
    else:
        # 高牌
        all_cards = all_cards[all_cards[:, 1].argsort()]  # 按点数排序
        all_cards = np.flip(all_cards, axis=0)
        cards = all_cards[:5]
        nums = cards[:, 1]
        value = encode_card_value(nums, 0)
        return value, max_is_public_cards(value, public_cards, 0)


def is_royal_straight_flush(all_cards):
    """
    判断能否组成皇家同花顺，如果是则同时会返回其最大牌力值

    :param all_cards: 公共牌（可以为3，4或5张）和手牌（2张）
    :return: flag表示能否组成皇家同花顺，value表示其最大牌值（只有flag=True时才有效）
    """
    royal_num = np.array([8, 9, 10, 11, 12])
    card_suits = all_cards[all_cards[:, 0].argsort()]  # 按照第一列（花色）排序
    # 1. 先找同花
    max_flush, start_index, max_count = extract_most_count(card_suits, col=0)
    if max_count < 5:
        return False, 0x0
    # 2. 在同花中是否包含10到A
    in_royal_num = np.isin(royal_num, max_flush[:, 1])
    if in_royal_num.all():
        return True, 0x9EDCBA  # 皇家同花顺的牌力值为0xA00000
    else:
        return False, 0x0


def is_straight_flush(all_cards):
    """
    判断能否组成同花顺（不包含皇家同花顺）并返回牌力值

    :param all_cards: 公共牌（可以为3，4或5张）和手牌（2张）
    :return: flag表示能否组成同花顺，value表示其最大牌值（只有flag=True时才有效）
    """
    card_suits = all_cards[all_cards[:, 0].argsort()]  # 按照第一列（花色）排序
    # 1. 先找同花（同一花色中不可能存在2张点数相同的牌）
    # max_flush, max_count = extract_most_count_flush(card_suits)
    max_flush, start_index, max_count = extract_most_count(card_suits, col=0)
    if max_count < 5:
        return False, 0x0
    # 2. 再找顺子
    max_flush = max_flush[max_flush[:, 1].argsort()]  # 按点数排序
    flag = True
    for i in range(max_count - 1, 3, -1):  # 即使7张全是同花，顺子至少也要从倒数第3张开始
        j = -1
        for j in range(i, i - 5, -1):
            if max_flush[j, 1] != max_flush[j - 1, 1] + 1:
                flag = False
                break
        if j == 0:
            flag = True
            break
    # 特殊情况：A2345
    num = [12, 3, 2, 1, 0]
    isin = np.isin(num, max_flush[:, 1])
    if isin.all():
        flag = True
        return flag, 0x854321  # 把A当作1处理
    # 特殊情况 皇家同花顺（不算在同花顺之内）
    if flag is True and is_royal_straight_flush(all_cards)[0]:
        flag = False
    # 3. 评估牌力值（A2345的最大牌已在前面找出）
    if flag:
        # max_card = max_flush[4, :]
        nums = max_flush[-5:, 1]
        nums = np.sort(nums)
        nums = np.flip(nums)  # 从大到小排序
        value = encode_card_value(nums, 8)
        return flag, value
    else:  # 只有为皇家同花顺时才会执行这段else（不确定）
        return flag, 0x0


def is_four_kind(all_cards):
    """
    判断能否组成四条，如果是则同时会返回其最大牌力值

    :param all_cards: 公共牌（可以为3，4或5张）和手牌（2张）
    :return: flag==True表示所有牌面是否能组成四条，value表示其最大牌值（只有flag=True时才有效）
    """
    # 判断哪种点数出现次数最多
    all_cards = all_cards[all_cards[:, 1].argsort()]  # 按点数排序
    most_cards, start_index, most_count = extract_most_count(all_cards, 1)
    if most_count == 4:  # 能组成四条
        high_cards = np.delete(all_cards, range(start_index, start_index + most_count), axis=0)
        # cards = all_cards[start_index:start_index + most_count]
        cards = np.vstack((most_cards, high_cards[-1]))
        nums = cards[:, 1]
        value = encode_card_value(nums, 7)
        return True, value
    else:  # 不能组成四条
        return False, 0x0


def is_full_house(all_cards):
    """
    判断能否组成葫芦
    """
    all_cards = all_cards[all_cards[:, 1].argsort()]  # 按点数排序
    all_cards = np.flip(all_cards, axis=0)  # 按行翻转数组（即按点数从大到小排序）
    most_cards, start_index, most_count = extract_most_count(all_cards, col=1)
    if most_count == 3:  # 已选出最大的三条
        # 判断剩余牌面中是否包含一个对子
        other_cards = np.delete(all_cards, range(start_index, start_index + most_count), axis=0)
        most_cards2, start_index2, most_count2 = extract_most_count(other_cards, col=1)
        if most_count2 >= 2:
            cards = np.vstack((most_cards, most_cards2[:2]))  # 如果有两组三条，较小的那组只取两张
            nums = cards[:, 1]
            value = encode_card_value(nums, 6)
            return True, value
        else:
            return False, 0x0

    else:
        return False, 0x0


def is_flush(all_cards):
    """
    判断能否组成同花
    """
    # 找出同一花色最多的牌
    all_cards = all_cards[all_cards[:, 0].argsort()]  # 按照第一列（花色）排序
    most_cards, start_index, most_count = extract_most_count(all_cards, col=0)
    # 从这些牌中选出最大的5张牌组成同花
    if most_count >= 5:
        # 按点数排序
        most_cards = most_cards[most_cards[:, 1].argsort()]
        nums = np.flip(most_cards[-5:, 1])  # 取最后5张牌
        value = encode_card_value(nums, 5)
        return True, value
    else:
        return False, 0x0


def is_straight(all_cards):
    """
    判断能否组成顺子
    """
    all_cards = all_cards[all_cards[:, 1].argsort()]
    # 去除重复点数的牌
    unique_cards = all_cards[0]
    for i in range(1, all_cards.shape[0]):
        if all_cards[i, 1] == all_cards[i - 1, 1]:
            continue
        else:
            unique_cards = np.vstack((unique_cards, all_cards[i]))
    # 点数不重复的牌的数量<5，一定不能组成顺子
    if unique_cards.shape[0] < 5:
        return False, 0x0
    flag = True
    value = 0x0
    unique_cards = np.flip(unique_cards, axis=0)  # 从大到小排序
    # 判断并找出最大的顺子
    for i in range(unique_cards.shape[0] - 4):
        j = 0
        for j in range(5):
            # j可能取值为0，1，2，3，4，正常结束循环（未执行break）时，j的值是4而不是5
            # j只需要执行到3
            if j == 4:  # 只是用于判断j==3是否正常执行了，没有执行第二个break
                j = 5
                break
            if unique_cards[i + j, 1] != unique_cards[i + j + 1, 1] + 1:
                i = i + j
                flag = False
                break
        if j == 5:
            flag = True
            nums = unique_cards[i:i + 5, 1]
            value = encode_card_value(nums, 4)
            break
    if flag:
        return True, value
    else:
        return False, 0x0


def is_three_kind(all_cards):
    """
    判断能否组成三条（不能判断最大牌型是否为三条）

    能区分三条与四条，葫芦（即当all_cards能组成葫芦时返回False），不能区分三条和同花、顺子
    """
    all_cards = all_cards[all_cards[:, 1].argsort()]  # 按点数排序
    all_cards = np.flip(all_cards, axis=0)  # 按行翻转数组（即按点数从大到小排序）
    most_cards, start_index, most_count = extract_most_count(all_cards, col=1)
    if most_count == 3:  # 已选出最大的三条
        other_cards = np.delete(all_cards, range(start_index, start_index + most_count), axis=0)
        flag = True
        # 需要满足剩下的牌两两不相等
        for i in range(other_cards.shape[0] - 1):
            if other_cards[i, 1] == other_cards[i + 1, 1]:
                flag = False
                break
        if flag:
            cards = np.vstack((most_cards, other_cards[:2]))
            # cards.astype(np.int)
            nums = cards[:, 1]
            value = encode_card_value(nums, 3)
            return True, value
        else:
            return False, 0x0
    else:
        return False, 0x0


def is_two_pairs(all_cards):
    all_cards = all_cards[all_cards[:, 1].argsort()]  # 按点数排序
    all_cards = np.flip(all_cards, axis=0)  # 按行翻转数组（即按点数从大到小排序）
    most_cards, start_index, most_count = extract_most_count(all_cards, col=1)
    if most_count == 2:  # 已选出最大的对子
        other_cards = np.delete(all_cards, range(start_index, start_index + most_count), axis=0)
        most_cards2, start_index2, most_count2 = extract_most_count(other_cards, col=1)
        if most_count2 == 2:  # 选出第二大的对子
            cards = np.vstack((most_cards, most_cards2))
            other_cards = np.delete(other_cards, range(start_index2, start_index2 + most_count2), axis=0)
            # 再选出最大的单牌
            cards = np.vstack((cards, other_cards[0]))
            # 计算牌力值
            nums = cards[:, 1]
            value = encode_card_value(nums, 2)
            return True, value
        else:
            return False, 0x0
    else:
        return False, 0x0


def is_one_pair(all_cards):
    all_cards = all_cards[all_cards[:, 1].argsort()]  # 按点数排序
    all_cards = np.flip(all_cards, axis=0)  # 按行翻转数组（即按点数从大到小排序）
    most_cards, start_index, most_count = extract_most_count(all_cards, col=1)
    if most_count == 2:
        other_cards = np.delete(all_cards, range(start_index, start_index + most_count), axis=0)
        flag = True
        # 需要满足剩下的牌两两不相等
        for i in range(other_cards.shape[0] - 1):
            if other_cards[i, 1] == other_cards[i + 1, 1]:
                flag = False
                break
        if flag:
            cards = np.vstack((most_cards, other_cards[:3]))
            nums = cards[:, 1]
            value = encode_card_value(nums, 1)
            return True, value
        else:
            return False, 0x0
    else:
        return False, 0x0


def extract_most_count(all_cards, col=0):
    """
    从按花色（或按点数）排序的牌面中，找出出现次数最多的花色（或点数）。

    如果同时有多组花色（或点数）相同的牌，则返回位置最靠前的那组牌。
    :param all_cards: 待提取的牌面
    :param col: col==0表示提取出现次数最多的花色，col==1表示提取次数最多的点数
    """
    # note 下面代码相当于找出有序数组中出现次数最多的元素
    cur_count = 1
    most_count = 0
    end_index = 0  # 数组中出现次数最多的元素的末尾下标
    i = 0
    for i in range(all_cards.shape[0] - 1):
        if all_cards[i, col] == all_cards[i + 1, col]:
            cur_count += 1
        else:
            if cur_count > most_count:
                most_count = cur_count
                end_index = i
            cur_count = 1
    if cur_count > most_count:
        most_count = cur_count
        end_index = i + 1
    start_index = end_index - most_count + 1
    most_cards = all_cards[start_index:end_index + 1, :]
    return most_cards, start_index, most_count


def encode_card_value(nums, level):
    """
    给出有序的5张牌的点数和牌型编码出牌力值（value）

    例如cards=[1,2,3,4,5] level=8（同花顺），编码结果value=0x854321
    :param nums: 5张牌的点数（一维数组）
    :param level: 牌型
    :return: 牌力值
    """
    # nums = np.sort(nums)
    # nums = np.flip(nums)  # 从大到小排序
    value = level
    nums = nums.astype(np.int32)
    for i in range(len(nums)):
        value = value << 4
        value += (nums[i] + 2)  # 点数2对应0，点数3对应1...点数A对应12
    return value


def max_is_public_cards(value, public_cards, level):
    """
    判断公共牌的牌力值是否与value相等
    """
    # 公共牌不是5张时，直接返回False
    if public_cards.shape[0] != 5:
        return False
    public_cards = public_cards[public_cards[:, 1].argsort()]
    public_cards = np.flip(public_cards, axis=0)  # 从大到小排序
    nums = public_cards[:, 1]
    public_cards_value = encode_card_value(nums, level)
    if public_cards_value == value:
        return True
    else:
        return False


def max_value_by_public_cards(public_cards, exclude_cards):
    """
    当前公共牌所能组成的最大牌型

    :param public_cards: 参与组牌的公共牌
    :param exclude_cards: 排除在外的牌（例如自己的手牌）
    :return:
    """

    """
    示例
    9 9 8
    同花顺、同花、顺子都不可能
    最小牌型是两对
    9 9 8  9 9 四条
    9 9 8  9 8 葫芦
    9 9 8  9 X(除9,8之外) 三条
    9 9 8  8 8 三条
    9 9 8  X X(除9,8之外)两对
    9 9 8  8 X(除9,8之外) 两对
    9 9 8  X Y(除9,8之外) 一对

    3 4 6
    可以组成同花顺，同花，顺子，三条，两对

    3张公共牌的情况
    1. 公共牌中有除了10 J Q K A中其他牌，则不可能组成皇家同花顺（单独判断是否能组成皇家同花顺，后面不再考虑皇家同花顺）
    3. 公共牌中有三条，可以组成四条、葫芦、三条，不可能组成同花顺、同花、顺子、两对、一对、高牌
    4. 公共牌中有对子，可以组成四条、葫芦、三条、两对、一对，不可能组成同花顺、同花、顺子、高牌
    5. 公共牌中都是单张，可以组成同花顺、同花、顺子、三条、两对、一对、高牌，不可能组成四条、葫芦

    公共牌-->同花顺的概率，同花顺牌力值
         -->四条 如1.34%，1000   牌力值如何确定？每种牌型的概率*对应的牌力值应该接近于某个定值（类似于期望的概念）？或者是其他什么方法？
         -->葫芦 将每种牌型再细分成子牌型，如葫芦分成三条是2-6的葫芦、6-10的葫芦和J-A的葫芦？
         -->同花
         -->顺子...
    对对手手牌的建模：绝大多数情况下注量越大，牌型越好。需要一个具体的公式/模型表示下注量与牌力值的关系
    皇家同花顺：1种
    同花顺：只需考虑最大牌 2-A，13种
    四条：2-A，13种
    葫芦：2-A，13种
    同花：考虑最大的两张牌 C^2_13=78种（C^3_13=286种， C^4_13=715种）
    顺子：只需考虑最大牌 2-A，13种
    三条： 2-A，13种
    两对：只考虑两个对子的大小 C^2_13=78
    一对：只考虑对子和最大单张的牌 C^2_13=78
    高牌：只考虑最大的两张牌 C^2_13=78
    共计：378种（C^2_13），1210（C^3_13），2926（C^4_13）
    如果有3张公共牌，还需进行遍历13*13=169次（代价太大）；如果有4张公共牌，还需进行13次遍历（代价不大，可以接受）
    """
    pass


def high_value_prob(hand_cards, public_cards, count=2000):
    hand_cards_count = hand_cards.shape[0]
    public_cards_count = public_cards.shape[0]
    hands_value, _ = best_cards(hand_cards, public_cards)
    print(hands_value)
    score = 0
    for _ in range(count):
        remain_public_cards = poker_util.deal_cards(7 - hand_cards_count - public_cards_count,
                                                    np.vstack((hand_cards, public_cards)))
        all_public_cards = np.vstack((remain_public_cards, public_cards))
        value, flag = best_cards(hand_cards, all_public_cards)
        # oppo_hands=poker_util.deal_cards(2,)
        if hands_value < value:
            score += 1
    print("score:", score)
    return score / count


if __name__ == '__main__':
    a = 0x9
    a = a << 4
    a += 11
    print(a, 0x9B)
    b = np.array([1, 2, 3, 4, 5])
    c = encode_card_value(b, 8)
    print(c)
    print(0x854321)

    print("测试同花顺")
    cs = np.array([[1, 6], [1, 9], [1, 11], [1, 10], [1, 7], [1, 8], [2, 5]])
    f, v = is_straight_flush(cs)
    print(f, v, 0x8DCBA9)

    cs = np.array([[3, 9], [1, 9], [2, 9], [1, 10], [2, 10], [3, 10], [0, 3]])
    cs = cs[cs[:, 1].argsort()]
    print(cs)
    cs = np.flip(cs, axis=0)
    print(cs)
    a, b, c = extract_most_count(cs, col=1)
    print("a:", a, "b:", b, "c:", c)

    print("测试四条")
    cs = np.array([[3, 9], [1, 9], [2, 8], [0, 9], [2, 9], [1, 8], [0, 3]])
    f, v = is_four_kind(cs)
    print(f, v, 0x7BBBBA)

    print("测试葫芦")
    cs = np.array([[3, 7], [1, 7], [2, 8], [0, 8], [2, 9], [1, 9], [0, 8]])
    f, v = is_full_house(cs)
    print(f, v, 0x6AAABB)

    print("测试同花")
    cs = np.array([[3, 7], [3, 4], [3, 8], [3, 2], [3, 9], [1, 9], [0, 8]])
    f, v = is_flush(cs)
    print(f, v, 0x5BA964)

    print("测试顺子")
    cs = np.array([[3, 7], [2, 9], [1, 8], [3, 5], [3, 6], [1, 10], [0, 8]])
    f, v = is_straight(cs)
    print(f, v, 0x4A9876)

    print("测试三条")
    cs = np.array([[3, 8], [2, 9], [1, 8], [3, 7], [3, 6], [1, 10], [0, 8]])
    f, v = is_three_kind(cs)
    print(f, v, 0x3AAACB)

    print("测试两对")
    cs = np.array([[3, 8], [2, 9], [1, 8], [3, 9], [3, 10], [1, 10], [0, 4]])
    f, v = is_two_pairs(cs)
    print(f, v, 0x2CCBBA)

    print("测试一对")
    cs = np.array([[3, 8], [2, 1], [1, 8], [3, 2], [3, 3], [1, 10], [0, 4]])
    f, v = is_one_pair(cs)
    print(f, v, 0x1AAC65)

    print("测试best_cards")
    a = np.array([[1, 2], [1, 3]])
    b = np.array([[1, 8], [1, 9], [1, 10], [1, 11], [1, 12]])
    v, f = best_cards(a, b)
    print(f, v, 0x9EDCBA)

    print("测试随机数组")
    a = np.random.randint(0, 4, (2, 1))
    b = np.random.randint(0, 13, (2, 1))
    c = np.hstack((a, b))
    print(a)
    print(b)
    print(c)

    # print("10000组随机数据测试")
    # for _ in range(10000):
    #     a = poker_util.deal_one_card()
    #     b = np.vstack((a, poker_util.deal_one_card(a)))  # 手牌
    #     c = poker_util.deal_one_card(b)  # 公共牌
    #     for _ in range(4):
    #         d = poker_util.deal_one_card(c)
    #         c = np.vstack((c, d))
    #     v, f = best_cards(b, c)
    #     print("b:", b)
    #     print("c:", c)
    #     print("value:", v, "flag:", f, end="\n\n")

    # for _ in range(1000):
    #     hands = poker_util.deal_cards(2)
    #     public_cards = poker_util.deal_cards(5, hands)
    #     v, f = best_cards(hands, public_cards)
    #     print("hands:", hands)
    #     print("public:", public_cards)
    #     print("value:", v, "flag:", f, end='\n\n')

    print("两头顺子听牌")
    my = np.array([[1, 2], [3, 3]])
    public = np.array([[1, 4], [2, 5], [0, 12]])
    print(high_value_prob(my, public, 2000))
    print("自定义数据")
    my = np.array([[0, 12], [3, 12]])
    # public = np.array([[1, 12], [2, 12], [1, 11], [1, 9], [1, 10]])
    public = np.array([[1, 12], [2, 12], [1, 11]])
    print(high_value_prob(my, public, 20000))
    print("同花顺子听牌")
    my = np.array([[1, 2], [1, 3]])
    public = np.array([[1, 4], [2, 5], [1, 12]])
    print(high_value_prob(my, public, 2000))
