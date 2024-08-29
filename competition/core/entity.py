"""
@File   : entity.py
@Desc   :
@Author : gql
@Date   : 2023/4/2 18:52
"""
import math

import numpy as np

from competition.core import constants
from competition.util.common_util import print_exception
from competition.util.poker_util import extract_cards_from_reply


class Player:
    def __init__(self, name):
        self.name = name
        # self.strategy = strategy
        # 仅在一局开始时需要更新的变量
        self.identify = "SMALLBLIND"
        self.hand_cards = np.empty((0, 2))
        self.hand_cards_raw = ''
        # 每做出一次动作就需要更新的变量
        self.operation = np.zeros((1, 2), dtype="int")  # 记录玩家单局内的所有行为
        self.all_operation = np.zeros((1, 2), dtype="int")  # 记录玩家整个对弈过程中的所有行为
        self.game_chip = 0  # 玩家本局总下注量
        # 每进入下一阶段就要更新的变量 or 每做出一个动作就需要更新的变量
        self.stage_chip = 0  # 玩家本阶段下注量
        # 每进入下一阶段需要更新的变量
        # self.stage = preflop_stage  # 初始处于preflop阶段（此变量作用不大）
        # 一局结束时才需要更新的变量
        self.total_earn_chip = 0  # 玩家累计赢得的筹码量（已对弈的所有局的总和）
        # 一局开始或结束时需要更新的变量
        # self.is_current_play = constants.UNCERTAIN_PLAY_THIS  # 玩家是否继续本局对弈

        # 跟策略有关的变量
        # self.hands_win_rate = 0  # 预估的当前手牌胜率

    def init_identity(self, identity, hand_cards=None, hands_cards_raw=None):
        self.operation = np.zeros((1, 2), dtype="int")
        if identity == "SMALLBLIND":
            self.identify = constants.SMALLBLIND
            self.stage_chip = 50
            self.game_chip = 50
        elif identity == "BIGBLIND":
            self.identify = constants.BIGBLIND
            self.stage_chip = 100
            self.game_chip = 100
        else:
            msg = "玩家" + self.name + "获得未知的身份信息"
            print_exception(self.init_identity, msg)
            self.identify = constants.SMALLBLIND
            self.stage_chip = 50
            self.game_chip = 50
        if hand_cards is not None:
            self.hand_cards = hand_cards
            self.hand_cards_raw = hands_cards_raw
            # 手牌胜率预测，一种方法是采用蒙特卡洛模拟，另一种是查表，此处为蒙特卡洛模拟
            # self.hands_win_rate = evaluate_hands_win_rate(hand_cards, 2000)  # 手牌胜率预测

    def player_action(self, action_type, chip=0):
        """
        玩家采取的某种行为（如raise行为）及其对应的筹码量，仅涉及Player类的相关属性，不修改整个对弈环境
        """
        if action_type == constants.player_allin_action:
            # allin行为
            past_stage_chip = self.game_chip - self.stage_chip
            self.game_chip = constants.total_chip
            self.stage_chip = constants.total_chip - past_stage_chip
        elif action_type == constants.player_fold_action:
            # fold行为
            chip = -1
        elif action_type == constants.player_check_action:
            # check行为
            pass
            # chip = 0
        else:
            # call和raise行为
            last_stage_chip = self.stage_chip
            self.stage_chip = chip
            self.game_chip += chip - last_stage_chip
        self.operation = np.vstack((self.operation, (action_type, chip)))
        self.all_operation = np.vstack((self.all_operation, (action_type, chip)))

    def add_chip(self, chip):
        """
        下注筹码
        """
        self.stage_chip += chip
        self.game_chip += chip

    # def update_is_current_play(self, is_current_play):
    #     """
    #     通过此方法更新is_current_play
    #     """
    #     if is_current_play == constants.PLAY_THIS:
    #         self.is_current_play = constants.PLAY_THIS
    #     elif is_current_play == constants.NOT_PLAY_THIS:
    #         self.is_current_play = constants.NOT_PLAY_THIS
    #     elif is_current_play == constants.UNCERTAIN_PLAY_THIS:
    #         self.is_current_play = constants.UNCERTAIN_PLAY_THIS

    def enter_next_stage(self):
        """
        进入下一个阶段将要更新的信息
        """
        self.stage_chip = 0

    # 此方法和init_identify()方法重复
    def reset(self):
        # 仅在一局开始时需要更新的变量
        self.identify = "SMALLBLIND"
        self.hand_cards = np.empty((0, 2))
        # 每做出一次动作就需要更新的变量
        self.operation = np.zeros((1, 2), dtype="int")  # 记录玩家单局内的所有行为
        self.game_chip = 0  # 玩家本局总下注量
        # 每进入下一阶段就要更新的变量 or 每做出一个动作就需要更新的变量
        self.stage_chip = 0  # 玩家本阶段下注量
        # 每进入下一阶段需要更新的变量
        # self.stage = preflop_stage  # 初始处于preflop阶段（此变量作用不大）
        # 一局结束时才需要更新的变量
        self.total_earn_chip = 0  # 玩家累计赢得的筹码量（已对弈的所有局的总和）
        # self.is_current_play = constants.UNCERTAIN_PLAY_THIS


class Game:
    """
    单局游戏环境
    """

    def __init__(self, catzzz: Player, opponent: Player):
        self.catzzz = catzzz
        self.opponent = opponent
        # self.strategy = strategy
        self.total_chip = 0  # 一局游戏下注量
        self.stage = constants.preflop_stage  # 当前游戏阶段
        self.public_cards = np.empty((0, 2))  # 公共牌 [花色，点数]
        self.last_reply = None  # 服务端最后一次发来的指令

    # def parse_server_cmd(self, reply: str):
    #     """
    #     解析服务端发来的所有指令
    #
    #     :return: 值为301表示我方需要行动，值为302表示不需要行动
    #     """
    #     # catzzz_action_flag = 0  # 我方是否需要采取行动的标志
    #     self.last_reply = reply
    #     if "flop" in reply or "turn" in reply or "river" in reply:  # 阶段类指令
    #         self.parse_stage_cmd(reply)
    #         if (self.stage == constants.preflop_stage and self.catzzz.identify == constants.SMALLBLIND) \
    #                 or (self.stage == constants.flop_stage and self.catzzz.identify == constants.BIGBLIND) \
    #                 or (self.stage == constants.turn_stage and self.catzzz.identify == constants.BIGBLIND) \
    #                 or (self.stage == constants.river_stage and self.catzzz.identify == constants.BIGBLIND):
    #             catzzz_action_flag = constants.take_action  # 需要采取行动
    #         else:
    #             catzzz_action_flag = constants.no_take_action
    #     elif "earnChips" in reply:  # earnChips指令
    #         reply_split = reply.split(" ")
    #         earn_chip = int(reply_split[1])
    #         # allin行为不需要river_stage阶段，直接比牌
    #         if self.stage == constants.river_stage \
    #                 and self.catzzz.stage_chip == self.opponent.stage_chip:
    #             catzzz_action_flag = constants.no_take_action
    #         else:
    #             catzzz_action_flag = constants.game_over
    #         self.stage = constants.earn_chip_stage
    #         self.catzzz.total_earn_chip += earn_chip
    #         self.opponent.total_earn_chip -= earn_chip
    #         # catzzz_action_flag = constants.game_over
    #     elif "oppo_hands" in reply:  # oppo_hands指令
    #         # todo 接收并更新对手手牌
    #         self.stage = constants.show_oppo_card
    #         catzzz_action_flag = constants.game_over
    #     else:  # 对手行为类指令
    #         oppo_action = self.parse_client_cmd(reply, self.opponent, self.catzzz)
    #         # 下面的代码主要是用于判断我方是否需要做出行动
    #         print("oppo_action", oppo_action)
    #         if oppo_action == constants.player_fold_action:
    #             catzzz_action_flag = constants.no_take_action
    #         elif oppo_action == constants.player_call_action:
    #             # 小盲注在preflop阶段跟注时，大盲注仍可以下注，其他阶段跟注，对方在本阶段都不能再下注
    #             if self.stage == constants.preflop_stage and self.catzzz.identify == "BIGBLIND":
    #                 catzzz_action_flag = constants.take_action
    #             else:
    #                 catzzz_action_flag = constants.no_take_action
    #         else:
    #             catzzz_action_flag = constants.take_action
    #     self.last_reply = reply
    #     return catzzz_action_flag

    def parse_server_cmd_1(self, reply: str):
        """
        解析服务端发来的所有指令
        :param reply:
        :return: 游戏开始标志，我方是否需要采取行动标志
        """
        # catzzz_action_flag = 0  # 我方是否需要采取行动的标志
        catzzz_first_action_flag = 0  # 我方是否需要在本阶段内先下注，只在刚进入某一阶段内判断
        game_flag = 0
        self.last_reply = reply
        if "flop" in reply or "turn" in reply or "river" in reply:  # 阶段类指令
            game_flag = self.parse_stage_cmd_1(reply)  # 是否开始了新的一局游戏
            if (self.stage == constants.preflop_stage and self.catzzz.identify == constants.SMALLBLIND) \
                    or (self.stage == constants.flop_stage and self.catzzz.identify == constants.BIGBLIND) \
                    or (self.stage == constants.turn_stage and self.catzzz.identify == constants.BIGBLIND) \
                    or (self.stage == constants.river_stage and self.catzzz.identify == constants.BIGBLIND):
                catzzz_action_flag = constants.take_action  # 需要采取行动
                catzzz_first_action_flag = 1
            else:
                catzzz_action_flag = constants.no_take_action
        elif "earnChips" in reply:  # earnChips指令
            # 进入earnChips阶段之后，stage_chip就不清零了
            # if self.catzzz.operation[-1, 0] != constants.player_fold_action and \
            #         self.opponent.operation[-1, 0] != constants.player_fold_action and \
            #         self.stage == constants.river_stage:
            #     self.opponent.player_action(constants.player_call_action, self.catzzz.stage_chip)
            reply_split = reply.split(" ")
            try:
                earn_chip = int(reply_split[1])
            except Exception as e:
                print("earn 错误")
                earn_chip = 100
            if self.opponent.game_chip < math.fabs(earn_chip):
                self.opponent.player_action(constants.player_call_action, self.catzzz.stage_chip)

            self.stage = constants.earn_chip_stage
            self.catzzz.total_earn_chip += earn_chip
            print("earn total chip:", self.catzzz.total_earn_chip)
            self.opponent.total_earn_chip -= earn_chip
            catzzz_action_flag = constants.no_take_action
        elif "oppo_hands" in reply:  # oppo_hands指令
            self.stage = constants.show_oppo_card
            cards, cards_raw = extract_cards_from_reply(reply)
            self.opponent.hand_cards = cards
            self.opponent.hand_cards_raw = cards_raw
            catzzz_action_flag = constants.no_take_action
        else:  # 对手行为类指令
            oppo_action, oppo_action_chip = self.parse_client_cmd_1(reply, self.opponent, self.catzzz)
            if oppo_action == constants.player_fold_action:
                catzzz_action_flag = constants.no_take_action
            elif oppo_action == constants.player_call_action:
                # 小盲注在preflop阶段跟注时，大盲注仍可以下注
                if self.stage == constants.preflop_stage and self.catzzz.identify == "BIGBLIND":
                    catzzz_action_flag = constants.take_action
                else:
                    catzzz_action_flag = constants.no_take_action
            else:
                catzzz_action_flag = constants.take_action
        self.last_reply = reply
        self.total_chip = self.catzzz.game_chip + self.opponent.game_chip
        return game_flag, catzzz_action_flag, catzzz_first_action_flag

    # raise 200, call, check, fold, allin

    def parse_stage_cmd_1(self, reply: str):
        """
        解析服务端发来的阶段类指令（不包括earnChips和oppo_hands）并更新信息
        """
        reply_split = reply.split("|")
        cards, cards_raw = extract_cards_from_reply(reply)
        catzzz = self.catzzz
        opponent = self.opponent
        start_game_flag = constants.game_keeping  # 为304表示开始新的一局游戏
        if reply_split[0] == "flop":
            self.stage = constants.flop_stage
            self.public_cards = cards
        elif reply_split[0] == "turn":
            self.stage = constants.turn_stage
            self.public_cards = np.vstack((self.public_cards, cards))
        elif reply_split[0] == "river":
            self.stage = constants.river_stage
            self.public_cards = np.vstack((self.public_cards, cards))
        if reply_split[0] == "preflop":
            # preflop指令表示开始新的一局游戏，结束上一局游戏
            self.stage = constants.preflop_stage
            catzzz.init_identity(reply_split[1], cards, cards_raw)
            start_game_flag = constants.game_start
            # 确认对手身份
            if reply_split[1] == "SMALLBLIND":
                opponent.init_identity(constants.BIGBLIND, None)
            else:
                opponent.init_identity(constants.SMALLBLIND, None)
            self.public_cards = np.empty((0, 2), int)
            self.total_chip = catzzz.game_chip + opponent.game_chip
        else:
            # 进入其他阶段时，可能会隐藏了一次对手call行为
            if catzzz.stage_chip > opponent.stage_chip:
                opponent.player_action(constants.player_call_action, catzzz.stage_chip)
            catzzz.enter_next_stage()
            opponent.enter_next_stage()
        self.total_chip = self.catzzz.game_chip + self.opponent.game_chip
        return start_game_flag

    def parse_client_cmd_1(self, cmd: str, sender: Player, opponent: Player):
        """
        解析属于客户端（包括对手和自己）产生的指令（修改sender的实体类信息）
        :return: 指令类型、下注量（一般表示加注到，跟注到多少筹码量）
        """
        cmd_split = cmd.split(" ")
        client_action = 0
        client_chip = 0
        if cmd_split[0] == "call":
            sender.player_action(constants.player_call_action, opponent.stage_chip)
            # self.total_chip = self.catzzz.game_chip + self.opponent.game_chip
            client_action = constants.player_call_action
            client_chip = opponent.stage_chip
            # return constants.player_call_action, opponent.stage_chip
        elif cmd_split[0] == "check":
            sender.player_action(constants.player_check_action)
            client_action = constants.player_check_action
            client_chip = sender.stage_chip
            # return constants.player_check_action, sender.stage_chip
        elif cmd_split[0] == "raise":
            # 服务端有bug，raise 500\ 这样的指令并不会报错，而是当作raise 500处理
            chip = 0
            x = [str(x) for x in range(0, 10)]  # 产生字符0-9
            for i in cmd_split[1]:
                if i in x:
                    chip = chip * 10 + int(i)
                else:
                    break
            if chip < 50:
                print_exception(self.parse_client_cmd_1, '无法解析的客户端指令')
                client_action = constants.player_unknown_action
                client_chip = -1
                # return constants.player_unknown_action, -1
            else:
                sender.player_action(constants.player_raise_action, chip)
                client_action = constants.player_raise_action
                client_chip = chip
                # return constants.player_raise_action, chip
        elif cmd_split[0] == "allin":
            sender.player_action(constants.player_allin_action)
            client_action = constants.player_allin_action
            client_chip = constants.total_chip
            # return constants.player_allin_action, constants.total_chip
        elif cmd_split[0] == "fold":
            sender.player_action(constants.player_fold_action, -1)
            # client_action = constants.player_fold_action
            client_action = constants.player_fold_action
            client_chip = -1
            # return constants.player_fold_action, -1
        else:
            print_exception(self.parse_client_cmd, "未知的客户端指令")
            client_action = constants.player_unknown_action
            client_chip = -1
            # return constants.player_unknown_action, -1
        self.total_chip = sender.game_chip + opponent.game_chip
        return client_action, client_chip

    # def parse_stage_cmd(self, reply: str):
    #     """
    #     解析服务端发来的阶段类指令（不包括earnChips和oppo_hands）并更新信息
    #     """
    #     reply_split = reply.split("|")
    #     cards = extract_cards_from_reply(reply)
    #     catzzz = self.catzzz
    #     opponent = self.opponent
    #
    #     # if reply_split[0] == "oppo_hands":
    #     #     pass
    #     # 阶段类指令
    #     if reply_split[0] == "flop":
    #         self.stage = constants.flop_stage
    #         self.public_cards = cards  # 添加公共牌（此处可以直接用'='）
    #     elif reply_split[0] == "turn":
    #         self.stage = constants.turn_stage
    #         self.public_cards = np.vstack((self.public_cards, cards))
    #     elif reply_split[0] == "river":
    #         self.stage = constants.river_stage
    #         self.public_cards = np.vstack((self.public_cards, cards))
    #     # preflop阶段需要确认身份，确认手牌，下大小盲注
    #     if reply_split[0] == "preflop":
    #         # catzzz.hand_cards = cards
    #         self.stage = constants.preflop_stage
    #         catzzz.init_identity(reply_split[1], cards)
    #         # 确认对手身份
    #         if reply_split[1] == "SMALLBLIND":
    #             opponent.init_identity(constants.BIGBLIND, None)
    #         else:
    #             opponent.init_identity(constants.SMALLBLIND, None)
    #         self.total_chip = catzzz.game_chip + opponent.game_chip
    #     else:
    #         # 进入其他阶段时，可能会隐藏了一次对手call行为
    #         if catzzz.stage_chip > opponent.stage_chip:
    #             opponent.player_action(constants.player_call_action, catzzz.stage_chip)
    #         catzzz.enter_next_stage()
    #         opponent.enter_next_stage()

    def print_info(self):
        """
        输出当前局面信息，己方信息和对手信息

        ----------------对局信息-------------------------------------
        | stage        | river                                    |
        | public cards | 0 A, 1 T, 0 8, 3 9,  3 4 |
        ----------------己方信息-------------------------------------
        | our identity    |   SMALLBLIND      |
        | our hand cards  |   club 8, diam 4  |
        | our stage chip  |   1000            |
        | our game chip   |   3600            |
        ----------------对方信息-----------------
        | oppo identity   |   BIGBLIND        |
        | oppo stage chip |   500             |
        | oppo game chip  |   3100            |
        ----------------双方行动-----------------
        | our last action |   raise 1000      |
        | oppo last action|   raise 500       |
        ---------------------------------------
        """
        # print("\n last raw reply: ", self.last_reply)
        print("----------------当前对局信息---------------------------------")
        if self.stage == "flop_strategy" or self.stage == "turn" or self.stage == "river":
            print("| stage        |      " + self.stage)
        else:
            print("| stage        |      " + self.stage)
        print("| public cards |      ", end="")
        for i in range(self.public_cards.shape[0]):
            print("[" + str(self.public_cards[i, 0]) + ", " + str(self.public_cards[i, 1]) + "] ", end="")
        print("\n----------------己方信息-------------------------------------")
        if self.catzzz.identify == "SMALLBLIND":
            print("| our identity    |   " + self.catzzz.identify)
        else:
            print("| our identity    |   " + self.catzzz.identify)
        print("| our hand cards  |   ["
              + str(self.catzzz.hand_cards[0, 0]) + ", " + str(self.catzzz.hand_cards[0, 1]) + "] ["
              + str(self.catzzz.hand_cards[1, 0]) + ", " + str(self.catzzz.hand_cards[1, 1]) + "]")
        print("| our stage chip  |   " + str(self.catzzz.stage_chip))
        print("| our game chip   |   " + str(self.catzzz.game_chip))
        print("| our earn chip   |   " + str(self.catzzz.total_earn_chip))
        print("----------------对方信息------------------")
        print("| oppo identity   |   " + self.opponent.identify)
        print("| oppo stage chip |   " + str(self.opponent.stage_chip))
        print("| oppo game chip  |   " + str(self.opponent.game_chip))
        if self.opponent.hand_cards.size > 0:
            print("| oppo hand cards |   ["
                  + str(self.opponent.hand_cards[0, 0]) + ", " + str(self.opponent.hand_cards[0, 1]) + "] ["
                  + str(self.opponent.hand_cards[1, 0]) + ", " + str(self.opponent.hand_cards[1, 1]) + "]")
        print("----------------双方行动------------------")
        print("| oppo last action|   " + str(self.opponent.operation[-1, :]))
        print("| our last action |   " + str(self.catzzz.operation[-1, :]))
        print("----------------------------------------")

    def parse_client_cmd(self, cmd: str, sender: Player, opponent: Player):
        """
        解析属于客户端（包括对手和自己）产生的指令（如call指令）（直接修改实体类信息）

        :return: 指令类型、下注量（仅raise行为）
        """
        cmd_split = cmd.split(" ")
        if cmd_split[0] == "call":
            # stage_chip = opponent.stage_chip - sender.stage_chip
            sender.player_action(constants.player_call_action, opponent.stage_chip)
            return constants.player_call_action
        elif cmd_split[0] == "check":
            sender.player_action(constants.player_check_action, 0)
            return constants.player_check_action
        elif cmd_split[0] == "raise":
            if cmd_split[1].isdigit():
                sender.player_action(constants.player_raise_action, int(float(cmd_split[1])))
            else:
                print_exception(self.parse_client_cmd, '无法解析的指令')
                sender.player_action(constants.player_fold_action)
            return constants.player_raise_action
        elif cmd_split[0] == "allin":
            sender.player_action(constants.player_allin_action)
            return constants.player_allin_action
        elif cmd_split[0] == "fold":
            sender.player_action(constants.player_fold_action, -1)
            return constants.player_fold_action
        else:
            print_exception(self.parse_client_cmd, "未知的客户端指令")
            return constants.player_unknown_action


if __name__ == '__main__':
    chip1 = 0
    cmd_split1 = "500\1"
    x1 = [str(x) for x in range(0, 10)]  # 产生字符0-9
    for i1 in cmd_split1:
        if i1 in x1:
            chip1 = chip1 * 10 + int(i1)
        else:
            break
    print(chip1)
