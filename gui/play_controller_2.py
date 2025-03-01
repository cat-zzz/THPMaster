"""
@project: THPMaster
@File   : play_controller.py
@Desc   :
@Author : gql
@Date   : 2025/2/21 10:19
"""
import sys
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout

from gui import play_ui
from gui.env import constants
from gui.env.rl_game import NoLimitHoldemGame
from gui.strategy.strategy import Strategy

catzzz_player_idx = 0
oppo_player_idx = 1
max_action_timer_num = 120


class StrategyWorker(QObject):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, strategy: Strategy):
        super().__init__()
        self.oppo_strategy = strategy

    def run(self):
        print('调用StrategyWorker，调用策略')
        time.sleep(2)
        oppo_action = self.oppo_strategy.strategy_2()
        self.finished.emit(oppo_action)
        print('发送StrategyWorker完成信号')


class NoLimitHoldemRunner(QObject):
    oppo_action_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.oppo_strategy = None
        self.oppo_strategy_worker_1 = None
        # self.oppo_strategy_thread_1 = None
        self.oppo_strategy_thread_1 = QThread()
        # 由于同一玩家可能连续行动两次（preflop到flop阶段），每一次行动对应一个worker和thread
        self.oppo_strategy_worker_2 = None
        self.oppo_strategy_thread_2 = QThread()
        # 下面是强化学习环境
        self.game = NoLimitHoldemGame()
        # 下面是PyQt
        self.timer_num = max_action_timer_num
        self.my_timer = QTimer()  # 定时器
        self.oppo_timer = QTimer()
        self.app = QApplication(sys.argv)
        self.mainWindow = QMainWindow()
        self.ui = play_ui.Ui_MainWindow()
        self.ui.setupUi(self.mainWindow)
        self.mainWindow.show()
        # self.init_play_ui()  # 初始化UI
        self.event_bind()  # 绑定事件汇总

    def event_bind(self):
        """
        绑定各个事件
        """
        self.ui.raise_chip_slider.valueChanged.connect(self.label_show_slider_chip)
        self.ui.raise_chip_slider.valueChanged.connect(self.btn_show_slider_chip)
        self.ui.raise_btn.clicked.connect(self.raise_btn_step)
        self.ui.twice_bet_btn.clicked.connect(self.twice_bet_btn_label_chip)
        self.ui.third_bet_btn.clicked.connect(self.third_bet_btn_label_chip)
        self.ui.fourth_bet_btn.clicked.connect(self.fourth_bet_btn_label_chip)
        self.ui.sixth_bet_btn.clicked.connect(self.sixth_bet_btn_label_chip)
        self.ui.call_btn.clicked.connect(self.call_btn_step)
        self.ui.fold_btn.clicked.connect(self.fold_btn_step)
        self.ui.allin_btn.clicked.connect(self.allin_btn_step)
        self.my_timer.timeout.connect(self.show_my_action_timer_label)
        # self.my_timer.start(1000)  # 每隔1000ms触发一次show_my_timer_label()函数
        self.oppo_timer.timeout.connect(self.show_oppo_action_timer_label)

    def run(self):
        # self.game.reset()
        # self.oppo_strategy = Strategy(self.game, oppo_player_idx)
        # # UI布局
        # self.show_my_hand_cards()
        # self.show_public_cards()
        # self.ui.raise_chip_slider.setMinimum(200)
        # self.set_my_stage_chips()
        # self.set_oppo_stage_chips()
        # print('开始新的对局')
        self.start_game()
        self.app.exec_()

    def start_game(self):
        self.game.reset()
        self.oppo_strategy = Strategy(self.game, oppo_player_idx)
        # UI布局
        self.show_my_hand_cards()
        self.show_public_cards()
        self.ui.raise_chip_slider.setMinimum(200)
        self.show_my_stage_chips()
        self.set_oppo_stage_chips()
        print('开始新的对局')
        self.my_timer.start(1000)  # 每隔1000ms触发一次show_my_timer_label()函数

    def start_oppo_strategy_thread_1(self):
        # if self.oppo_strategy_thread_1 is not None and self.oppo_strategy_thread_1.isRunning():   # 这行代码有问题
        #     print('----oppo_strategy_thread_1线程正在运行---')
        # 设置对手策略与多线程
        self.oppo_strategy_thread_1 = QThread()
        self.oppo_strategy_worker_1 = StrategyWorker(self.oppo_strategy)
        self.oppo_strategy_worker_1.moveToThread(self.oppo_strategy_thread_1)
        self.oppo_strategy_thread_1.started.connect(self.oppo_strategy_worker_1.run)
        self.oppo_strategy_worker_1.finished.connect(self.oppo_strategy_finished_1)
        self.oppo_strategy_worker_1.finished.connect(self.oppo_strategy_thread_1.quit)
        self.oppo_strategy_worker_1.finished.connect(self.oppo_strategy_worker_1.deleteLater)
        self.oppo_strategy_thread_1.finished.connect(self.oppo_strategy_thread_1.deleteLater)
        print('启动oppo_strategy_thread_1线程')
        self.oppo_strategy_thread_1.start()
        print('启动oppo_timer行动定时器')
        self.timer_num = max_action_timer_num
        self.oppo_timer.start(1000)
        self.ui.oppo_timer_label.setText(str(max_action_timer_num))

    def start_oppo_strategy_thread_2(self):
        # if self.oppo_strategy_thread_2 is not None and self.oppo_strategy_thread_2.isRunning():   # 这行代码有问题
        #     print('----oppo_strategy_thread_2线程正在运行---')
        # 设置对手策略与多线程
        self.oppo_strategy_thread_2 = QThread()
        self.oppo_strategy_worker_2 = StrategyWorker(self.oppo_strategy)
        self.oppo_strategy_worker_2.moveToThread(self.oppo_strategy_thread_2)
        self.oppo_strategy_thread_2.started.connect(self.oppo_strategy_worker_2.run)
        self.oppo_strategy_worker_2.finished.connect(self.oppo_strategy_finished_2)
        self.oppo_strategy_worker_2.finished.connect(self.oppo_strategy_thread_2.quit)
        self.oppo_strategy_worker_2.finished.connect(self.oppo_strategy_worker_2.deleteLater)
        self.oppo_strategy_thread_2.finished.connect(self.oppo_strategy_thread_2.deleteLater)
        print('启动oppo_strategy_thread_2线程')
        self.oppo_strategy_thread_2.start()
        print('启动oppo_timer行动定时器')
        self.timer_num = max_action_timer_num
        self.oppo_timer.start(1000)
        self.ui.oppo_timer_label.setText(str(max_action_timer_num))

    def oppo_strategy_finished_1(self, result):
        """
        OppoStrategyWorker执行完毕后的槽函数
        :param result: OppoStrategyWorker中finished信号携带的参数
        :return:
        """
        print(f'调用oppo_strategy_finished_2(), 其oppo_strategy为{result}\n')
        # 做出行动
        state, is_legal, game_state_flag, down, info = self.take_oppo_action(result)
        self.set_oppo_stage_chips()
        self.set_pot_chips()
        # 进入earnChips阶段，结算本局，开始新的一局
        if self.game.cur_stage == constants.earn_chip_stage:
            print('进入earnChips阶段')
            print(f'结算筹码, {info["payoff"]}')
            self.ui.my_last_action_label.setText('')
            self.ui.oppo_last_action_label.setText('')
            my_earn_chip = info['payoffs'][catzzz_player_idx]
            if my_earn_chip > 0:
                self.ui.my_stage_chip.setText(f'我方赢得{my_earn_chip}筹码')
            elif my_earn_chip < 0:
                self.ui.my_stage_chip.setText(f'我方损失{my_earn_chip}筹码')
            else:
                self.ui.my_stage_chip.setText('平局')
            # todo 更新右侧对局信息，更新已进行的对局数
            #  新的对局会交互双方位置（大小盲注），行动顺序也有区别
            self.start_game()  # 开始新的一局
            enable_buttons_in_layout(self.ui.gridLayout_11)
            self.set_action_bar_ui()
            # todo 所有的动作都在一个函数中处理，
            #  主要流程
            #  1 根据点击事件得到执行的动作
            #  2 执行动作, self.game.step()
            #  3 执行完毕后，更新UI信息
            #  4 判断下一步该哪位玩家行动（通过self.game.cur_player_idx判断）
            #  4.1 如果下一步为我方行动，则激活操作栏，启动我方行动倒计时
            #  4.2 如果为对方行动，则禁用操作栏，启动对手策略线程和对手行动倒计时
            #  4.2.1 对手行动执行完毕后，通过回调函数继续判断后续状态，对应oppo_strategy_finished_1和oppo_strategy_finished_2函数
            #  4.3 如果对局结束（没有下一步），则结算奖励，更新右侧历史对局信息，更新对局数，开始新的对局
            return
        elif game_state_flag == self.game.CheckStateFuncResult.enter_next_state:
            self.enter_next_stage()
        # 以下是关于UI的设置：启用操作栏，设置对手的stage_chip，设置滑动条的取值范围
        print('设置对手的stage_chip')
        self.set_oppo_stage_chips()
        print('启用操作栏')
        enable_buttons_in_layout(self.ui.gridLayout_11)
        self.set_action_bar_ui()
        # 以下用于判断对手是否需要连续行动两次
        if state['cur_player_idx'] == oppo_player_idx:  # 正常情况下，最多只能连续行动两次，此处不需要while
            print('仍轮到对手行动')
            disable_buttons_in_layout(self.ui.gridLayout_11)
            self.start_oppo_strategy_thread_2()
        else:
            # 启动我方定时器
            self.timer_num = max_action_timer_num
            self.my_timer.start(1000)
            self.ui.my_timer_label.setText(str(max_action_timer_num))

    def game_logic(self, event_id, action):
        # 1 执行动作
        if event_id == 1:
            self.game.step(action)
        elif event_id == 2:
            pass
        # 2 更新UI
        # 2.1 底部操作栏
        # 2.2 我方手牌
        # 2.3 我方下注和最近一次的行动
        # 2.4 公共牌和底池大小
        # 2.5 对方下注和最近一次的行动
        # 2.6 对方倒计时
        # 2.7 我方倒计时
        # 其中2.1, 2.2, 2.3, 2.4, 2.5是静态信息，可以直接从self.game中获取
        self.show_my_hand_cards()   # 显示我方手牌
        self.show_my_stage_chips()

    def get_raise_action_by_ui(self):
        """
        生成raise动作，下注筹码为用户在UI窗口上选择的筹码量
        """
        chips = self.ui.raise_chip_slider.value()
        action = np.array([constants.RAISE_ACTION, chips])
        return action

    def show_my_hand_cards(self):
        my_hands = self.game.get_hand_cards(catzzz_player_idx)  # 我方手牌
        oppo_hands = self.game.get_hand_cards(oppo_player_idx)  # 对方手牌
        poker_png_path = f'./pokers/{my_hands[0][0]}-{my_hands[0][1]}.png'
        pixmap = QPixmap(poker_png_path)  # 加载图片
        self.ui.my_hands_1.setPixmap(pixmap)  # 设置图片到QLabel中
        poker_png_path = f'./pokers/{my_hands[1][0]}-{my_hands[1][1]}.png'
        pixmap = QPixmap(poker_png_path)
        self.ui.my_hands_2.setPixmap(pixmap)
        print(f'我方手牌:{my_hands}, 对方手牌:{oppo_hands}')

    def show_public_cards(self):
        poker_png_path = f'./pokers/back.png'
        back_pixmap = QPixmap(poker_png_path)
        # 通过self.game获取当前阶段，根据当前阶段展示对应数量的公共牌
        if self.game.cur_stage == constants.preflop_stage:
            self.ui.public_cards_1.setPixmap(back_pixmap)
            self.ui.public_cards_2.setPixmap(back_pixmap)
            self.ui.public_cards_3.setPixmap(back_pixmap)
            self.ui.public_cards_4.setPixmap(back_pixmap)
            self.ui.public_cards_5.setPixmap(back_pixmap)
        elif self.game.cur_stage == constants.flop_stage:
            public_cards = self.game.get_public_cards()
            public_pixmap_list = [QPixmap(f'./pokers/{public_cards[0][0]}-{public_cards[0][1]}.png'),
                                  QPixmap(f'./pokers/{public_cards[1][0]}-{public_cards[1][1]}.png'),
                                  QPixmap(f'./pokers/{public_cards[2][0]}-{public_cards[2][1]}.png'), ]
            self.ui.public_cards_1.setPixmap(public_pixmap_list[0])
            self.ui.public_cards_2.setPixmap(public_pixmap_list[1])
            self.ui.public_cards_3.setPixmap(public_pixmap_list[2])
        elif self.game.cur_stage == constants.turn_stage:
            public_cards = self.game.get_public_cards()
            public_pixmap_list = [QPixmap(f'./pokers/{public_cards[3][0]}-{public_cards[3][1]}.png'), ]
            self.ui.public_cards_4.setPixmap(public_pixmap_list[0])
        elif self.game.cur_stage == constants.river_stage:
            public_cards = self.game.get_public_cards()
            public_pixmap_list = [QPixmap(f'./pokers/{public_cards[4][0]}-{public_cards[4][1]}'), ]
            self.ui.public_cards_5.setPixmap(public_pixmap_list[0])
        else:
            pass

    def show_my_stage_chips(self):
        self.ui.my_stage_chip.setText(str(self.game.players[catzzz_player_idx].cur_stage_chip))



    # 以下是之前的内容
    def set_action_bar_ui(self):
        """
        每次轮到我方行动时对操作栏UI的更新
        :return:
        """
        catzzz = self.game.players[catzzz_player_idx]
        oppo = self.game.players[oppo_player_idx]
        print('设置滑动条的上下限')
        legal_actions = self.game.get_legal_actions()
        self.ui.raise_chip_slider.setMinimum(int(legal_actions[-1]))
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum())
        self.ui.raise_chip_slider.setMaximum(
            int(constants.total_chip - (catzzz.game_total_chip - catzzz.cur_stage_chip)))
        print('设置Call的筹码值')
        self.ui.call_btn.setText('Call ' + str(oppo.cur_stage_chip))

    def oppo_strategy_finished_2(self, result):
        """
        对手连续第二次行动的回调函数，并不只是在preflop阶段能连续行动两次，每进入一个新阶段都有可能再行动一次
        :param result:
        :return:
        """
        print(f'调用oppo_strategy_finished_2(), 其oppo_strategy为{result}\n')
        # 做出行动
        state, is_legal, game_state_flag, down, info = self.take_oppo_action(result)
        print('设置对手的stage_chip')
        self.set_oppo_stage_chips()
        self.set_pot_chips()
        print('启用操作栏')
        enable_buttons_in_layout(self.ui.gridLayout_11)
        self.set_action_bar_ui()
        # 启动我方定时器
        self.timer_num = max_action_timer_num
        self.my_timer.start(1000)
        self.ui.my_timer_label.setText(str(max_action_timer_num))

    def raise_btn_step(self):
        """
        下注按钮绑定事件
        """
        """
        note: 只有此按钮才真正执行下注动作，2bet等按钮用于快速选择下注筹码量，并没有真正下注
        1. 执行我方下注动作（下注后游戏逻辑）
        2. 让对手行动 or 进入下一阶段
        设置滑动条的取值范围需要等到对手行动之后才能确定
        """
        # 1. 执行下注动作
        chips = self.ui.raise_chip_slider.value()
        my_action = np.array([constants.RAISE_ACTION, chips])
        state, is_legal, game_state_flag, down, info = self.take_my_action(my_action)
        # 更新UI信息
        self.show_my_stage_chips()
        self.set_pot_chips()
        # 2 判断下一状态
        if self.game.cur_player_idx == oppo_player_idx:
            disable_buttons_in_layout(self.ui.gridLayout_11)  # 对手行动时，禁用我方操作栏UI
            self.start_oppo_strategy_thread_1()
        else:
            print(f'test444, game_state_flag={game_state_flag}, info={info}, my_action={my_action}\nstate={state}')
            pass
        # 我方执行下注动作，轮到对手行动
        # todo 什么情况下轮到对手行动，直接通过game.cur_player_idx判断
        #  双方每次行动后，都通过game.cur_player_idx判断轮到哪位玩家行动
        #  1 我方加注（无论什么情况）
        #  2 每阶段的首次行动（有可能）

    def call_btn_step(self):
        """
        Check/Call点击事件
        :return:
        """
        # todo 判断接下来该哪个玩家行动
        #  我方执行call会进入下一阶段，执行check则不进入
        #  需要区分执行的call还是check，通过get_legal_actions()得到check和call哪个合法，执行合法的那个
        legal_actions = self.game.get_legal_actions()  # [check, call, fold, allin, raise chip]
        if legal_actions[0] == 1:  # check动作合法
            my_action = np.array([constants.CHECK_ACTION, 0])
        elif legal_actions[1] == 1:
            my_action = np.array([constants.CALL_ACTION, 0])
        else:
            print('程序出错，出错代码在call_btn_step()函数')
            my_action = np.array([constants.FOLD_ACTION, -1])
        state, is_legal, game_state_flag, down, info = self.take_my_action(my_action)
        # 不能仅仅只进行当前行动，还需要对行动后的下一状态进行判断
        if self.game.cur_stage == constants.earn_chip_stage:
            print('进入earnChips阶段2222')
            pass
        elif game_state_flag == self.game.CheckStateFuncResult.enter_next_state:
            self.enter_next_stage()
        self.show_my_stage_chips()
        if state['cur_player_idx'] == catzzz_player_idx:  # 下次轮到我方行动
            print('轮到我方行动')
            # 启动我方定时器
            self.timer_num = max_action_timer_num
            self.my_timer.start(1000)
            self.ui.my_timer_label.setText(str(max_action_timer_num))
            pass
        else:  # 轮到对方行动
            print('轮到对方行动')
            self.start_oppo_strategy_thread_1()
            pass

    def fold_btn_step(self):
        print('我方: fold')
        pass

    def allin_btn_step(self):
        print('我方执行: allin')
        pass

    def take_my_action(self, my_action):
        my_action_str = action_numpy_to_str(my_action)
        print(f"我方执行: {my_action_str}")
        self.ui.my_last_action_label.setText(my_action_str)
        state, is_legal, game_state_flag, down, info = self.game.step(my_action)
        print(f"is_legal={is_legal}, game_state_flag={game_state_flag}, down={down}, info={info}\nstate={state}\n")
        # 结束我方行动定时器
        self.my_timer.stop()
        self.ui.my_timer_label.setText('')
        return state, is_legal, game_state_flag, down, info

    def take_oppo_action(self, oppo_action):
        oppo_action_str = action_numpy_to_str(oppo_action)
        print(f"对手执行: {oppo_action_str}")
        self.ui.oppo_last_action_label.setText(oppo_action_str)
        state, is_legal, game_state_flag, down, info = self.game.step(oppo_action)
        print(f"is_legal={is_legal}, game_state_flag={game_state_flag}, down={down}, info={info}\nstate={state}\n")
        # 结束对手行动定时器
        self.oppo_timer.stop()
        self.ui.oppo_timer_label.setText('')
        return state, is_legal, game_state_flag, down, info

    def show_my_action_timer_label(self):
        self.timer_num -= 1
        if self.timer_num <= 0:
            my_action = np.array([constants.FOLD_ACTION, 0])
            print(f"我方超时，自动执行: {my_action[0]}")
            state, is_legal, game_state_flag, down, info = self.game.step(my_action)
            # todo 未考虑一局结束后的处理
        else:  # 倒计时未归零，继续倒计时
            time_format = str(self.timer_num)
            self.ui.my_timer_label.setText(time_format)

    def show_oppo_action_timer_label(self):
        self.timer_num -= 1
        if self.timer_num <= 0:
            oppo_action = np.array([constants.FOLD_ACTION, 0])
            print(f"对方超时，自动执行: {oppo_action[0]}")
            state, is_legal, game_state_flag, down, info = self.game.step(oppo_action)
            # todo 未考虑一局结束后的处理
        else:
            time_format = str(self.timer_num)
            self.ui.oppo_timer_label.setText(time_format)

    def enter_next_stage(self):
        """
        进入下一阶段的UI操作，暂不包括earnChips阶段
        :return:
        """
        # 更新stage_chip
        self.show_my_stage_chips()
        self.set_oppo_stage_chips()
        # 发公共牌
        self.show_public_cards()
        # 更新UI
        if self.game.cur_stage == constants.preflop_stage:
            self.ui.stage_label.setText('preflop')
        elif self.game.cur_stage == constants.flop_stage:
            self.ui.stage_label.setText('flop')
        elif self.game.cur_stage == constants.turn_stage:
            self.ui.stage_label.setText('turn')
        elif self.game.cur_stage == constants.river_stage:
            self.ui.stage_label.setText('river')
        self.ui.my_last_action_label.setText('')
        self.ui.oppo_last_action_label.setText('')

    def set_oppo_stage_chips(self):
        self.ui.oppo_stage_chip.setText(str(self.game.players[oppo_player_idx].cur_stage_chip))

    def set_pot_chips(self):
        self.ui.pot_chip.setText(str(self.game.pot_chip))

    def label_show_slider_chip(self):
        """
        滑动块旁边的Label同步显示筹码量
        """
        self.ui.raise_chip_label.setText(f"{self.ui.raise_chip_slider.value()}")

    def btn_show_slider_chip(self):
        self.ui.raise_btn.setText(f'Raise {self.ui.raise_chip_slider.value()}')

    def twice_bet_btn_label_chip(self):
        """
        在滑动条上确定下注筹码，没有进行下注动作
        """
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum())  # 滑动条的最小值就是最低加注筹码

    def third_bet_btn_label_chip(self):
        self.ui.raise_chip_slider.setValue(int(self.ui.raise_chip_slider.minimum() / 2 * 3))

    def fourth_bet_btn_label_chip(self):
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum() * 2)

    def sixth_bet_btn_label_chip(self):
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum() * 3)


# 禁用布局中的所有按钮
def disable_buttons_in_layout(layout: QGridLayout):
    for i in range(layout.rowCount()):
        for j in range(layout.columnCount()):
            item = layout.itemAtPosition(i, j)
            if item is not None:
                widget = item.widget()
                if isinstance(widget, QPushButton):  # 检查控件是否是按钮
                    widget.setEnabled(False)


def enable_buttons_in_layout(layout: QGridLayout):
    for i in range(layout.rowCount()):
        for j in range(layout.columnCount()):
            item = layout.itemAtPosition(i, j)
            if item is not None:
                widget = item.widget()
                if isinstance(widget, QPushButton):  # 检查控件是否是按钮
                    widget.setEnabled(True)


def action_numpy_to_str(action: np.ndarray):
    if action[0] == constants.CHECK_ACTION:
        return 'Check'
    elif action[0] == constants.CALL_ACTION:
        return 'Call'
    elif action[0] == constants.FOLD_ACTION:
        return 'Fold'
    elif action[0] == constants.ALLIN_ACTION:
        return 'Allin'
    elif action[0] == constants.RAISE_ACTION:
        return 'Raise ' + str(action[1])
    else:
        return 'Unknown action'


def test_act(game: NoLimitHoldemGame, act):
    print('做出的动作', act)
    state, is_legal, game_state_flag, down, info = game.step(act)
    print('下一状态')
    # game.print_info()
    print(state)
    print('is legal:', is_legal, 'game_state_flag:', game_state_flag, 'info:', info)
    print('\n\n')


def test(game=NoLimitHoldemGame()):
    game.prepare_round()
    game.print_info()
    print(game.get_legal_actions())
    # preflop
    chips = np.random.randint(200, 2000, size=5)
    print('preflop')
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)
    act = np.array([constants.RAISE_ACTION, chips[0]])
    test_act(game, act)
    act = np.array([constants.CALL_ACTION, 100])
    test_act(game, act)
    print('flop')
    act = np.array([constants.CHECK_ACTION, 0])
    test_act(game, act)
    act = np.array([constants.CALL_ACTION, 0])
    test_act(game, act)


if __name__ == '__main__':
    runner = NoLimitHoldemRunner()
    runner.run()
    # game1 = NoLimitHoldemGame()
    # test(game1)
