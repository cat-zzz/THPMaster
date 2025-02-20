"""
@project: THPMaster
@File   : play_func.py
@Desc   :
@Author : gql
@Date   : 2025/2/1 15:27
"""
import sys
import time

import numpy as np
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton

# from competition.core.entity import Player, Game
# from competition.core.strategy.strategy_2 import Strategy
from gui import play
from gui.comp.util.common_util import print_exception
from gui.env.rl_game import NoLimitHoldemGame
from gui.strategy.strategy import Strategy
from src.env import constants


# from PySide6.QtCore import Signal, QObject


# 业务逻辑

# 绑定各组件的事件


def func():
    game = NoLimitHoldemGame()
    game.reset()
    act = np.array([constants.CALL_ACTION, 100])
    i = 0
    down = False
    while not down:
        state, is_legal, game_state_flag, down, info = game.step(act)
        print(f"i={i}, is_legal={is_legal}, game_state_flag={game_state_flag}, down={down}\ninfo={info}\nstate={state}")
    pass


catzzz_player_idx = 0
oppo_player_idx = 1
max_action_timer_num = 120


class NoLimitHoldemRunner(QObject):
    """
    管理窗口，同时处理游戏逻辑
    """
    oppo_action_signal = pyqtSignal()

    class StrategyQThread(QThread):
        def __init__(self):
            super().__init__()

        def run(self):
            # todo 未实现
            pass

    def __init__(self):
        super().__init__()
        # 下面是用于比赛的代码
        # self.catzzz = Player("catzzz")
        # self.opponent = Player("opponent")
        # self.comp_game = Game(self.catzzz, self.opponent)
        # self.oppo_strategy = Strategy()  # 对手的策略
        # 下面是强化学习环境
        self.game = NoLimitHoldemGame()
        self.oppo_strategy = None
        # 下面是PyQt
        self.timer_num = 120
        self.my_timer = QTimer()  # 定时器
        self.oppo_timer = QTimer()
        self.app = QApplication(sys.argv)
        self.mainWindow = QMainWindow()
        self.ui = play.Ui_MainWindow()
        self.ui.setupUi(self.mainWindow)
        self.mainWindow.show()
        self.init_play_ui()  # 初始化UI
        self.event_bind()  # 绑定事件汇总

    def run(self):
        """
        NoLimitHoldemRunner类的入口函数
        """
        self.game.reset()
        self.oppo_strategy = Strategy(self.game, oppo_player_idx)
        self.display_hand_cards()
        self.display_public_cards()
        self.ui.raise_chip_slider.setMinimum(200)
        self.set_my_stage_chip()
        self.set_oppo_stage_chip()
        print('开始新的对局')
        self.app.exec_()

    def display_hand_cards(self):
        my_hands = self.game.get_hand_cards(catzzz_player_idx)  # 我方手牌
        oppo_hands = self.game.get_hand_cards(oppo_player_idx)  # 对方手牌
        poker_png_path = f'./pokers/{my_hands[0][0]}-{my_hands[0][1]}.png'
        pixmap = QPixmap(poker_png_path)  # 加载图片
        self.ui.my_hands_1.setPixmap(pixmap)  # 设置图片到QLabel中
        poker_png_path = f'./pokers/{my_hands[1][0]}-{my_hands[1][1]}.png'
        pixmap = QPixmap(poker_png_path)
        self.ui.my_hands_2.setPixmap(pixmap)
        print(f'我方手牌:{my_hands}, 对方手牌:{oppo_hands}')

    def display_public_cards(self):
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
            self.ui.public_cards_4.setPixmap(back_pixmap)
            self.ui.public_cards_5.setPixmap(back_pixmap)
        elif self.game.cur_stage == constants.turn_stage:
            public_cards = self.game.get_public_cards()
            public_pixmap_list = [QPixmap(f'./pokers/{public_cards[0][0]}-{public_cards[0][1]}.png'),
                                  QPixmap(f'./pokers/{public_cards[1][0]}-{public_cards[1][1]}.png'),
                                  QPixmap(f'./pokers/{public_cards[2][0]}-{public_cards[2][1]}.png'),
                                  QPixmap(f'./pokers/{public_cards[3][0]}-{public_cards[3][1]}.png'), ]
            self.ui.public_cards_1.setPixmap(public_pixmap_list[0])
            self.ui.public_cards_2.setPixmap(public_pixmap_list[1])
            self.ui.public_cards_3.setPixmap(public_pixmap_list[2])
            self.ui.public_cards_4.setPixmap(public_pixmap_list[3])
            self.ui.public_cards_5.setPixmap(back_pixmap)
        elif self.game.cur_stage == constants.river_stage:
            public_cards = self.game.get_public_cards()
            public_pixmap_list = [QPixmap(f'./pokers/{public_cards[0][0]}-{public_cards[0][1]}'),
                                  QPixmap(f'./pokers/{public_cards[1][0]}-{public_cards[1][1]}'),
                                  QPixmap(f'./pokers/{public_cards[2][0]}-{public_cards[2][1]}'),
                                  QPixmap(f'./pokers/{public_cards[3][0]}-{public_cards[3][1]}'),
                                  QPixmap(f'./pokers/{public_cards[4][0]}-{public_cards[4][1]}'), ]
            self.ui.public_cards_1.setPixmap(public_pixmap_list[0])
            self.ui.public_cards_2.setPixmap(public_pixmap_list[1])
            self.ui.public_cards_3.setPixmap(public_pixmap_list[2])
            self.ui.public_cards_4.setPixmap(public_pixmap_list[3])
            self.ui.public_cards_5.setPixmap(public_pixmap_list[4])
        else:
            pass

    def init_play_ui(self):
        """
        主要用于设置那些无法直接在QtDesigner的属性
        """
        # 设置下注筹码Label的字体大小
        font = QFont()
        font.setPointSize(12)  # 设置字体大小
        # self.ui.raise_chip_label.setFont(font)
        # self.ui.raise_chip_label.setText('200')  # 筹码表示标签初始值设为100
        # self.ui.my_stage_chip.setFont(font)
        # self.ui.oppo_stage_chip.setFont(font)

    def event_bind(self):
        """
        绑定各个事件
        """
        self.ui.raise_chip_slider.valueChanged.connect(self.label_display_slider_chip)
        self.ui.raise_btn.clicked.connect(self.raise_btn_step)
        self.ui.twice_bet_btn.clicked.connect(self.twice_bet_btn_label_chip)
        self.oppo_action_signal.connect(self.handle_oppo_action_signal)
        self.ui.call_btn.clicked.connect(self.call_btn_step)
        self.ui.fold_btn.clicked.connect(self.fold_btn_step)
        self.ui.allin_btn.clicked.connect(self.allin_btn_step)
        self.my_timer.timeout.connect(self.display_my_timer_label)
        self.my_timer.start(1000)   # 每隔1000ms触发一次display_my_timer_label()函数
        self.oppo_timer.timeout.connect(self.display_oppo_timer_label)
        # self.oppo_timer.start(1000)

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
        chips = self.ui.raise_chip_slider.value()
        my_action = np.array([constants.RAISE_ACTION, chips])
        state, is_legal, game_state_flag, down, info = self.take_my_action(my_action)

        self.set_my_stage_chip()
        # legal_actions = self.game.get_legal_actions()
        # print(f'我方执行: raise {my_action[1]}')
        # print(f'legal_action: {legal_actions}')
        # print(f"is_legal={is_legal}, game_state_flag={game_state_flag}, down={down}\ninfo={info}\nstate={state}\n")
        # 此时执行的是raise动作，一定需要对手行动，所以可以发送对手行动信号
        self.oppo_action_signal.emit()


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
            state, is_legal, game_state_flag, down, info = self.game.step(my_action)
            print('我方执行: check')
        else:
            my_action = np.array([constants.CALL_ACTION, 0])
            state, is_legal, game_state_flag, down, info = self.game.step(my_action)
            print('我方执行: call')
            if game_state_flag == self.game.CheckStateFuncResult.enter_next_state:
                self.enter_next_stage()
        pass

    def fold_btn_step(self):
        print('我方: fold')
        pass

    def allin_btn_step(self):
        print('我方执行: allin')
        pass

    def handle_oppo_action_signal(self):
        """
        轮到对手行动时触发的函数，对手的操作都在此函数中进行，此函数执行完毕将轮到我方行动
        :return:
        """
        print('轮到对手行动，暂时禁用操作栏')
        disable_buttons_in_layout(self.ui.gridLayout_11)
        oppo_action = self.oppo_strategy.strategy_2()
        time.sleep(2)
        state, is_legal, game_state_flag, down, info = self.take_oppo_action(oppo_action)
        # 判断接下来该哪个玩家行动
        # 如果该对手行动，则继续调用策略函数，得到一个动作并执行，如此重复下去直到轮到自己行动
        # 如果该我方行动，则结束此函数、激活底部UI的操作栏
        if game_state_flag == self.game.CheckStateFuncResult.enter_next_state:
            self.enter_next_stage()
        if state['cur_player_idx'] == oppo_player_idx:  # 正常情况下，最多只能连续行动两次，此处不需要while
            print('仍然轮到对手行动')
            time.sleep(1)
            oppo_action = self.oppo_strategy.strategy_2()
            state, is_legal, game_state_flag, down, info = self.take_oppo_action(oppo_action)
        # 以下是关于UI的设置：启用操作栏，设置对手的stage_chip，设置滑动条的取值范围
        print('设置对手的stage_chip')
        self.set_oppo_stage_chip()
        print('启用操作栏')
        enable_buttons_in_layout(self.ui.gridLayout_11)
        print('设置滑动条的最小值')
        legal_actions = self.game.get_legal_actions()
        self.ui.raise_chip_slider.setMinimum(int(legal_actions[-1]))
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum())

    def enter_next_stage(self):
        # 更新stage_chip
        self.set_my_stage_chip()
        self.set_oppo_stage_chip()
        # 发公共牌
        self.display_public_cards()
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

    def take_my_action(self, my_action):
        my_action_str = action_numpy_to_str(my_action)
        print(f"我方执行: {my_action_str}")
        self.ui.oppo_last_action_label.setText(my_action_str)
        state, is_legal, game_state_flag, down, info = self.game.step(my_action)
        print(f"is_legal={is_legal}, game_state_flag={game_state_flag}, down={down}\ninfo={info}\nstate={state}\n")
        self.my_timer.stop()
        self.timer_num = max_action_timer_num
        self.ui.my_timer_label.setText('')
        self.oppo_timer.start(1000)
        self.ui.oppo_timer_label.setText(str(max_action_timer_num))
        return state, is_legal, game_state_flag, down, info

    def take_oppo_action(self, oppo_action):
        oppo_action_str = action_numpy_to_str(oppo_action)
        print(f"对手执行: {oppo_action_str}")
        self.ui.oppo_last_action_label.setText(oppo_action_str)
        state, is_legal, game_state_flag, down, info = self.game.step(oppo_action)
        print(f"is_legal={is_legal}, game_state_flag={game_state_flag}, down={down}\ninfo={info}\nstate={state}\n")
        self.oppo_timer.stop()
        self.ui.oppo_timer_label.setText('')
        self.timer_num = max_action_timer_num
        self.ui.my_timer_label.setText(str(max_action_timer_num))
        self.my_timer.start(1000)
        return state, is_legal, game_state_flag, down, info

    def set_my_stage_chip(self):
        """
        设置我方当前阶段下注筹码量，在我方行动后或进入新阶段时手动调用
        """
        self.ui.my_stage_chip.setText(str(self.game.players[catzzz_player_idx].cur_stage_chip))

    def set_oppo_stage_chip(self):
        """
            设置对方当前阶段下注筹码量，在对方行动后或进入新阶段时手动调用
        """
        self.ui.oppo_stage_chip.setText(str(self.game.players[oppo_player_idx].cur_stage_chip))

    def display_my_timer_label(self):
        self.timer_num -= 1
        if self.timer_num <= 0:
            my_action = np.array([constants.FOLD_ACTION, 0])
            print(f"我方超时，自动执行: {my_action[0]}")
            state, is_legal, game_state_flag, down, info = self.game.step(my_action)
            # todo 未考虑一局结束后的处理
        else:  # 倒计时未归零，继续倒计时
            time_format = str(self.timer_num)
            self.ui.my_timer_label.setText(time_format)

    def display_oppo_timer_label(self):
        # 获取系统现在的时间
        # time = QDateTime.currentDateTime()
        # 设置系统时间显示格式
        # time_format = time.toString("yyyy-MM-dd hh:mm:ss dddd")
        self.timer_num -= 1
        if self.timer_num <= 0:
            oppo_action = np.array([constants.FOLD_ACTION, 0])
            print(f"对方超时，自动执行: {oppo_action[0]}")
            state, is_legal, game_state_flag, down, info = self.game.step(oppo_action)
            # todo 未考虑一局结束后的处理
        else:
            time_format = str(self.timer_num)
            self.ui.oppo_timer_label.setText(time_format)

    def label_display_slider_chip(self):
        """
        滑动块旁边的Label同步显示筹码量
        """
        self.ui.raise_chip_label.setText(f"{self.ui.raise_chip_slider.value()}")

    def twice_bet_btn_label_chip(self):
        """
        在滑动条上确定下注筹码，没有进行下注动作
        """
        """
        滑动条的最小值是2bet对应的筹码量
        """
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum())  # 滑动条的最小值就是最低加注筹码

    def get_oppo_strategy(self):
        return self.oppo_strategy


def client_cmd_to_numpy(client_cmd: str) -> np.ndarray:
    cmd_split = client_cmd.split(' ')
    if cmd_split[0] == 'call':
        client_action = np.array([constants.CALL_ACTION, 100])
    elif cmd_split[0] == 'check':
        client_action = np.array([constants.CHECK_ACTION, 0])
    elif cmd_split[0] == 'fold':
        client_action = np.array([constants.FOLD_ACTION, 0])
    elif cmd_split[0] == 'raise':
        chip = 0
        x = [str(x) for x in range(0, 10)]  # 产生字符0-9
        for i in cmd_split[1]:
            if i in x:
                chip = chip * 10 + int(i)
            else:
                break
        client_action = np.array([constants.RAISE_ACTION, chip])
    elif cmd_split[0] == 'allin':
        client_action = np.array([constants.ALLIN_ACTION, 20000])
    else:
        print_exception(client_cmd_to_numpy, "未知的客户端指令")
        client_action = np.array([constants.FOLD_ACTION, 0])
    return client_action


# 禁用布局中的所有按钮
def disable_buttons_in_layout(layout: QGridLayout):
    for i in range(layout.rowCount()):
        for j in range(layout.columnCount()):
            item = layout.itemAtPosition(i, j)
            if item is not None:
                widget = item.widget()
                if isinstance(widget, QPushButton):  # 检查控件是否是按钮
                    widget.setEnabled(False)


# 启用布局中的所有按钮
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


if __name__ == '__main__':
    runner = NoLimitHoldemRunner()
    runner.run()
    # app = QApplication(sys.argv)
    # MainWindow = QMainWindow()
    # ui_1 = play.Ui_MainWindow()
    # ui_1.setupUi(MainWindow)
    # MainWindow.show()
    # init_play_ui(ui_1)  # 初始化UI
    # event_bind(ui_1)  # 绑定事件汇总
    # func()
    # sys.exit(app.exec_())
