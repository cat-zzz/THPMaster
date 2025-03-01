"""
@project: THPMaster
@File   : play_controller_3.py
@Desc   :
@Author : gql
@Date   : 2025/2/26 15:12
"""
import sys
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread, Qt
from PyQt5.QtGui import QPixmap, QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QGridLayout, QLabel, QWidget, QVBoxLayout

from gui import play_ui
from gui.env import constants
from gui.env.rl_game import NoLimitHoldemGame
from gui.strategy.strategy import Strategy
from gui.util import card_util

catzzz_player_idx = 0
oppo_player_idx = 1
max_action_timer_num = 120


class StrategyWorker(QObject):
    finished = pyqtSignal(np.ndarray)  # 其参数即是对手策略的动作

    def __init__(self, strategy: Strategy):
        super().__init__()
        self.oppo_strategy = strategy

    def run(self):
        time.sleep(2)
        oppo_action = self.oppo_strategy.strategy_2()
        print(f'发送StrategyWorker完成信号，生成的动作{oppo_action}')
        self.finished.emit(oppo_action)


class NoLimitHoldemRunner(QObject):
    # oppo_action_signal = pyqtSignal()
    window_closed = pyqtSignal()

    def __init__(self, total_episode=100, seed=None):
        super().__init__()
        self.episode = 0  # 当前对局数
        self.total_episode = total_episode
        self.oppo_strategy = None
        self.oppo_strategy_worker_1 = None
        self.oppo_strategy_thread_1 = QThread()
        # 由于同一玩家可能连续行动两次（preflop到flop阶段），每一次行动对应一个worker和thread
        self.oppo_strategy_worker_2 = None
        self.oppo_strategy_thread_2 = QThread()
        # 下面是强化学习环境
        game_config = {
            'seed': 0 if seed is None else seed,  # 随机数种子，为0表示不设置随机数种子
        }
        self.game = NoLimitHoldemGame(game_config)
        # 下面是PyQt
        self.game_history_model = QStandardItemModel()
        self.timer_num = max_action_timer_num
        self.my_timer = QTimer()  # 定时器
        self.oppo_timer = QTimer()
        # self.app = QApplication(sys.argv)
        self.mainWindow = QMainWindow()
        self.ui = play_ui.Ui_MainWindow()
        self.ui.setupUi(self.mainWindow)
        self.init_tree_view()
        self.mainWindow.show()
        self.event_bind()  # 绑定事件汇总
        self.mainWindow.closeEvent = self.close_event

    def show_window(self):
        """安全显示窗口的方法"""
        if not self.mainWindow.isVisible():
            self.mainWindow.show()

    def close_event(self, event):
        """
        重写关闭事件，关闭当前窗口时，自动显示人机对弈配置页面
        """
        self.window_closed.emit()  # 发送关闭信号
        self.mainWindow.close()  # 执行默认关闭操作
        event.accept()  # 接受关闭事件

    def init_tree_view(self):
        # self.model = QStandardItemModel()
        self.game_history_model.setHorizontalHeaderLabels(["牌局信息序列"])  # 设置表头
        # 直接添加牌局信息
        # item1 = QStandardItem("对局1：Tommy赢得20000个编码")
        # item2 = QStandardItem("对局2：Tommy赢得50个筹码")
        # 将牌局信息添加到模型
        # self.game_history_model.appendRow(item1)
        # self.game_history_model.appendRow(item2)
        # 将模型设置到 QTreeView
        self.ui.game_history.setModel(self.game_history_model)

    def event_bind(self):
        """
        绑定各个事件
        """
        self.ui.raise_chip_slider.valueChanged.connect(self.label_show_slider_chip)
        self.ui.raise_chip_slider.valueChanged.connect(self.btn_show_slider_chip)
        self.ui.raise_btn.clicked.connect(self.raise_btn_click_event)
        self.ui.twice_bet_btn.clicked.connect(self.twice_bet_btn_label_chip)
        self.ui.third_bet_btn.clicked.connect(self.third_bet_btn_label_chip)
        self.ui.fourth_bet_btn.clicked.connect(self.fourth_bet_btn_label_chip)
        self.ui.sixth_bet_btn.clicked.connect(self.sixth_bet_btn_label_chip)
        self.ui.call_btn.clicked.connect(self.call_btn_click_event)
        self.ui.fold_btn.clicked.connect(self.fold_btn_click_event)
        self.ui.allin_btn.clicked.connect(self.allin_btn_click_event)
        self.my_timer.timeout.connect(self.show_my_action_timer_label)
        self.oppo_timer.timeout.connect(self.show_oppo_action_timer_label)
        self.ui.scrollArea.verticalScrollBar().rangeChanged.connect(self.handle_scroll_bar_range_changed)

    def run(self):
        self.start_game_2()
        # self.app.exec_()

    def start_game(self):
        # 开始新的一局游戏需要判断：大小盲注位置，首次下注玩家，禁用/激活操作栏
        self.game.reset()
        self.oppo_strategy = Strategy(self.game, oppo_player_idx)
        # UI布局
        self.show_my_hand_cards()
        self.show_public_cards()
        self.ui.raise_chip_slider.setMinimum(200)
        self.show_my_stage_chips()
        self.show_oppo_stage_chips()
        print('开始新的对局')
        self.my_timer.start(1000)  # 每隔1000ms触发一次show_my_timer_label()函数

    def start_game_2(self):
        """
        开始新的一局（每开始一局都要调用此函数），此函数不会判断总对局数
        :return:
        """
        self.episode += 1
        print(f'开始第{self.episode}局')
        self.game.reset()
        self.oppo_strategy = Strategy(self.game, oppo_player_idx)
        self.game.print_info()

        # 计时器
        self.stop_oppo_action_timer()
        self.stop_my_action_timer()
        # UI相关
        self.clear_scroll_view()  # 每局开始时清除历史动作
        self.show_my_hand_cards()
        self.show_public_cards()
        self.show_my_stage_chips()
        self.show_oppo_stage_chips()
        if self.game.cur_player_idx == catzzz_player_idx:  # 开局我方首次行动
            self.ui.raise_chip_slider.setMinimum(constants.bigblind_chip * 2)
            self.start_my_action_timer()
        else:
            disable_buttons_in_layout(self.ui.gridLayout_11)
            self.start_oppo_action_timer()
            self.start_oppo_strategy_thread_1()

    def game_logic(self, player_id, continuous_action_flag, action):
        """
        核心逻辑处理
        :param player_id:
        :param continuous_action_flag: 表示是否为连续的第二次行动
        :param action:
        :return:
        """
        # 双方的行动都在此函数中进行
        # 动作合法性判断
        if player_id != self.game.cur_player_idx:  # 判断调用此函数的player_id应和self.game中的cur_player_idx是否相同
            card_util.print_exception(self.game_logic,
                                      f'未轮到玩家{player_id}行动，self.game指示应为玩家{self.game.cur_player_idx}行动')
            return
        legal_actions = self.game.get_legal_actions()
        # 1 执行动作
        state, is_legal, game_state_flag, down, info = self.game.step(action)
        # print(
        #     f'本次行动玩家索引: {player_id}, 采取的动作:{action}, 本次合法动作: {legal_actions}\n行动后的状态信息:\n{state}\n'
        #     f'game_state_flag={game_state_flag}, down={down}, info={info}\n')
        print(
            f'本次行动玩家索引: {player_id}, 采取的动作:{action}, 本次行动的合法动作: {legal_actions}\n行动后的状态信息:')
        self.game.print_info()
        self.add_action_to_label(player_id, action)
        # 2 更新UI，主要包括我方手牌、我方下注、我方上次的行动、公共牌、底池大小、对方下注和对方上次的行动
        self.show_my_hand_cards()
        self.show_my_stage_chips()
        self.show_public_cards()
        self.show_pot_chips()
        self.show_oppo_stage_chips()
        self.show_stage_label()
        # 显示双方最新一次的动作
        if action[0] == constants.CALL_ACTION and game_state_flag == self.game.CheckStateFuncResult.enter_next_state:
            self.ui.my_last_action_label.setText('')
            self.ui.oppo_last_action_label.setText('')
        elif player_id == catzzz_player_idx:
            self.show_my_last_action_label(action)
        elif player_id == oppo_player_idx:
            self.show_oppo_last_action_label(action)
        # 判断是否进入下一阶段，用于更新右上部分UI的信息
        if game_state_flag == self.game.CheckStateFuncResult.enter_next_state:
            self.handle_enter_next_stage()

        # 3 判断是否进入earnChip阶段（即本局是否结束）（earnChip包括正常摊牌，有玩家allin，仅一位玩家未fold）
        # todo 或许可以通过game.step()返回的down标志判断本局是否结束
        if self.game.cur_stage == constants.earn_chip_stage:
            print('下注轮结束，正常比牌')
            self.handle_earn_chip_stage(info['payoffs'])
            if self.episode <= self.total_episode:
                # todo 延迟10秒再开始下一局
                self.start_game_2()
            else:
                # todo 全部结束
                print('全部对局结束')
            return
        if game_state_flag == self.game.CheckStateFuncResult.folded_enter_earn_chip_stage:
            print('其余玩家均弃牌，结算奖励')
            self.handle_earn_chip_stage(info['payoffs'])
            if self.episode <= self.total_episode:
                self.start_game_2()
            else:
                # todo 全部结束
                print('全部对局结束')
            return
        elif game_state_flag == self.game.CheckStateFuncResult.allin_enter_earn_chip_stage:
            print('剩余玩家Allin，直接比牌')
            self.handle_earn_chip_stage(info['payoffs'])
            return

        # 4 判断下一位行动玩家索引
        if self.game.cur_player_idx == catzzz_player_idx:  # 下次由我方行动
            enable_buttons_in_layout(self.ui.gridLayout_11)  # 激活操作栏
            self.show_my_action_bar()
            self.stop_oppo_action_timer()  # 停止对方倒计时
            self.start_my_action_timer()  # 开始我方倒计时
            pass
        else:  # 下次该对方行动
            disable_buttons_in_layout(self.ui.gridLayout_11)
            self.stop_my_action_timer()
            self.start_oppo_action_timer()
            if self.game.cur_player_idx == player_id:
                self.start_oppo_strategy_thread_2()
            else:
                self.start_oppo_strategy_thread_1()

    def handle_earn_chip_stage(self, payoffs):
        print(f'进入earnChips阶段, 结算筹码{payoffs}')
        # 左侧主要UI
        self.ui.my_last_action_label.setText('')
        self.ui.oppo_last_action_label.setText('')
        my_earn_chip = payoffs[catzzz_player_idx]
        if my_earn_chip > 0:
            win_info = f'对局{self.episode}: 玩家赢得{np.abs(my_earn_chip)}筹码'
            self.ui.my_stage_chip.setText(f'我方赢得{my_earn_chip}筹码')
        elif my_earn_chip < 0:
            win_info = f'对局{self.episode}: AI赢得{np.abs(my_earn_chip)}筹码'
            self.ui.my_stage_chip.setText(f'我方输了{np.abs(my_earn_chip)}筹码')
        else:
            win_info = f'平局'
            self.ui.my_stage_chip.setText('平局')
        self.show_all_public_cards()
        disable_buttons_in_layout(self.ui.gridLayout_11)  # 禁用操作栏
        # 向右下UI写入本局输赢信息
        item = QStandardItem(win_info)
        self.game_history_model.appendRow(item)
        # todo 停止双方的计时器，展示双方手牌，开始下一局按钮
        self.show_oppo_hand_cards()
        self.stop_my_action_timer()
        self.stop_oppo_action_timer()

    def handle_enter_next_stage(self):
        """
        进入下一阶段的UI操作，如右上部分的发牌，翻开新公共牌
        :return:
        """
        if self.game.cur_stage == constants.earn_chip_stage:
            return
        elif self.game.cur_stage == constants.flop_stage:
            public_cards = self.get_cur_stage_public_cards()
            public_cards_str = f'发牌{public_cards[0]}, {public_cards[1]}, {public_cards[2]}'
        else:
            public_cards = self.get_cur_stage_public_cards()
            public_cards_str = f'发牌{public_cards}'
        self.add_action_to_label(-1, public_cards_str)

    def start_oppo_strategy_thread_1(self):
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

    def start_oppo_strategy_thread_2(self):
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

    def oppo_strategy_finished_1(self, oppo_action):
        print('处理对手策略生成的动作')
        self.game_logic(oppo_player_idx, 0, oppo_action)

    def oppo_strategy_finished_2(self, oppo_action):
        print('处理对手策略连续第二次生成的动作')
        self.game_logic(oppo_player_idx, 1, oppo_action)

    def raise_btn_click_event(self):
        chips = self.ui.raise_chip_slider.value()
        action = np.array([constants.RAISE_ACTION, chips])
        self.game_logic(catzzz_player_idx, 0, action)

    def call_btn_click_event(self):
        # 判断check和call哪个动作合法
        legal_actions = self.game.get_legal_actions()  # [check, call, fold, allin, raise chip]
        if legal_actions[0] == 1:  # check动作合法
            action = np.array([constants.CHECK_ACTION, 0])
        elif legal_actions[1] == 1:
            action = np.array([constants.CALL_ACTION, 0])
        else:
            card_util.print_exception(self.call_btn_click_event, '程序出错，出错代码在call_btn_step()函数')
            action = np.array([constants.FOLD_ACTION, -1])
        self.game_logic(catzzz_player_idx, 0, action)

    def fold_btn_click_event(self):
        action = np.array([constants.FOLD_ACTION, -1])
        self.game_logic(catzzz_player_idx, 0, action)

    def allin_btn_click_event(self):
        action = np.array([constants.ALLIN_ACTION, 0])
        self.game_logic(catzzz_player_idx, 0, action)

    def add_action_to_label(self, player_idx, action):
        """
        :param player_idx: 行动者索引，其中-1表示系统动作（如发牌）
        :param action:
        :return:
        """
        if player_idx == catzzz_player_idx:
            action_str = action_numpy_to_str(action)
            action_item = '玩家--' + action_str
        elif player_idx == oppo_player_idx:
            action_str = action_numpy_to_str(action)
            action_item = 'AI----' + action_str
        elif player_idx == -1:
            action_str = str(action)
            action_item = '系统--' + action_str
        else:
            action_item = '未知的动作'
        label = QLabel(action_item)
        label.setFixedHeight(30)
        label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.ui.scrollArea.widget().layout().setAlignment(Qt.AlignTop)  # 需要在此处重新设置Alignment
        self.ui.scrollArea.widget().layout().addWidget(label)

    def handle_scroll_bar_range_changed(self, min_value, max_value):
        """
        滚动条自动滚动到最下方
        ScrollBar的范围改变时，总会调用此函数
        :param min_value: 必要的参数
        :param max_value: 必要的参数
        :return:
        """
        self.ui.scrollArea.verticalScrollBar().setValue(max_value)

    def start_my_action_timer(self):
        """
        启动定时器
        """
        self.timer_num = max_action_timer_num
        self.my_timer.start(1000)
        self.ui.my_timer_label.setText(str(max_action_timer_num))

    def stop_my_action_timer(self):
        self.my_timer.stop()
        self.ui.my_timer_label.setText('')

    def show_my_action_timer_label(self):
        """
        定时器的触发函数
        """
        self.timer_num -= 1
        if self.timer_num <= 0:
            # todo 超时自动弃牌（未完成）
            my_action = np.array([constants.FOLD_ACTION, 0])
            print(f"我方超时，自动执行: {my_action[0]}（未完成）")
            # state, is_legal, game_state_flag, down, info = self.game.step(my_action)
            # todo 未考虑一局结束后的处理
        else:  # 倒计时未归零，继续倒计时
            time_format = str(self.timer_num)
            self.ui.my_timer_label.setText(time_format)

    def start_oppo_action_timer(self):
        self.timer_num = max_action_timer_num
        self.oppo_timer.start(1000)
        self.ui.oppo_timer_label.setText(str(max_action_timer_num))

    def stop_oppo_action_timer(self):
        self.oppo_timer.stop()
        self.ui.oppo_timer_label.setText('')

    def get_cur_stage_public_cards(self):
        public_cards = self.game.get_public_cards()
        if self.game.cur_stage == constants.flop_stage:
            return public_cards[0], public_cards[1], public_cards[2]
        elif self.game.cur_stage == constants.turn_stage:
            return public_cards[3]
        elif self.game.cur_stage == constants.river_stage:
            return public_cards[4]
        else:
            pass

    def clear_scroll_view(self):
        layout = self.ui.scrollArea.widget().layout()
        # 遍历并删除所有子控件
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # 可选：更新布局
        # content_widget.updateGeometry()

    def show_oppo_action_timer_label(self):
        self.timer_num -= 1
        if self.timer_num <= 0:
            # todo 超时自动弃牌（未完成）
            oppo_action = np.array([constants.FOLD_ACTION, 0])
            print(f"对方超时，自动执行: {oppo_action[0]}（未完成）")
            # state, is_legal, game_state_flag, down, info = self.game.step(my_action)
            # todo 未考虑一局结束后的处理
        else:  # 倒计时未归零，继续倒计时
            time_format = str(self.timer_num)
            self.ui.oppo_timer_label.setText(time_format)

    def show_my_action_bar(self):
        """
        每次轮到我方行动时对操作栏UI的更新
        """
        catzzz = self.game.players[catzzz_player_idx]
        oppo = self.game.players[oppo_player_idx]
        # 设置滑动条的上下限
        legal_actions = self.game.get_legal_actions()
        self.ui.raise_chip_slider.setMinimum(int(legal_actions[-1]))
        self.ui.raise_chip_slider.setValue(self.ui.raise_chip_slider.minimum())
        self.ui.raise_chip_slider.setMaximum(
            int(constants.total_chip - (catzzz.game_total_chip - catzzz.cur_stage_chip))
        )
        # 设置call对应的筹码量
        self.ui.call_btn.setText('Call ' + str(oppo.cur_stage_chip))

    def show_my_hand_cards(self):
        my_hands = self.game.get_hand_cards(catzzz_player_idx)  # 我方手牌
        poker_png_path = f'./pokers/{my_hands[0][0]}-{my_hands[0][1]}.png'
        pixmap = QPixmap(poker_png_path)  # 加载图片
        self.ui.my_hands_1.setPixmap(pixmap)  # 设置图片到QLabel中
        poker_png_path = f'./pokers/{my_hands[1][0]}-{my_hands[1][1]}.png'
        pixmap = QPixmap(poker_png_path)
        self.ui.my_hands_2.setPixmap(pixmap)

    def show_oppo_hand_cards(self):
        oppo_hands = self.game.get_hand_cards(oppo_player_idx)  # 对方手牌
        poker_png_path = f'./pokers/{oppo_hands[0][0]}-{oppo_hands[0][1]}.png'
        pixmap = QPixmap(poker_png_path)  # 加载图片
        self.ui.oppo_hands_1.setPixmap(pixmap)  # 设置图片到QLabel中
        poker_png_path = f'./pokers/{oppo_hands[1][0]}-{oppo_hands[1][1]}.png'
        pixmap = QPixmap(poker_png_path)
        self.ui.oppo_hands_2.setPixmap(pixmap)

    def show_all_public_cards(self):
        all_public_cards = self.game.get_all_public_cards()
        self.ui.public_cards_1.setPixmap(QPixmap(f'./pokers/{all_public_cards[0][0]}-{all_public_cards[0][1]}.png'))
        self.ui.public_cards_2.setPixmap(QPixmap(f'./pokers/{all_public_cards[1][0]}-{all_public_cards[1][1]}.png'))
        self.ui.public_cards_3.setPixmap(QPixmap(f'./pokers/{all_public_cards[2][0]}-{all_public_cards[2][1]}.png'))
        self.ui.public_cards_4.setPixmap(QPixmap(f'./pokers/{all_public_cards[3][0]}-{all_public_cards[3][1]}.png'))
        self.ui.public_cards_5.setPixmap(QPixmap(f'./pokers/{all_public_cards[4][0]}-{all_public_cards[4][1]}.png'))

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

    def show_my_last_action_label(self, action):
        action_str = action_numpy_to_str(action)
        self.ui.my_last_action_label.setText(action_str)

    def show_pot_chips(self):
        self.ui.pot_chip.setText(str(self.game.pot_chip))

    def show_oppo_stage_chips(self):
        self.ui.oppo_stage_chip.setText(str(self.game.players[oppo_player_idx].cur_stage_chip))

    def show_oppo_last_action_label(self, action):
        oppo_action_str = action_numpy_to_str(action)
        self.ui.oppo_last_action_label.setText(oppo_action_str)

    def show_stage_label(self):
        if self.game.cur_stage == constants.preflop_stage:
            self.ui.stage_label.setText('preflop')
        elif self.game.cur_stage == constants.flop_stage:
            self.ui.stage_label.setText('flop')
        elif self.game.cur_stage == constants.turn_stage:
            self.ui.stage_label.setText('turn')
        elif self.game.cur_stage == constants.river_stage:
            self.ui.stage_label.setText('river')

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


if __name__ == '__main__':
    runner = NoLimitHoldemRunner(100, 864022131)
    runner.run()
