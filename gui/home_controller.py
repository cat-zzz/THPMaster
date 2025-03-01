"""
@project: THPMaster
@File   : home_controller.py
@Desc   :
@Author : gql
@Date   : 2025/2/28 16:11
"""
import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow

from gui.human_ai_ui import Ui_MainWindow as HumanAI_UI
from gui.play_controller_3 import NoLimitHoldemRunner


class HomeUIRunner(QMainWindow, HumanAI_UI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.event_bind()
        self.runner = None

    def event_bind(self):
        self.confirm_btn.clicked.connect(self.confirm_btn_click_event)

    def confirm_btn_click_event(self):
        """
        打开人机对弈页面
        """
        self.runner = NoLimitHoldemRunner(100, 864022131)
        # self.runner.show_window()
        self.runner.window_closed.connect(self.on_runner_closed)
        self.runner.run()
        self.hide()
        pass

    def on_runner_closed(self):
        """响应窗体2关闭事件"""
        self.show()  # 重新显示窗体1
        self.runner = None  # 释放引用


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = HomeUIRunner()
    window.show()
    sys.exit(app.exec_())
    pass
