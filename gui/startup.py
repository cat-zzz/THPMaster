"""
@project: THPMaster
@File   : startup.py
@Desc   :
@Author : gql
@Date   : 2025/1/23 19:41
"""
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow

from gui import play
from gui.play import Ui_MainWindow


def show_image_2(ui: Ui_MainWindow):
    pixmap = QPixmap(':card/pokers/0-3.png')
    pixmap = pixmap.scaled(int(pixmap.width() * 1.2), int(pixmap.height() * 1.2), Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)
    ui.my_hands_1.setPixmap(pixmap)
    pixmap = QPixmap(':card/pokers/0-3.png')

    pixmap = pixmap.scaled(int(pixmap.width() * 1.2), int(pixmap.height() * 1.2), Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)
    ui.my_hands_2.setPixmap(pixmap)


def event_bind(ui: Ui_MainWindow):
    """
    绑定各个事件
    :param ui:
    :return:
    """
    ui.fold_btn.clicked.connect(lambda: show_image_2(ui))   # 学习PyQt


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui_1 = play.Ui_MainWindow()
    ui_1.setupUi(MainWindow)
    MainWindow.show()
    event_bind(ui_1)  # 绑定事件汇总
    sys.exit(app.exec_())
