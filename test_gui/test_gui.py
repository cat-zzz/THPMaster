"""
@project: THPMaster
@File   : test_gui.py
@Desc   :
@Author : gql
@Date   : 2025/1/15 14:49
"""
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
import demo1


def click_success():
    print("点击事件")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = demo1.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.pushButton.clicked.connect(click_success)
    sys.exit(app.exec_())
