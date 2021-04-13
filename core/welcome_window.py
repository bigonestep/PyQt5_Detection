import sys
import os
from os import path as os_path, makedirs

env_path = os_path.join(os_path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QApplication, QMainWindow
from core.detect_window import QDetectWindow


class QWelcomeWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ------------------------------设置全屏----------------------------------
        self.desktop = QApplication.desktop()
        # 获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.screen_height = self.screenRect.height()
        self.screen_width = self.screenRect.width()
        ui_file_path = os.path.join(os.path.dirname(__file__), '../lib/AppQt/welcomewindow.ui')
        loadUi(ui_file_path, self)
        self.ui = self
        # self.ui.resize(self.screen_width, self.screen_height)    # 设置软件启动为全屏

        self.main_detect_window = None   # 初始化检测界面类为None
        # ----------------------------四个检测任务按钮的槽函数------------------------------------
        self.ui.detect_task_one_btn.clicked.connect(self.do_task1_window)
        self.ui.detect_task_two_btn.clicked.connect(self.do_task2_window)
        self.ui.detect_task_three_btn.clicked.connect(self.do_task3_window)
        self.ui.detect_task_four_btn.clicked.connect(self.do_task4_window)

        self.current_btn = -1       # 记录当前按钮，初始化为-1，即没有一个按钮按下

        # ##################################################################

    # ============ 槽函数 ==============================
    def do_task1_window(self):
        if self.current_btn != 0:      # 如果该按钮已经按下过，再次按下则无效
            self.current_btn = 0
            print("1111111111111111111111")
            # 开启检测界面
            if self.main_detect_window is None:         # 如果检测界面还没有初始化则初始化
                self.main_detect_window = QDetectWindow(self)
                self.main_detect_window.start_task(task=0)  # 开启yolo检测进程
                self.main_detect_window.show()  # 展示界面
            else:
                print("一一一一一一一一一一一一一一一一一一一一")
                # try:
                self.main_detect_window.close_process()
                # 如果检测界面初始化过了，说明可能有其他的yolo进程在运行，那么先杀死进程
                # except
                self.main_detect_window.start_task(task=0)  # 开启新的yolo检测进程
                self.main_detect_window.show()

    def do_task2_window(self):
        # 同上
        if self.current_btn != 1:
            self.current_btn = 1
            print("22222222222222222")
            # 开启检测界面
            if self.main_detect_window is None:
                self.main_detect_window = QDetectWindow(self)
                self.main_detect_window.start_task(task=1)
                self.main_detect_window.show()
            else:
                print("二二二二二二二二二二二二二二二二二")
                # try:
                self.main_detect_window.close_process()
                # except
                self.main_detect_window.start_task(task=1)
                self.main_detect_window.show()
    def do_task3_window(self):
        if self.current_btn != 2:
            self.current_btn = 2
            print("3333333333333333333333")
            # 开启检测界面
            if self.main_detect_window is None:
                self.main_detect_window = QDetectWindow(self)
                self.main_detect_window.start_task(task=2)
                self.main_detect_window.show()
            else:
                print("三三三三三三三三三三三三三三三三三三三三")
                # try:
                self.main_detect_window.close_process()
                # except
                self.main_detect_window.start_task(task=2)
                self.main_detect_window.show()
    def do_task4_window(self):
        if self.current_btn != 3:
            self.current_btn = 3
            print("4444444444444444444444444444444")
            # 开启检测界面
            if self.main_detect_window is None:
                self.main_detect_window = QDetectWindow(self)
                self.main_detect_window.start_task(task=3)
                self.main_detect_window.show()
            else:
                print("四四四四四四四四四四四四四四四四四四四四四四四四")
                # try:
                self.main_detect_window.close_process()
                # except
                self.main_detect_window.start_task(task=3)
                self.main_detect_window.show()

    # def change_task(self,):
    # ============ 事件处理 =============================
    # 窗口关闭
    def closeEvent(self, event):
        super().closeEvent(event)

    # def resizeEvent(self, QResizeEvent):
    #     print('窗口变化', QResizeEvent.size())
    #     w = self.ui.show_img_lbl.width()
    #     h = self.ui.show_img_lbl.height()
    #
    #
    #     self.pp = self.pixmap.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     self.label.setPixmap(self.pp)
    # self.pp2 = self.pixmap2.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    # self.label_2.setPixmap(self.pp2)


#  ============窗体测试程序 ================================
if __name__ == "__main__":  # 用于当前窗体测试
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = QWelcomeWindow()  # 创建窗体
    form.show()
    sys.exit(app.exec_())
