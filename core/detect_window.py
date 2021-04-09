import sys
import os
import time
from os import path as os_path, makedirs

env_path = os_path.join(os_path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from ctypes import windll
from multiprocessing import Process

windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QUrl, QThread, QSize

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist

from multiprocessing import Queue, Lock
from core.detect import Detect_Process
from core.camera import CameraThread
from core.show_img import ShowImageThread
from core.tools import kill_pid


# TODO:  多线程语音播放
# TODO： 添加配置文件
# TODO: 询问师兄界面怎么设计,美化界面


def print_num(num):
    while True:
        print(str(num) * 10)
        time.sleep(0.1)


class QDetectWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.ui = Ui_MainWindow()
        # self.ui.setupUi(self)
        loadUi("E:/Master/PyQt5_Detection/lib/AppQt/detectwindow.ui", self)
        self.ui = self
        self.setFixedSize(self.width(), self.height())

        # 槽函数
        self.ui.detect_task_one_rbtn.clicked.connect(self.do_switch_task)
        self.ui.detect_task_two_rbtn.clicked.connect(self.do_switch_task)
        self.ui.detect_task_three_rbtn.clicked.connect(self.do_switch_task)
        self.ui.detect_task_four_rbtn.clicked.connect(self.do_switch_task)
        # 当前按钮
        self.current_rbtn = 0
        # ##################################################################

        # 进程列表
        self.process_num = 1
        self.process_list = [None] * self.process_num

        # 显示线程列表
        self.show_img_thread = None
        # 摄像机线程
        self.camera_thread = None
        self.img_queue = []
        self.target_queue = []
        self.create_queue()
        self.camera_task_thread()
        self.show_image_thread()

    def create_queue(self):
        for i in range(self.process_num):
            self.img_queue.append(Queue(360))  # 摄像头传给检测网络   编码   1：img
            self.target_queue.append(Queue(360))  # 检测网络传给展示线程   检测后的目标坐标  1：{'name': target}

    def camera_task_thread(self):
        # 相机获取图像
        self.camera_thread = CameraThread(self.ui, self.img_queue)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def show_image_thread(self):
        self.show_img_thread = ShowImageThread(self.ui, self.target_queue)
        self.show_img_thread.daemon = True
        self.show_img_thread.start()

    def detect_task_process(self):
        for i in range(len(self.img_queue)):
            self.process_list[i] = Detect_Process(self.img_queue[i], self.target_queue[i])
            self.process_list[i].daemon = True
            self.process_list[i].start()

    def is_alive_process(self):
        is_alive = True
        for i in self.process_list:
            if i is None:
                is_alive = False
        return is_alive

    # ============ 槽函数 ==============================
    def do_switch_task(self):
        if self.ui.detect_task_one_rbtn.isChecked():
            if self.current_rbtn != 1:
                self.current_rbtn = 1
                self.switch_task()

        elif self.ui.detect_task_two_rbtn.isChecked():
            if self.current_rbtn != 2:
                self.current_rbtn = 2
                self.switch_task()

        elif self.ui.detect_task_three_rbtn.isChecked():
            if self.current_rbtn != 3:
                self.current_rbtn = 3
                self.switch_task()
                print("333333333333333333333")
        elif self.ui.detect_task_four_rbtn.isChecked():
            if self.current_rbtn != 4:
                self.current_rbtn = 4
                self.switch_task()
                print("44444444444444444444444444")

    def switch_task(self):
        # 首先判断，进程是否存在，如果存在则关闭
        if self.is_alive_process():
            # 关闭线程
            for i in range(self.process_num):
                if self.process_list[i].is_alive():
                    print("关闭进程    ", self.process_list[i].pid)
                    kill_pid(self.process_list[i].pid)
                    self.process_list[i] = None
                    # 杀死了进程。
            # 暂停线程
            if self.camera_thread is not None:
                self.camera_thread.pause()
            for i in range(len(self.target_queue)):
                while not self.target_queue[i].empty():
                    self.target_queue.acquire()

        # 然后开启新的进程
        time.sleep(2)
        self.detect_task_process()
        self.camera_thread.resume()
        self.show_img_thread.resume()
        for i in range(self.process_num):
            print("创建进程   ", i, '  ', self.process_list[i].pid)

    # ============ 事件处理 =============================
    # 窗口关闭
    def closeEvent(self, event):
        pass

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
    form = QDetectWindow()  # 创建窗体
    form.show()
    sys.exit(app.exec_())
