import sys
import os
import time
from os import path as os_path

env_path = os_path.join(os_path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from PyQt5.uic import loadUi

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel

from multiprocessing import Queue
from core.detect import DetectProcess
from core.camera import CameraThread
from core.show_img import ShowImageThread
from core.tools import kill_pid
from core.play_sound import PlaySound


class QDetectWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        print("detect window pid:", os.getpid())
        self.parent = parent   # 获取上一个界面的对象
        ui_file_path = os.path.join(os.path.dirname(__file__), '../lib/AppQt/detectwindow.ui')
        loadUi(ui_file_path, self)
        self.ui = self
        self.task = 0
        # ------------------------------设置全屏----------------------------------
        self.desktop = QApplication.desktop()
        # 获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.screen_height = self.screenRect.height()
        self.screen_width = self.screenRect.width()
        # self.ui.resize(self.screen_width, self.screen_height)
        # ##################################################################

        # ---------------------------- 进程列表 ----------------------------
        self.process_num = 1     # 进程数
        # 注意：需要配置相应的yolo权重
        self.process_list = [None] * self.process_num   # 进程对象的列表，保存各个进程以便拿到各个进程的pid
        # ---------------------------- 进程间通信队列创建 ----------------------------
        self.img_queue = []
        self.target_queue = []
        self.create_queue()
        # ---------------------初始化 声音播放 -------------------
        self.playsound_thread = PlaySound()
        self.playsound_thread.daemon = True
        self.playsound_thread.start()
        # ---------------------------- 显示线程对象和 ----------------------------
        self.camera_thread = None
        self.camera_task_thread()  # 初始化显示线程
        # ----------------------摄像机线程对象--------------------------
        self.show_img_thread = None
        self.show_image_thread()  # 初始化摄像机线程


    def create_queue(self):
        for i in range(self.process_num):
            self.img_queue.append(Queue(360))  # 摄像头传给检测网络   编码   1：img
            self.target_queue.append(Queue(360))  # 检测网络传给展示线程   检测后的目标坐标
            # 编码  1：[img, label, target]

    def camera_task_thread(self):
        # 相机获取图像
        self.camera_thread = CameraThread(self.ui, self.img_queue)
        self.camera_thread.daemon = True    # 设置为守护线程
        self.camera_thread.start()

    def show_image_thread(self):
        self.show_img_thread = ShowImageThread(self.ui, self.target_queue)
        self.show_img_thread.daemon = True
        self.show_img_thread.start()

    def detect_task_process(self, task):
        for i in range(len(self.img_queue)):
            self.process_list[i] = DetectProcess(task, i, self.img_queue[i], self.target_queue[i])
            self.process_list[i].daemon = True    # 设置为守护进程，防止软件关闭之后产生僵尸进程
            self.process_list[i].start()

    def is_alive_process(self):
        """
        判断yolo检测进程是否存活
        :return:  存活为True
        """
        is_alive = True
        for i in self.process_list:
            if i is None:
                is_alive = False
        return is_alive

    def start_task(self, task):
        """
        开启新的进程
        :param task: 任务编号
        :return: None
        """
        time.sleep(0.1)
        self.detect_task_process(task)     # 开启yolo进程
        self.playsound_thread.resume()
        self.camera_thread.resume()        # 唤醒摄像机线程
        self.show_img_thread.resume()      # 唤醒显示线程
        for i in range(self.process_num):
            print("创建进程   ", i, '  ', self.process_list[i].pid)   # 打印进程pid

    def close_process(self):
        """
        关闭yolo进程以及睡眠摄像机和显示线程
        :return:  None
        """
        if self.is_alive_process():    # 判断进程是否存活，若存活则杀死
            # 关闭线程
            for i in range(self.process_num):
                if self.process_list[i].is_alive():
                    print("关闭进程    ", self.process_list[i].pid)
                    kill_pid(self.process_list[i].pid)     # 杀死了进程。
                    self.process_list[i] = None            # 将进程对象置为None，方便系统回收
            # 暂停线程
            if self.playsound_thread is not None:
                self.playsound_thread.pause()
            if self.camera_thread is not None:
                self.camera_thread.pause()     # 睡眠摄像机线程
            if self.show_img_thread is not None:
                self.show_img_thread.pause()
            for i in range(len(self.target_queue)):  # 倘若队列中还有数据则清空
                while not self.target_queue[i].empty():
                    self.target_queue.acquire()

    # ============ 事件处理 =============================
    # 窗口关闭
    def closeEvent(self, event):
        self.close_process()       # 关闭窗口则关闭进程
        self.parent.current_btn = -1   # 把按钮标志设置为  -1
        super().closeEvent(event)


    # def resizeEvent(self, QResizeEvent):
    #     print('窗口变化', QResizeEvent.size())
    #     w = self.ui.show_img_lbl.width()
    #     h = self.ui.show_img_lbl.height()
    #
    #
    #     self.pp = self.pixmap.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    #     self.label.setPixmap(self.pp)


