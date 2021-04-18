import cv2
from threading import Thread
from PyQt5.QtCore import QThread, QWaitCondition
import time


# from multiprocessing import Process, Queue

# img_queue = Queue(60)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel


class CameraThread(QThread):
    """
    摄像机获取图像
    """
    def __init__(self, ui_obj, img_queue, src=0):
        super(CameraThread, self).__init__()
        # self.setName("CameraThread")
        self.img_queue = img_queue   # 将获取图像送给检测进程的队列，由于检测进程可能有多个因此该处为队列
        self.frame = None
        self.status = None
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)   # 打开摄像机
        self.synchron_num = 0    # 进程同步标志位
        self._isPause = True     # 是否睡眠标志位

    def pause(self):      # 可以调用该函数使摄像机获取图像睡眠
        self._isPause = True

    def resume(self):    # 唤醒
        self._isPause = False
        # self.cond.wakeAll()

    def update(self):   # 读取图像
        # Read the next frame from the stream in a different thread
        if self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
        time.sleep(.01)

    def add_frame_to_queue(self):
        """
        将获取的图像推送队列
        :return:
        """
        self.update()
        if self.capture.isOpened() and self.frame is not None:
            # 如果队列满了，那么需要把前面的图像清除一张，然后再加入新的
            for i in range(1):
                if self.img_queue[i].full():
                    # self.img_lock.acquire()
                    self.img_queue[i].get()  # 清除一张
                    self.img_queue[i].put({self.synchron_num: self.frame})  # 加入新的
                    # self.img_lock.release()
                    print("满队列加入队列：", self.frame.shape)
                time.sleep(0.03)
            else:  # 否则直接加入新的
                # self.img_lock.acquire()
                for i in range(len(self.img_queue)):
                    self.img_queue[i].put({self.synchron_num: self.frame})
                # self.img_lock.release()
                print("加入队列：", self.frame.shape)
                time.sleep(0.02)
            if self.synchron_num >= 360:   # 设置同步编号
                self.synchron_num = 0
            else:
                self.synchron_num += 1

    def run(self):
        while True:
            if not self._isPause:
                self.add_frame_to_queue()
            else:
                print("摄像头睡眠啦")
                time.sleep(0.1)
