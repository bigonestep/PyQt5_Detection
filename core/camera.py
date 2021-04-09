import cv2
from threading import Thread
from PyQt5.QtCore import QThread, QWaitCondition
import time


# from multiprocessing import Process, Queue

# img_queue = Queue(60)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel


class CameraThread(QThread):
    def __init__(self, ui_obj, img_queue, src=0):
        super(CameraThread, self).__init__()
        self.img_queue = img_queue

        self.frame = None
        self.status = None
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.synchron_num = 0    # 进程同步标志位
        self._isPause = True
        # self.cond = QWaitCondition()

    def pause(self):
        self._isPause = True

    def resume(self):
        self._isPause = False
        # self.cond.wakeAll()

    def update(self):
        # Read the next frame from the stream in a different thread
        if self.capture.isOpened():
            (self.status, self.frame) = self.capture.read()
        time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def add_frame_to_queue(self):

        self.update()
        if self.capture.isOpened() and self.frame is not None:
            for i in range(1):
                if self.img_queue[i].full():
                    # self.img_lock.acquire()
                    self.img_queue[i].get()
                    self.img_queue[i].put({self.synchron_num: self.frame})
                    # self.img_lock.release()
                    print("满队列加入队列：", self.frame.shape)
                time.sleep(0.03)
            else:
                # self.img_lock.acquire()
                for i in range(len(self.img_queue)):
                    self.img_queue[i].put({self.synchron_num: self.frame})
                # self.img_lock.release()
                print("加入队列：", self.frame.shape)
                time.sleep(0.02)
            if self.synchron_num >= 360:
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
