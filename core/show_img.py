import time
from threading import Thread

import cv2
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QSize

from core.tools import plot_one_box, cv_put_text


class ShowImageThread(Thread):
    def __init__(self, ui_obj, target_queue):
        # 初始化展示图片的窗口
        super().__init__()
        self.ui_obj = ui_obj
        self.target_queue = target_queue
        self.show_img_lbl = QLabel(ui_obj)
        self.ui_obj.detect_show_img.addWidget(self.show_img_lbl)

        # 创建数据队列,模仿队列数据结构，先进先出
        self.queue = [[]] * len(target_queue)
        # 里面的数据格式
        # 假设有三个检测进程
        # [[[图像次序0， 原始图像， 检测后的标签和坐标{标签：坐标}],[图像次序1， 原始图像， 检测后的标签和坐标{标签：坐标}], ....],
        # [[[图像次序0， 原始图像， 检测后的标签和坐标{标签：坐标}],[图像次序1， 原始图像， 检测后的标签和坐标{标签：坐标}], ....],
        # [[[图像次序0， 原始图像， 检测后的标签和坐标{标签：坐标}],[图像次序1， 原始图像， 检测后的标签和坐标{标签：坐标}], ....]]
        # 因此要拿到每一个进程的图像次序需要    self.queue[i][0][0]
        # 拿到一个进程中的图像： self.queue[i][0][1]

        self._isPause = True
        # self.cond = QWaitCondition()

    def pause(self):
        self._isPause = True

    def resume(self):
        self._isPause = False
        # self.cond.wakeAll()

    @staticmethod
    def cv2img(cv_img):
        """
        将cv2的BGR转成qt使用的RGB
        :param cv_img: cvBGR图像
        :return: qt可以展示的图像
        """
        img_cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pyqt = QImage(img_cv_rgb[:], img_cv_rgb.shape[1], img_cv_rgb.shape[0], img_cv_rgb.shape[1] * 3,
                          QImage.Format_RGB888)
        pixmap = QPixmap(img_pyqt)
        return pixmap

    def show(self, cv_img):

        w = self.show_img_lbl.width()
        h = self.show_img_lbl.height()

        pixmap = self.cv2img(cv_img)
        pixmap = pixmap.scaled(QSize(w, h), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.show_img_lbl.setPixmap(pixmap)

    def get_img(self):
        max_num = -1
        for i in range(len(self.target_queue)):
            det = self.target_queue[i].get()
            self.queue[i].append(det)
            if self.queue[i][0][0] > max_num:
                max_num = self.queue[i][0][0]
            print(self.queue[i][0][0], end=" ")
        print("")
        # 如果三个进程中图像序列不一致，则小的序列丢弃，
        # 不管最大的序列和最小的序列相差多少，多几个循环就可以把三个进程的第一个元素的图像序列一样。
        for i in range(len(self.queue)):
            if self.queue[i][0][0] < max_num:
                self.queue[i].pop(0)

    def run(self):
        while True:
            if not self._isPause:
                t0 = time.time()
                self.get_img()
                tar = self.queue[0][0]
                is_ok = True
                for i in range(len(self.queue)):
                    if self.queue[i][0][0] != tar[0]:
                        is_ok = False
                print("is_ok:", is_ok)
                if is_ok:
                    self.img0 = self.queue[i][0][1]
                    for i in range(len(self.target_queue)):
                        synchron_num, _, target = self.queue[i].pop(0)
                        print(synchron_num, "---->", target)
                        # target = {'name': 坐标}
                        names = []
                        for name, xyxy in target.items():
                            if xyxy:
                                plot_one_box(xyxy,  self.img0, label=name, line_thickness=3)
                                names.append(name)
                    self.img0 = cv_put_text(self.img0, names)
                    self.show(self.img0)
                    time.sleep(0.01)
                print("图片展示环节：{:.2f}".format(time.time() - t0))
            else:
                time.sleep(0.1)
