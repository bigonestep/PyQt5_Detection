import os
from multiprocessing import Process
import time

from lib.yolov5 import yolov5_detect


class Parameters:
    """
    yolo检测网络的各项参数
    """

    def __init__(self, weights, img_size, conf_thres, iou_thres, device,
                 agnostic_nms, augment, classes=None):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.classes = classes  # 0代表一类


class DetectProcess(Process):
    def __init__(self, task, sub_task, img_queue, target_queue):
        """
        初始化检测进程
        :param task:  任务编号   必须从0开始
        :param sub_task:   # 子任务编号，必须从0开始
        :param img_queue:  # 摄像机传送图片队列
        :param target_queue:  检测之后的结果送到展示线程的队列
        """
        Process.__init__(self)
        self.task = task
        self.sub_task = sub_task
        self.img_queue = img_queue
        self.target_queue = target_queue
        print('../db/yolov5s_%d_%d.pt' % (self.task, self.sub_task))
        weights_file_path = os.path.join(os.path.dirname(__file__),
                                         '../db/yolov5s_%d_%d.pt' % (self.task, self.sub_task))
        # 获取权重路径
        print(weights_file_path)
        # 初始化模型
        self.opt = Parameters(     # 根据yolo网络配置
            weights=weights_file_path,
            img_size=640,
            conf_thres=0.25,
            iou_thres=0.45,
            device='',
            agnostic_nms='',
            augment='',
        )

    def run(self):
        det = yolov5_detect.DetectCla(self.opt)   # 初始化yolo检测网络
        while True:
            t0 = time.time()
            print("running")
            if not self.img_queue.empty():    # 如果图像队列不为空
                print("队列不为空:", self.img_queue.qsize())
                # self.img_lock.acquire()
                dic = self.img_queue.get()         # 首先获取图像，{图像编号：图像}
                if dic is None:
                    continue
                (synchron_num, img0), = dic.items()      # 解包
                # self.img_lock.release()
                ret = det.detect_updates(img0)        # 对图片进行检测
                self.target_queue.put((synchron_num, img0, ret))   # 将检测结果推送到展示队列
                # time.sleep(.01)
                print("检测环节使用了：{:.2f}".format(time.time() - t0))
            else:
                print("队列为空")
                time.sleep(.02)

