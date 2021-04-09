import os
from multiprocessing import Process
import time

from lib.yolov5 import detect


# target_queue = Queue(60)    # 两秒


class Parameters:
    def __init__(self, weights, img_size, conf_thres, iou_thres, device,
                 agnostic_nms, augment, project, name):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.project = project
        self.name = name


class Detect_Process(Process):
    def __init__(self, img_queue, target_queue):
        Process.__init__(self)

        self.img_queue = img_queue
        self.target_queue = target_queue
        weights_file_path = os.path.join(os.path.dirname(__file__), '../db/yolov5s.pt')
        print(weights_file_path)
        # 初始化模型
        self.opt = Parameters(
            weights=weights_file_path,
            # 四种： 1、图像的文件本身，2、图像所在的文件夹，3、已经读取的图像 source=1 4、摄像头  source=0
            img_size=640,
            conf_thres=0.25,
            iou_thres=0.45,
            device='',
            agnostic_nms='',
            augment='',
            project='runs/detect',
            name='exp'
        )

        # time.sleep(.01)

    def run(self):
        det = detect.DetectCla(self.opt)
        while True:
            t0 = time.time()
            print("running")
            if not self.img_queue.empty():
                print("队列不为空:", self.img_queue.qsize())
                # self.img_lock.acquire()
                dic = self.img_queue.get()
                if dic is None:
                    continue
                (synchron_num, img0), = dic.items()
                # self.img_lock.release()
                ret = det.detect_updates(img0)
                self.target_queue.put((synchron_num, img0, ret))
                # time.sleep(.01)
            else:
                print("队列为空")
                time.sleep(.02)
            print("检测环节使用了：{:.2f}".format(time.time() - t0))
