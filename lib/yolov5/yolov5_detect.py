import argparse
import time
from pathlib import Path

import cv2
import torch
from torch._C import set_flush_denormal
import torch.backends.cudnn as cudnn
from numpy import random

import sys
from os import path as os_path
env_path = os_path.join(os_path.dirname(__file__), '.')
if env_path not in sys.path:
    sys.path.append(env_path)


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, DealImage
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


# 把模型初始化拿出来作为一个函数 detect_init()xl
# 把检测更新作为一个函数   update_detect();
class DetectCla:
    def __init__(self, opt):
        self.opt = opt
        self.weights, self.imgsz = opt.weights, opt.img_size
        # Initialize
        # set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.detect_init()

    def detect_init(self):
        with torch.no_grad():
            # Load model
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            if self.half:
                self.model.half()  # to FP16

            img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
            _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
            # Get names and colors
            # names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    def detect_updates(self, img0):
        with torch.no_grad():
            res = {}
            im0s = img0
            img = DealImage(img0).img

            t0 = time.time()

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS

            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                       classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # t2 = time_synchronized()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = im0s.copy()
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # print("len(det):", len(det))
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        res[str(int(cls.item()))] = [int(i.item()) for i in xyxy]
            print('Done. (%.3fs)' % (time.time() - t0))
            return res













































'''
def detect_init(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    if source.isnumeric():
        webcam = int(source)



    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    return model
# ----------------------

def update_detect(opt, model):
    ret = []
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    if source.isnumeric():
        webcam = int(source)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA



    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam == 0:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    elif webcam == 1 and not opt.img:
        dataset = DealImage(opt.img, img_size=imgsz, stride=stride)  # 输入的为cv读取的原始图像
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference



    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        # print("----->",path, im0s, vid_cap)  # path, [[[]]] , None 
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # print('\n',{str(int(cls.item())): xywh}, '\n')
                    ret.append({str(int(cls.item())): xywh})

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    return ret




def run(weights='E:/mater/PyQt5_Detection/lib/yolov5/yolov5s.pt', 
        source='E:/mater/PyQt5_Detection/lib/yolov5/data/images',
        # 四种： 1、图像的文件本身，2、图像所在的文件夹，3、已经读取的图像 source=1 4、摄像头  source=0
        img=None,          # 对应输入的为图像的数值
        img_size=640, 
        conf_thres=0.25, 
        iou_thres=0.45, 
        device='', 
        view_img='', 
        save_txt='', 
        save_conf='',
        nosave='', 
        classes=0, 
        agnostic_nms='', 
        augment='', 
        update='', 
        project='runs/detect', 
        name='exp', 
        exist_ok=''):
# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='E:/mater/PyQt5_Detection/lib/yolov5/yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='E:/mater/PyQt5_Detection/lib/yolov5/data/images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt',default='', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # opt = parser.parse_args()
    opt = Parameters(weights, source, img_size, conf_thres, iou_thres, device, view_img, save_txt, save_conf,
        nosave, classes, agnostic_nms, augment, update, project, name, exist_ok)

    # print("------------------------")
    print(opt)
    # print("------------------------")



    check_requirements(exclude=('pycocotools', 'thop'))
    model = detect_init(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                ret = detect(opt)
                strip_optimizer(opt.weights)
        else:
            ret = detect(opt)
    return ret 
'''