import random
import sys

import cv2
import os
from yaml import load, FullLoader

import numpy as np

yaml_file_path = os.path.join(os.path.dirname(__file__), '../conf/classes.yaml')  # 配置文件的路径


def get_yaml_data(yaml_file):
    """
    读取配置文件
    :param yaml_file:  路径
    :return:  配置文件内容
    """
    # 打开yaml文件
    print("***获取yaml配置文件数据***")
    print("配置文件路径：", yaml_file)
    if os.path.exists(yaml_file) and (".yaml" in yaml_file or ".yml" in yaml_file):
        file = open(yaml_file, 'r', encoding="utf-8")
        file_data = file.read()
        file.close()
        # 将字符串转化为字典或列表
        print("***转化yaml数据为字典或列表***")
        data = load(file_data, Loader=FullLoader)
        return data
    else:
        return None


label_text = get_yaml_data(yaml_file_path)["names"]   # 在导入该文件的时候执行，获取配置文件分类


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    画框函数
    :param x:  坐标格式 [x1, y1, x2, y2]
    :param img:  图片
    :param color:  框的颜色，若不设置则随机产生
    :param label:  标签编号，根据/db/classes.yaml可以获取编号的具体是什么
    :param line_thickness:  线和字体的粗细
    :return:
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]  # 产生随机框的颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)  # 获取框的大小

    if label:  # 如果标签存在则画标出
        tf = max(tl - 1, 1)  # font thickness
        if label_text is not None:
            print(label_text[int(label)])
            t_size = cv2.getTextSize(label_text[int(label)], 0, fontScale=tl / 3, thickness=tf)[0]
        else:
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        if label_text is not None:  # filled
            cv2.putText(img, label_text[int(label)], (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                        thickness=tf, lineType=cv2.LINE_AA)
        else:
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def kill_pid(pid):
    """
    杀死一个进程，原理模拟命令行杀死进程，先判断系统，然后调用杀死进程命令
    :param pid:   进程的pid
    :return:
    """
    if 'win' in sys.platform.lower():
        find_kill = 'taskkill -f -pid %s' % pid
    elif 'linux' in sys.platform.lower():
        find_kill = 'kill -9 %s' % pid
    else:
        raise OSError("请自行添加该系统下杀死进程的命令")
    print(find_kill)
    result = os.popen(find_kill)
    print(result)


def cv_put_text(img0, names):
    """
    在图像上画出一个横条展示标签
    :param img0:  图像
    :param names:  标签代码为一个列表，比如  ['0', '2']
    :return: 设置好文字的图片
    """
    text = ''
    if label_text:
        for name in names:
            text += label_text[int(name)] + "  "  # 先拼接所有的标签文字
    # text = "person  cell phone  img_height"
    img_height, img_width, _ = img0.shape   # 获取图像的长宽
    font = cv2.FONT_HERSHEY_SIMPLEX   # 设置字体
    # 480, 640, 3
    # 先获取文本框大小
    # 文字，字体，字体大小，粗细
    text_width, text_height = cv2.getTextSize(text, font, 1, 2)[0]   # 获取文字框的大小，方便调整文字在图像中的位置
    print("text_size", text_width, text_height)
    # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    x = img_height * 3 // 4      # 计算出文字在图像中的位置     长为图像靠下的四分之三位置
    y = (img_width - text_width) // 2         # 宽度居中

    img0_ndarray = np.array(img0)    # 把图像转为numpy，方便直接赋值，产生背景

    img0_ndarray[x - text_height - text_height: x + text_height,
    y - y // 2: y + text_width + y // 2,
    :] = [27, 0, 221]    # 将文字的背景设置红色
    # print("----------------",type(img0_ndarray.tolist()))
    img0 = cv2.putText(img0_ndarray.astype(np.uint8), text, ((img_width - text_width) // 2, img_height * 3 // 4), font,
                       1, (255, 255, 255), 3)   # 设置文字
    return img0
