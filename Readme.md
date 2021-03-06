# 基于PyQt5与yolov5的物体检测软件
本软件是依照与本实验的的一个项目进行开发的，是按照该项目的需求进行编写，一个进程只进行一个物体类型的检测。

其中相机采集和图像展示使用的是多线程，进程之间采用的是队列通信技术，其中还涉及数据同步，图像处理等技术。

本项目的两个界面都是全屏展示（根据项目需求），因此如个人有需求可以更改为自己合适的窗口大小。

本项目采用纯python语言编写，可以跨平台使用，需要注意的是如果你的电脑没有GPU，那么你可能需要修改一下检测的代码。 

选择界面和检测界面为一个进程，该进程还包括一个相机采集线程，检测后图像显示线程，声音播报检测目标线程，还有一个检测进程，进程间采用队列的方法进行通信。

代码结构：  
1. core中核心代码  
    * camera为相机线程
    * detect为检测进程
    * play_sound为声音播放线程
    * detect_window为检测软件界面
    * show_img为检测结果视频展示线程
    * welcome_window为任务切换界面
    * tools为几个工具函数
2. db存放的为几个任务的权重文件  
3. lib
    * yolov5 为yolov5的检测代码，其中detect.py为对yolov5的修改封装
    * AppQt为Qt创建的界面项目，采用Qt创建界面是因为使用较为方便
4. requriements.txt为所需的第三方库，注意该文件是个人写的，因此不能使用pip直接安装。
5. start.py为项目启动文件

运行项目： python start.py

权重文件命名：  
yolov5s_<主任务>_<下面的进程>

比如一个主任务1下面有2个检测进程  
那么四个权重文件的命名为:  
yolov5s_0_0.pt  
yolov5s_0_1.pt  
注意： 这里可以使用其他较重的权重网络，需要稍微修改一下代码。