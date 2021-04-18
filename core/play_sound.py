# -*- coding: utf-8 -*-
# @Time    : 2021/4/18 21:21
# @Author  : Wang.Zhihui
# @Email   : w-zhihui@qq.com
# @File    : play_sound.py
# @Software: PyCharm
import time
from threading import Thread

import pyttsx3
from core.tools import mp3_path

"""
pyttsx3默认为英文引擎
改为中文：
engine = pyttsx3.init()
engine.setProperty("voice","HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
engine.say("你好吗？")
engine.runAndWait()

"""


class PlaySound(Thread):
    def __init__(self):
        super().__init__()
        self.is_playing = False
        self.mp3_path = mp3_path
        self.engine = pyttsx3.init()
        self.engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
        self.label_text = None
        self._isPause = True
        # self.cond = QWaitCondition()

    def pause(self):
        self.label_text = None
        self._isPause = True

    def resume(self):
        self.label_text = None
        self._isPause = False

    def run(self):
        while True:
            if not self._isPause:
                if self.label_text:
                    print("self.label_text", self.label_text)
                    self.is_playing = True
                    self.engine.say(self.label_text)
                    self.engine.runAndWait()
                    time.sleep(1)
            else:
                print("声音睡眠啦")
                self.is_playing = False
                time.sleep(0.02)
