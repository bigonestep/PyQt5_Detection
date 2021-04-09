import sys
from os import path as os_path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from core.detect_window import QDetectWindow

env_path = os_path.join(os_path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = QDetectWindow()  # 创建窗体
    form.show()
    sys.exit(app.exec_())
