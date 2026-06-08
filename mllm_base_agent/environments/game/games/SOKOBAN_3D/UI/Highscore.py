# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '. 4\Grafkom\Proyek\SOKOBAN\UI\Highscore.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_HighscoreWindow(object):
    def setupUi(self, HighscoreWindow):
        HighscoreWindow.setObjectName("HighscoreWindow")
        HighscoreWindow.setWindowModality(QtCore.Qt.NonModal)
        HighscoreWindow.resize(547, 388)
        HighscoreWindow.setMinimumSize(QtCore.QSize(547, 388))
        HighscoreWindow.setMaximumSize(QtCore.QSize(547, 388))
        self.centralwidget = QtWidgets.QWidget(HighscoreWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(10, 90, 256, 221))
        self.listView.setObjectName("listView")
        self.listView_2 = QtWidgets.QListView(self.centralwidget)
        self.listView_2.setGeometry(QtCore.QRect(280, 90, 256, 221))
        self.listView_2.setObjectName("listView_2")
        self.BtnBack = QtWidgets.QPushButton(self.centralwidget)
        self.BtnBack.setGeometry(QtCore.QRect(440, 320, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        self.BtnBack.setFont(font)
        self.BtnBack.setObjectName("BtnBack")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 50, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(340, 50, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 10, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        HighscoreWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(HighscoreWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 547, 21))
        self.menubar.setObjectName("menubar")
        HighscoreWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(HighscoreWindow)
        self.statusbar.setObjectName("statusbar")
        HighscoreWindow.setStatusBar(self.statusbar)

        self.retranslateUi(HighscoreWindow)
        QtCore.QMetaObject.connectSlotsByName(HighscoreWindow)

    def retranslateUi(self, HighscoreWindow):
        _translate = QtCore.QCoreApplication.translate
        HighscoreWindow.setWindowTitle(_translate("HighscoreWindow", "Highscore"))
        self.BtnBack.setText(_translate("HighscoreWindow", "Close"))
        self.label.setText(_translate("HighscoreWindow", "Story"))
        self.label_2.setText(_translate("HighscoreWindow", "Time Attack"))
        self.label_3.setText(_translate("HighscoreWindow", "HIGHSCORES"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    HighscoreWindow = QtWidgets.QMainWindow()
    ui = Ui_HighscoreWindow()
    ui.setupUi(HighscoreWindow)
    HighscoreWindow.show()
    sys.exit(app.exec_())

