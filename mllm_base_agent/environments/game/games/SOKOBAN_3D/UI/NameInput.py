# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '. 4\Grafkom\Proyek\SOKOBAN\UI\NameInput.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(400, 76)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(-60, 40, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 211, 16))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(230, 10, 161, 20))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(10)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accepted)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Name Input Dialog"))
        self.label.setText(_translate("Dialog", "Please Enter Your Name:"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

