# from PyQt5 import QtWidgets, QtCore, QtGui
import tkinter ##沒有這行不能跑
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5 import QtCore, QtGui
import sys
# import MainWindow as ui
import ui_cvdl as ui
import os

from Q1.Q1 import Question1
from Q2.Q2 import Question2
from Q3.Q3 import Question3
from Q4.Q4 import Question4
from Q5.Q5 import Question5


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.image1Path = None
        self.image2Path = None
        # Load data
        self.pushButtonLoadFolder.clicked.connect(self.selectDir)
        self.pushButtonLoadImageL.clicked.connect(self.getImLPath)
        self.pushButtonLoadImageR.clicked.connect(self.getImRPath)
        
        
        # self.pushButtonFindCorners.clicked.connect(lambda: Q1Object.test(self.dirName))
        # Question 1
        self.pushButtonFindCorners.clicked.connect(lambda: Q1Object.findCorner(self.dirName))
        self.pushButtonFindIntrinsicMatrix.clicked.connect(lambda: Q1Object.findIntrinsic(self.dirName))
        self.pushButtonFindExtrinsicMatrix.clicked.connect(lambda: Q1Object.findExtrinsic(self.dirName, self.spinBox.value()))
        # self.pushButtonFindExtrinsicMatrix.clicked.connect(lambda: Q1Object.findExtrinsic(self.dirName, self.comboBoxFindExtrinsic.currentText()))
        self.pushButtonFindDistortionMatrix.clicked.connect(Q1Object.findDistortion)
        self.pushButtonShowUndistortedResult.clicked.connect(lambda: Q1Object.showUndistortion(self.dirName))

        # Question 2
        self.pushButtonShowWordsOnBoard.clicked.connect(lambda: Q2Object.onBoard(self.dirName, self.textEdit.toPlainText()))
        self.pushButtonShowWordsVertically.clicked.connect(lambda: Q2Object.verticalOnBoard(self.dirName, self.textEdit.toPlainText()))

        # Question 3
        self.pushButtonShowStereoDisparityMap.clicked.connect(lambda: Q3Object.stereoDisparityMap(self.ImLPath, self.ImRPath))

        # Question 4
        # Load data
        self.pushButtonLoadImage1.clicked.connect(self.getImage1Path)
        self.pushButtonLoadImage2.clicked.connect(self.getImage2Path)

        self.pushButtonShowKeypoints.clicked.connect(lambda: Q4Object.showKeypoints(self.image1Path))
        self.pushButtonShowMatchedKeypoints.clicked.connect(lambda: Q4Object.showMatchedKeypoints(self.image1Path, self.image2Path))
        
        # Question 5
        # load data
        self.pushButtonLoadImage.clicked.connect(self.getImagePath)

        # question
        self.pushButtonShowDataAugmentation.clicked.connect(Q5Object.showDataAugmentation)
        self.pushButtonShowModelStructure.clicked.connect(Q5Object.showModelStructure)
        self.pushButtonShowAccuracyAndLoss.clicked.connect(self.showAccuracyAndLoss)
        self.pushButtonShowInference.clicked.connect(lambda: self.showInference(self.imagePath))

    def selectDir(self):
        self.dirName = QtCore.QDir.toNativeSeparators(QFileDialog.getExistingDirectory(None, caption='Select a folder:', directory='C:\\', options=QFileDialog.ShowDirsOnly))
    
    def selectFile(self):
        fileName = QtCore.QDir.toNativeSeparators(QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get turple[0] which is file name
        return fileName
    
    def getImagePath(self):
        self.imagePath = self.selectFile()
        pixmap = QtGui.QPixmap(self.imagePath)
        pixmap = pixmap.scaled(128,128,QtCore.Qt.KeepAspectRatio) # resize img to 128*128
        self.photo.setPixmap(pixmap) # show the image on the UI
        print(self.imagePath)

    def showAccuracyAndLoss(self):
        Q5Object.makeAccuracyAndLoss()
        # self.photo.setPixmap(QtGui.QPixmap('Q5/result.png'))

    def showInference(self, imgPath):
        class_names = Q5Object.showInference(imgPath)
        # print(class_names)
        self.textArea.setText('Predicted = ' + class_names)
        # self.textArea.setText('Confidence = ' + str(conf) + '\n' + 'Prediction Label: ' + label)

    def getImLPath(self):
        self.ImLPath = self.selectFile()
    
    def getImRPath(self):
        self.ImRPath = self.selectFile()
    
    # overide to force exit
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        os._exit(0)
    def selectFile(self):
        fileName = QtCore.QDir.toNativeSeparators(QFileDialog.getOpenFileName(None, caption='Choose a File', directory='C:\\', filter='Image Files (*.png *.jpg *.bmp)')[0])  # get turple[0] which is file name
        return fileName
    
    def getImage1Path(self):
        self.image1Path = self.selectFile()
    
    def getImage2Path(self):
        self.image2Path = self.selectFile()
    
    # overide to force exit
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        os._exit(0)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    Q1Object = Question1() # worked
    Q2Object = Question2() # worked
    Q3Object = Question3() # bigsur version mac can't do resizeWindow 
    Q4Object = Question4()
    Q5Object = Question5()
    window = Main()
    window.show()
    sys.exit(app.exec_())