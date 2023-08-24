from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QVBoxLayout, QWidget, QDialog, QLineEdit, QPushButton, QHBoxLayout
from ultralytics import YOLO
import numpy as np
from PIL import Image
import folium
from folium.plugins import HeatMap
import pickle
import io
import os
import webbrowser
import random
os.chdir('SOLUTION')

class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        self.imageWidgets = []  # Store the image label widgets

        self.detectionsLabel = QLabel(self)  # Create the QLabel for detections
        self.detectionsLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.detectionsLabel.setFixedWidth(200)  # Adjust the width as needed
        self.detectionsLabel.setStyleSheet("background-color: rgba(255, 255, 255, 200)")  # Optional styling
        self.statusBar().addPermanentWidget(self.detectionsLabel)  

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)

        self.scrollContent = QWidget()
        self.scrollLayout = QHBoxLayout(self.scrollContent)
        self.scrollLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins

        self.scrollArea.setWidget(self.scrollContent)
        self.scrollArea.setVisible(False)

        self.welcomeLabel = QLabel("Добро пожаловать! Загрузите фото с ягодами")
        self.welcomeLabel.setAlignment(Qt.AlignCenter)
        self.welcomeLabel.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout = QVBoxLayout()  # Create a vertical layout
        layout.addWidget(self.welcomeLabel)  # Add the welcome label
        layout.addWidget(self.scrollArea)  # Add the scroll area to the layout

        central_widget = QWidget(self)
        central_widget.setLayout(layout)  # Set the layout for the central widget
        self.setCentralWidget(central_widget)


        self.createActions()
        self.createMenus()

        self.setWindowTitle("Berry Viewer")
        self.resize(800, 600)

        self.model = YOLO('weights/best.pt')
        self.detection_count = [0,0,0,0]
        self.map_obj = folium.Map(location = [61.919186, 34.065328], zoom_start = 13)

    def generate_coord(self, detections):
        nl=(random.uniform(61902, 61922))/1000
        el=(random.uniform(34040, 34064))/1000
        intensity = 0
        berry_count = sum(detections.values())
        if berry_count > 10:
            intensity = 1
        else:
            intensity = berry_count/10
        coord=[nl,el,intensity]
        return coord

    def updateHitmap(self, coord):
        # pass
        data = None
        with open('heatmap.pkl', 'rb') as f:
            data = pickle.load(f)
        data.append(coord)
        with open('heatmap.pkl', 'wb') as f:
            pickle.dump(data, f)
        print('APPENDED: ', coord)

    def run_model(self, img):
        results = self.model.predict([img], conf=0.5)
        unique_elements, counts = np.unique(results[0].boxes.cls, return_counts=True)
        detections = dict(zip(unique_elements, counts))
        im_np = results[0].plot(conf=False, labels=False)
        height, width, channels = im_np.shape
        bytes_per_line = channels * width
        qimage = QImage(im_np.data, width, height, bytes_per_line, QImage.Format_BGR888)
        coord = self.generate_coord(detections)
        self.updateHitmap(coord)
        return detections, qimage

    def open(self):
        options = QFileDialog.Options()
        self.welcomeLabel.setText("")
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'QFileDialog.getOpenFileNames()', '',
                                                    'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        if fileNames:
            # Clear existing image widgets and detection counts
            self.scrollLayout.removeWidget(self.scrollContent)
            self.scrollContent.deleteLater()
            self.scrollContent = QWidget()
            self.scrollLayout = QHBoxLayout(self.scrollContent)
            self.scrollLayout.setContentsMargins(0, 0, 0, 0)
            
            self.imageWidgets = []  # Clear existing image widgets
            self.detection_count = [0, 0, 0, 0]  # Reset detection counts

            for fileName in fileNames:
                detections, image_ = self.run_model(fileName)
                self.updateDetectionsLabel(detections)
                
                imageLabel = QLabel()
                imageLabel.setPixmap(QPixmap.fromImage(image_))
                
                imageWidget = QWidget()
                layout = QVBoxLayout()
                layout.addWidget(imageLabel)
                imageWidget.setLayout(layout)
                
                self.imageWidgets.append(imageWidget)
                self.scrollLayout.addWidget(imageWidget)
                
            self.scrollContent.setLayout(self.scrollLayout)
            self.scrollArea.setWidget(self.scrollContent)  # Set the new scroll content
            self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()


    def updateDetectionsLabel(self, detections):
        berries_dict = {0: "Малина", 1: "Черника", 2: "Морошка", 3: "Клубника"}
        detections_text = "Ягоды:\n"
        for class_id, count in detections.items():
            self.detection_count[int(class_id)] += count
        for class_id, count in enumerate(self.detection_count):
            if count == 0:
                continue
            detections_text += f"{berries_dict[class_id]}: {self.detection_count[int(class_id)]}\n"
        self.detectionsLabel.setText(detections_text)

    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>We love berries!</p>")
        
    def openHeatmap(self):
        coords = None
        with open('heatmap.pkl', 'rb') as f:
            coords = pickle.load(f)
        HeatMap(coords).add_to(self.map_obj)
        self.map_obj.save("heat_map.html")
        webbrowser.open('file://' + os.path.realpath('heat_map.html'))

    def createActions(self):
        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open)
        self.heatmapAct = QAction("&Heatmap...", self, shortcut="Ctrl+H", enabled=True, triggered=self.openHeatmap)
        self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=self.about)


    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.heatmapAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)


        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
