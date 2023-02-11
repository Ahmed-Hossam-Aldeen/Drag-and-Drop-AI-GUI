from PyQt5 import QtWidgets, uic
import sys
import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph as pg
from threading import *
from PyQt5.QtCore import Qt
from PyQt5.uic.properties import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.applications import DenseNet121,ResNet50
from keras import layers

from KerasCustomCall import CustomCallback

class MainWindow(QtWidgets.QMainWindow):      
    def __init__(self):   
        super(MainWindow, self).__init__()
        self.setAcceptDrops(True)
        uic.loadUi('GUI.ui', self)
        self.setWindowTitle("Drag & Drop AI")
        
        self.evaluation.setText(' ')
        self.acc_widget.setBackground('w')
        self.loss_widget.setBackground('w')
        
        self.loss_widget.setTitle("Loss vs. epochs")
        self.loss_widget.setLabel('left', "Loss")
        self.loss_widget.setLabel('bottom', "Epochs")  
        self.acc_widget.setTitle("Accuracy vs. epochs")
        self.acc_widget.setLabel('left', "Loss")
        self.acc_widget.setLabel('bottom', "Epochs")
        self.acc_widget.addLegend()
        self.loss_widget.addLegend()
        
       
        self.train_button.clicked.connect(self.TrainThread)
        self.evaluate_button.clicked.connect(self.evaluate)
        self.show()
        
    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            e.accept()
        else:
            e.ignore()
  
    def LoadingData(self,path):
        self.width = int(self.width_txtbox.toPlainText())
        self.height = int(self.height_txtbox.toPlainText())
        batch_size = int(self.batch_size_txtbox.toPlainText())
        
        train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        validation_split=0.20)
        test_datagen = ImageDataGenerator(rescale=1 / 255.0)
        
        
        self.train_generator = train_datagen .flow_from_directory(
            directory=path,
            target_size=(self.width, self.height),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="binary",
            subset='training',
            shuffle=True,
            seed=42
        )
                
        self.valid_generator = train_datagen .flow_from_directory(
            directory=path,
            target_size=(self.width, self.height),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="binary",
            subset='validation',
            shuffle=True,
            seed=42
        )
        # Message Box
        QMessageBox.about(self, "Data Loaded successfully", f'Found {len(self.train_generator)*self.train_generator.batch_size} images belonging to {self.train_generator.num_classes} classes')
    
    def TrainThread(self):      
        t1=Thread(target=self.train)
        t1.start()
        # Message Box
        QMessageBox.about(self, "Data Loaded successfully", f"{self.model_name} Model loaded successfully!")
                  
    def train(self):
        self.model_name = self.models_comboBox.currentText()
        print(self.model_name)
        base_model = eval(self.model_name)(weights='imagenet', include_top=False, input_shape=(self.width,self.height,3))
        
        # Model Building
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(1024,activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        print(self.model.summary())
        
         # Create a CustomCallback instance and pass a reference to the MainWindow instance
        self.callback = CustomCallback(main_window=self)
        self.textBrowser.append('Training has started......')
        self.history = self.model.fit(self.train_generator,
                                      validation_data=self.valid_generator, 
                                      callbacks=[self.callback], 
                                      epochs= self.epochs_spinBox.value())     
        self.evaluate_button.setEnabled(True)
        
    def evaluate(self):
        evaluation = self.model.evaluate(self.valid_generator)
        print(evaluation)
        self.evaluation.setText('Loss: '+ str(evaluation[0])+'\nAcc: '+str(evaluation[1])) 
        
      
    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            self.file_path = m.urls()[0].toLocalFile()
            print(self.file_path)
            self.LoadingData(self.file_path)      
        
app = 0            
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()          
            