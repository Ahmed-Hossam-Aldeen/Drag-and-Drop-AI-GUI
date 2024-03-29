from PyQt5 import QtWidgets, uic
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import pyqtgraph as pg
from threading import *
from PyQt5.QtCore import Qt, QUrl
from PyQt5.uic.properties import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121,ResNet50
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from tensorboard import program

from KerasCustomCall import CustomCallback
import metrics


class MainWindow(QtWidgets.QMainWindow):      
    def __init__(self):   
        super(MainWindow, self).__init__()
        self.setAcceptDrops(True)
        uic.loadUi('GUI.ui', self)
        self.setWindowTitle("Drag & Drop AI")
        
        self.widget.load(QUrl("http://localhost:6006/"))
        self.evaluation.setText(' ')
        self.acc_widget.setBackground('k')
        self.loss_widget.setBackground('k')
        
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
        self.tb_start_button.clicked.connect(self.tbThread)
        self.tb_end_button.clicked.connect(self.tbEnd)          
        self.show()



    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            self.file_path = m.urls()[0].toLocalFile()
            print(self.file_path)
            self.LoadingData(self.file_path)  


    def tbThread(self):      
        self.t2=Thread(target=self.tb)
        self.t2.start()
    def tb(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', 'logs'])
        url = tb.launch()
        print(url)
    def tbEnd(self):
        self.t2.join()

    def LoadingData(self,path):
        # self.width = int(self.width_txtbox.toPlainText())
        # self.height = int(self.height_txtbox.toPlainText())
        # batch_size = int(self.batch_size_txtbox.toPlainText())
        
        self.width = 100
        self.height = 100
        batch_size = 10
        
        data_generator_params = {
            'rescale': 1 / 255.0,
            'validation_split': 0.20  # You can set this value to None if the user doesn't want a validation split
        }
        train_datagen = ImageDataGenerator(**data_generator_params)

        test_datagen = ImageDataGenerator(rescale=1 / 255.0)
        

        if self.RGB.isChecked():
            color_mode = 'rgb'
            self.depth = 3
        if self.Grayscale.isChecked():
            color_mode = 'grayscale'
            self.depth = 1

        self.train_generator = train_datagen .flow_from_directory(
            directory=path,
            target_size=(self.width, self.height),
            color_mode=color_mode,
            batch_size=batch_size,
            class_mode="binary",
            subset='training',
            shuffle=True,
            seed=42
        )
                
        self.valid_generator = train_datagen .flow_from_directory(
            directory=path,
            target_size=(self.width, self.height),
            color_mode=color_mode,
            batch_size=batch_size,
            class_mode="binary",
            subset='validation',
            shuffle=True,
            seed=42
        )
        # Message Box
        QMessageBox.about(self, "Data Loaded successfully", f'Found {len(self.train_generator)*self.train_generator.batch_size} images belonging to {self.train_generator.num_classes} classes')
        QMessageBox.about(self, "Data Loaded successfully", f'Found {len(self.valid_generator)*self.valid_generator.batch_size} images belonging to {self.valid_generator.num_classes} classes')
    
    def TrainThread(self):      
        t1=Thread(target=self.train)
        t1.start()
        # Message Box
        QMessageBox.about(self, "Data Loaded successfully", f"{self.model_name} Model loaded successfully!")
                  
    def train(self):
        self.evaluate_button.setEnabled(False)
        self.model_name = self.models_comboBox.currentText()
        print(self.model_name)
        base_model = eval(self.model_name)(weights='imagenet', include_top=False, input_shape=(self.width,self.height,self.depth))
        
        if self.binary.isChecked():
            self.loss = 'binary_crossentropy'
        elif self.categorical.isChecked():
            self.loss = 'categorical_crossentropy'

        self.metrics = []
        if self.acc_metric.isChecked():
            self.metrics.append('accuracy')
        if self.mse_metric.isChecked():
            self.metrics.append('mse')
        if self.mae_metric.isChecked():
            self.metrics.append('mae')

        if self.precision_metric.isChecked():
            self.metrics.append(metrics.precision_m)    
        if self.recall_metric.isChecked():
            self.metrics.append(metrics.recall_m)        
        if self.f1_metric.isChecked():
            self.metrics.append(metrics.f1_m)    

        print(self.metrics)        

        # Model Building
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(layers.GlobalAveragePooling2D())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(1024,activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        
        self.model.compile(loss=self.loss, optimizer=Adam(lr=0.001), metrics=self.metrics)
        print(self.model.summary())
        
         # Create a CustomCallback instance and pass a reference to the MainWindow instance
        self.callback = CustomCallback(main_window=self)

        
        tensorboard = TensorBoard(log_dir="logs/{}".format(self.project_name_txtbox.toPlainText()))
        self.textBrowser.append('Training has started......')
        self.history = self.model.fit(self.train_generator,
                                      validation_data=self.valid_generator, 
                                      callbacks=[self.callback, tensorboard], 
                                      epochs= self.epochs_spinBox.value())  
           
        self.evaluate_button.setEnabled(True)
        
    def evaluate(self):
        evaluation = self.model.evaluate(self.valid_generator)
        print(evaluation)
        self.evaluation.setText('Loss: '+ str(evaluation[0])+'\nAcc: '+str(evaluation[1])) 
        
      
  
        
app = 0            
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()          
            