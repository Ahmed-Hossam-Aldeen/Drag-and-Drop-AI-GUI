from tensorflow import keras
import pyqtgraph as pg

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.epochs = []
        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
        
        
    def on_epoch_end(self, epoch, logs=None):     
        self.main_window.textBrowser.append(f'Loss for epoch {epoch+1}: {logs["loss"]}')
        self.main_window.textBrowser.append(f'Val Loss for epoch {epoch+1}: {logs["val_loss"]}')
        self.main_window.textBrowser.append(f'Acc for epoch {epoch+1}: {logs["accuracy"]}')
        self.main_window.textBrowser.append(f'Val Acc for epoch {epoch+1}: {logs["val_accuracy"]}')
        self.main_window.textBrowser.append('----------------------------------------')
        
        red_pen = pg.mkPen(width =2, color=(255, 0, 0))
        blue_pen = pg.mkPen(width =2, color=(0, 0, 255))
        
        self.epochs.append(epoch+1)
        self.loss.append(logs["loss"])
        self.val_loss.append(logs["val_loss"])
        self.accuracy.append(logs["accuracy"])
        self.val_accuracy.append(logs["val_accuracy"])
 
        
        self.main_window.loss_widget.plot(self.epochs, self.loss, name = "loss", pen=blue_pen ,clear=True)
        self.main_window.loss_widget.plot(self.epochs, self.val_loss, name = "val_loss", pen=red_pen)
        
        self.main_window.acc_widget.plot(self.epochs, self.accuracy, name = "accuracy", pen=blue_pen, clear=True)
        self.main_window.acc_widget.plot(self.epochs, self.val_accuracy, name = "val_accuracy", pen=red_pen)