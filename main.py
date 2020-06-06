# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'general.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
#Imports normales:
import numpy as np
import os
import sys
import time
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

#Imports de scripts:
sys.path.append('utils/')
from database_generator import database_generator
from segment_hash import get_hash
from inference import predict, predict_rythm

#Imports de interfaz:
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QInputDialog, QFileDialog
import pyqtgraph as pg
from main_ui import *





class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        #PANEL DATABASE:
        #Load database:
        self.pbLoadDB.clicked.connect(self.pbLoadDB_handler)

        #Generate database:
        self.pbCreateDB.clicked.connect(self.pbCreateDB_handler)

        #PANEL SEGMENTOS:
        self.pbSelectFolder.clicked.connect(self.pbSelectFolder_handler)
        self.pbLoadSeg.clicked.connect(self.pbSegments_handler)

        ##Panel gráfica de FT:
        self.pbPlot.clicked.connect(self.pbPlot_handler)
    
        #PREDECIR NOMBRE:
        self.pbPredict.clicked.connect(self.pbPredict_handler)

        #PREDECIR ANORMALIDAD:
        self.pbPredictRithm.clicked.connect(self.pbPredictRythm_handler)
        self._pbPredictRithm_counter = 0    
    
    #Abre el navegador para cargar la base de datos a partir de un archivo:
    def pbLoadDB_handler(self):
        #Para conocer cuando empieza y termina un proceso.
        self.lbStatus.setText("Running...") 
        global dataset
        filepath_database = QFileDialog.getOpenFileName()
        #Carga la base de datos solamente si se selecciona un archivo.
        if filepath_database != ('', ''):
            dataset = np.load(filepath_database[0],allow_pickle=True).item()
            self.lbStatus.setText(".")
        self.lbStatus.setText(".")

            
    

    #Abre el navegador para crear una base de datos a partir de los archivos de una carpeta.
    # #Esta función además imprime el progreso en la ventana de comandos. 
    def pbCreateDB_handler(self):
        self.lbStatus.setText("Running...") 
        #Para lanzar una nueva ventana de dialogo:
        filepath_database = QFileDialog.getExistingDirectory(self, 'Select the folder with the tracks')
        if filepath_database != (''):
            global dataset
            dataset = database_generator(filepath_database,False)
            self.lbStatus.setText(".")
        self.lbStatus.setText(".")
        

    #Sleccionar un folder para cargar los segmentos a predecir.:
    def pbSelectFolder_handler(self):
        self.lbStatus.setText("Running...")
        #Para reiniciar la lista en caso de volver a seleccionar otra carpeta o cancelar la selección: 
        self.cbSegments.clear()
        global filepath_seg, all_extrafiles
        #Para lanzar una nueva ventana de dialogo:
        filepath_seg = QFileDialog.getExistingDirectory(self, 'Select the folder with the tracks to predict')
        if filepath_seg != (''):
            #Extract all files in that path:
            all_files0 = os.listdir(filepath_seg)
            #Para filtrar lso archivos y obtener únicamente los .wav:
            all_files = []
            for names in all_files0:
                if names.endswith(".wav"):
                    all_files.append(names)
            self.lbStatus.setText(".")

            #Actualiza la lista:
            self.cbSegments.addItems(all_files)
        self.lbStatus.setText(".")
            


    #Para cargar la pista seleccionada y graficar su señal:
    def pbSegments_handler(self):
        #self.lbStatus.setText("Running...") 
        #Toma la pista seleccionada y la lee:
        track_selected = self.cbSegments.currentText()
        global FsHz, x1, TimeArray
        FsHz, x1 = wav.read(os.path.join(filepath_seg,track_selected))
        #Gráfica de la señal:
        self.gvTTF.clear()
        TimeArray = np.arange(0.0, np.size(x1)) / FsHz
        self.gvTTF.setYRange(-np.iinfo('int16').max, np.iinfo('int16').max)
        self.gvTTF.plot(TimeArray, x1, linewidth=1,pen='b')
        self.gvTTF.setLabel('left', "Signal")
        self.gvTTF.setLabel('bottom', "Time (sec.)")
        self.gvTTF.showGrid(x=True, y=True)
        #self.lbStatus.setText(".")
        # Al cargar una nueva pista, se borra las predicciones anteriores: 
        self.lbPredictName.setText(" ")
        self.lbPredictRithm.setText(" ")
        self.MplWidget.canvas.axes.clear()
        

    #Para obtener los hashes y graficar la transformada tiempo-frecuencia:
    def pbPlot_handler(self):
        self.lbStatus.setText("Running...")
        global hash_segment
        #Se calculan los hashes y se devuelven variables útiles para la gráfica:
        hash_segment, fft_matrix, FreqTestHz = get_hash(FsHz, x1)
        
        #Se limpia la gráfica
        self.MplWidget.canvas.axes.clear()
        #Se grafica la matriz:
        immat = self.MplWidget.canvas.axes.imshow(fft_matrix, cmap='hot', interpolation='none',
                                origin='lower', aspect='auto',
                                extent=[TimeArray[0], TimeArray[-1],
                                        FreqTestHz[0], FreqTestHz[-1]])
        #Se grafica la colorbar:
        cb = self.MplWidget.canvas.figure.colorbar(immat)
        self.MplWidget.canvas.axes.set_title('Time-frequency transform')
        self.MplWidget.canvas.axes.set_xlabel('Time (sec.)')
        self.MplWidget.canvas.axes.set_ylabel('Freq. (Hz)')
        self.MplWidget.canvas.draw()
        self.lbStatus.setText(".")
        #Se elimina para que no se repita la colorbar, ni sus ejes:
        cb.remove()


 
    #Predice el nombre de la pista y lo muestra:
    def pbPredict_handler(self):
        global prediction_name
        prediction_name = predict(dataset,hash_segment) #Llama a la función de utils
        if prediction_name != "Not found":
            prediction_name = str(prediction_name[:-4]) #Para quitarle la extensión, si la encuentra.
        
        self.lbPredictName.setText(prediction_name)
    #Predice los ritmos cardiacos:
    def pbPredictRythm_handler(self):
        global file_anormal
        #Para que solo solicite el archivo una vez:
        if self._pbPredictRithm_counter == 0: 
            filepath_anormality = QFileDialog.getOpenFileName(self, 'Select the annotation file for abnormalities')
            #Load the database:
            if filepath_anormality != ('', ''):
                file_anormal = open(filepath_anormality[0], "r")
        
        rythm_prediction = predict_rythm(file_anormal,prediction_name)
        self.lbPredictRithm.setText(rythm_prediction)
        self._pbPredictRithm_counter += 1

    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
