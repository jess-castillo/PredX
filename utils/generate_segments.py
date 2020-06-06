    #clc; close all; clear;

import numpy as np
import scipy.io.wavfile as wav
import os
import pdb
from random import randrange
from math import ceil

# Cargamos el archivo
DataPath = "./PistasCardio"
all_files0 = os.listdir(DataPath)
folder ='Segmentos_Cardio'

#Creamos los segmentos de la base de datos
for i in range(np.size(all_files0)):  
    FullFileName = "{:s}/{:s}".format(DataPath, all_files0[i])    
    FsHz, audio = wav.read(FullFileName)
    s_Time = 4 #Tiempo a evaluar. 
    s_NoSegm = 10
    s_Samples = s_Time*FsHz              
    for j in range(s_NoSegm):
        start = ceil(randrange(1)*np.size(audio,0))
        str_FileName = str(j) + all_files0[i]
        v_Segm = audio[start:start+s_Samples] 
        str_SaveFileName = os.path.join(folder,str_FileName)
        wav.write(str_SaveFileName,FsHz, v_Segm)

print ('Done') 

