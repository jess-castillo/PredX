# audio_tf_taller.py
# Author: Mario Valderrama
# Date: 2020
# Creación de la base de datos
# a partir de la transformada Tiempo-Frecuencia

# Importamos las librerías
import numpy as np
import scipy.io.wavfile as wav
import os
import pdb
from tqdm import tqdm

#Se define el generador de la base de datos. Argumentos: path al folder, y si se salva la base de datos o solo se carga
def database_generator(path,save):
    # Cargamos el archivo
    DataPath = path
    all_files0 = os.listdir(DataPath)

    #Filtrar solo las .wav
    all_files = []
    for names in all_files0:
        if names.endswith(".wav"):
            all_files.append(names)

    ##
    # Definimos una función para encontrar el
    # máximo dentro de cada parche de puntos conectados
    def GetConnectedPatchMat(MatIn, iIn, jIn):
        MatValues = np.array([])
        MatiValues = np.array([])
        MatjValues = np.array([])
        MatijCheck = np.array([[iIn, jIn]])
        while 1:
            if np.size(MatijCheck) == 0:
                break
            iInAux = np.int(1 * MatijCheck[0][0])
            jInAux = np.int(1 * MatijCheck[0][1])
            MatijCheck = np.delete(MatijCheck, 0, 0)
            i1 = iInAux - 1
            if i1 < 0:
                i1 = 0
            i2 = iInAux + 1
            if i2 >= np.size(MatIn, 0):
                i2 = np.size(MatIn, 0) - 1
            j1 = jInAux - 1
            if j1 < 0:
                j1 = 0
            j2 = jInAux + 1
            if j2 >= np.size(MatIn, 1):
                j2 = np.size(MatIn, 1) - 1
            for iCount in range(i1, i2 + 1):
                for jCount in range(j1, j2 + 1):
                    if MatIn[iCount, jCount] == 0.0:
                        continue
                    ValAux = 1.0 * MatIn[iCount, jCount]
                    MatValues = np.append(MatValues, [ValAux])
                    MatiValues = np.append(MatiValues, [iCount])
                    MatjValues = np.append(MatjValues, [jCount])
                    MatIn[iCount, jCount] = 0.0
                    if iCount == iIn and jCount == jIn:
                        continue
                    MatijCheck = np.append(MatijCheck, [[iCount, jCount]], 0)
                    
        return MatValues, MatiValues, MatjValues

    #Se crea el diccionario que almacena la base de datos:
    dataset = {}
    for file in tqdm(range(len(all_files)),desc='Pistas:'):    
        FileName = all_files[file]
        FullFileName = "{:s}/{:s}".format(DataPath, FileName)
        FsHz, x1 = wav.read(FullFileName)
        
        # Submuestreamos el archivo de audio a una
        # menor frecuencia para agilizar el procedimiento
        # Para una frecuencia de muestreo de 44100
        if FsHz == 44100:
            FsHzNew = FsHz / 21.0
            x1New = np.array([])
            F1 = 0
            while 1:
                F2 = F1 + 21
                if F2 > np.size(x1):
                    break
                x1New = np.append(x1New, np.mean(x1[F1:F2]))
                F1 = F2
            x1 = x1New
            FsHz = FsHzNew
        
        # Creamos un vector de tiempo cuyo número
        # de elementos corresponda a la duración
        # total y al periodo de muestreo.
        
        TimeArray = np.arange(0.0, np.size(x1)) / FsHz
        
        # Graficamos la señal de entrada
        """
        plt.figure()
        plt.plot(TimeArray, x1, linewidth=1)
        plt.xlabel("Time (sec)", fontsize=15)
        plt.ylabel("x1", fontsize=15)
        plt.grid(1)
        """
        # Definimos un rango de frecuencias
        # las cuales usaremos para crear nuestros
        # patrones oscilatorios de prueba
        FreqIniHz = 30.0
        FreqEndHz = 800.0
        FreqStepHz = 5.0
        FreqTestHz = np.arange(FreqIniHz, FreqEndHz + FreqStepHz, FreqStepHz)
        
        # Creamos una matriz que usaremos para
        # almacenar el resultado de las
        # convoluciones sucesivas. En esta matriz,
        # cada fila corresponde al resultado de
        # una convolución y cada columna a todos
        # los desplazamientos de tiempo.
        TFMat = np.zeros([np.size(FreqTestHz), np.size(x1)], dtype=complex)
        
        # Definimos el número de ciclos
        # por cada patrón de frecuencia
        NumCycles = 10
        
        # Creamos un arreglo de tiempo que esté
        # centrado en cero para que la ventana
        # gaussiana que usaremos para la generación
        # de patrones esté centrada en cero
        # TimeArrayGauss = TimeArray - (TimeArray[-1] / 2.0)
        
        # Se obtiene la transformada de Fourier
        # de la señal x1 para usarla en cada iteración
        x1fft = np.fft.fft(x1)
        
        # Ahora creamos un procedimiento iterativo
        # que recorra todas las frecuencias de prueba
        # definidas en el arreglo FreqTestHz
        for FreqIter in range(np.size(FreqTestHz)):
            # Generamos una señal de prueba
            # que oscile a la frecuencia de la iteración
            # FreqIter (FreqTestHz[FreqIter]) y que tenga
            # la misma longitud que la señal x1.
            # En este caso usamos una exponencial compleja.
            xtest = np.exp(1j * 2.0 * np.pi * FreqTestHz[FreqIter] * TimeArray)
        
            # Creamos una ventana gaussina para
            # limitar nuestro patrón en el tiempo
            # Definimos la desviación estándar de
            # acuerdo al número de ciclos definidos
            # Dividimos entre 2 porque para un ventana
            # gaussiana, una desviación estándar
            # corresponde a la mitad del ancho de la ventana
            xtestwinstd = ((1.0 / FreqTestHz[FreqIter]) * NumCycles) / 2.0
            # Definimos nuestra ventana gaussiana
            # En este caso, dado que vamos a implementar las
            # convoluciones usando la transformada de Fourier
            # y que vamos a anular la fase de la transformada
            # de la ventana gaussiana, no es  importante que el
            # vector de tiempo esté centrado en cero.
            # Esto sería importante si usáramos el procedimiento
            # de las convolucines en el dominio del tiempo!
            xtestwin = np.exp(-0.5 * (TimeArray / xtestwinstd) ** 2.0)
            # Multiplicamos la señal patrón por
            # la ventana gaussiana
            xtest = xtest * xtestwin
        
            # Se obtine la transformada de Fourier del patrón
            fftxtest = np.fft.fft(xtest)
            # Se toma únicamente la parte real para evitar
            # corrimientos de fase
            fftxtest = abs(fftxtest)
            # Se obtine el resultado de la convolución realizando
            # la multiplicación de las transformadas de Fourier de
            # la señal x1 por la del patrón
            TFMat[FreqIter, :] = np.fft.ifft(x1fft * fftxtest)
        
        
        # Graficamos la señal x1 con su
        # correspondiente matriz de evaluación de
        # patrones
        """
        axhdl = plt.subplots(2, 1, sharex=True, constrained_layout=True)
        
        # Graficamos la señal x1
        axhdl[1][0].plot(TimeArray, x1, linewidth=1)
        axhdl[1][0].set_ylabel("x1", fontsize=15)
        axhdl[1][0].grid(1)
        """
        # Graficamos la matriz resultante en escala de colores
        TFMatAbs = np.abs(TFMat)
        """
        immat = axhdl[1][1].imshow(TFMatAbs, cmap='hot', interpolation='none',
                                origin='lower', aspect='auto',
                                extent=[TimeArray[0], TimeArray[-1],
                                        FreqTestHz[0], FreqTestHz[-1]])
        axhdl[1][1].set_xlabel("Time (sec)", fontsize=15)
        axhdl[1][1].set_ylabel("Freq (Hz)", fontsize=15)
        axhdl[1][1].set_xlim([TimeArray[0], TimeArray[-1]])
        axhdl[0].colorbar(immat, ax=axhdl[1][1])
        """
        # Buscamos máximos en el tiempo-frecuencia
        # al interior de ventanas de tamaño del doble
        # de TimeLenSec y del doble de FreqLenHz
        # Los valores de TimeLenSec y FreqLenHz se pueden
        # ajustar a otros valores si se requiere
        TFMatAbsMax = np.zeros([np.size(TFMatAbs, 0), np.size(TFMatAbs, 1)])
        FreqLenHz = 20
        FreqLenPoints = np.int(np.round(FreqLenHz / FreqStepHz))
        TimeLenSec = 0.25
        TimeLenSam = np.int(np.round(TimeLenSec * FsHz))
        for i in range(np.size(TFMatAbsMax, 0)):
            for j in range(np.size(TFMatAbsMax, 1)):
                iF1 = i - FreqLenPoints
                iF2 = i + FreqLenPoints
                jF1 = j - TimeLenSam
                jF2 = j + TimeLenSam
                if iF1 < 0 or iF2 > np.size(TFMatAbsMax, 0) or \
                        jF1 < 0 or jF2 > np.size(TFMatAbsMax, 1):
                    continue
                MatAux = TFMatAbs[iF1:iF2, jF1:jF2]
                maxin = np.argmax(MatAux)
                maxin_i = maxin // np.size(MatAux, 1)
                maxin_j = maxin - (maxin_i  * np.size(MatAux, 1))
                TFMatAbsMax[iF1 + maxin_i, jF1 + maxin_j] = \
                    TFMatAbs[iF1 + maxin_i, jF1 + maxin_j]
                # TFMatAbsMax[iF1 + maxin_i, jF1 + maxin_j] = 1.0
        
        # TFMatAbsMaxAux = 1.0 * TFMatAbsMax
        
        ##
        # Graficamos la matrix con los máxmimos encontrados
        """
        plt.figure()
        plt.imshow(TFMatAbsMax, cmap='hot', interpolation='none',
                origin='lower', aspect='auto',
                extent=[TimeArray[0], TimeArray[-1],
                        FreqTestHz[0], FreqTestHz[-1]])
        plt.xlabel("Time (sec)", fontsize=15)
        plt.ylabel("Freq (Hz)", fontsize=15)
        """
        
        
        ##
        # Obtenemos el máximo dentro de cada parche
        # de punto conectados
        TFMatAbsMaxBin = np.zeros([np.size(TFMatAbsMax, 0), np.size(TFMatAbsMax, 1)])
        for i in range(np.size(TFMatAbsMax, 0)):
            for j in range(np.size(TFMatAbsMax, 1)):
                if TFMatAbsMax[i,j] == 0:
                    continue
        
                MatValues, MatiValues, MatjValues = \
                    GetConnectedPatchMat(TFMatAbsMax, i, j)
                # for c1 in range(np.size(MatiValues)):
                #     TFMatAbsMaxBin[np.int(MatiValues[c1]), np.int(MatjValues[c1])] = 0.5
        
                maxin = np.argmax(MatValues)
                maxin_i = np.int(MatiValues[maxin])
                maxin_j = np.int(MatjValues[maxin])
                TFMatAbsMaxBin[maxin_i, maxin_j] = 1.0
        
        ##
        # Graficamos la matrix con los máxmimos
        # encontrados dentro de cada parche de puntos
        # conectados
        """
        plt.figure()
        plt.imshow(TFMatAbsMaxBin, cmap='hot', interpolation='none',
                origin='lower', aspect='auto',
                extent=[TimeArray[0], TimeArray[-1],
                        FreqTestHz[0], FreqTestHz[-1]])
        """
        ##
        # Creamos la lista de valores "hash"
        # Definimos una ventana "target zone" 
        # de tamaño el doble de TimeLenSec y
        # el doble de FreqLenHz y que se encuentre
        # corrida una distancia de TimeForwardSec segundos
        # Los valores de TimeLenSec, FreqLenHz y
        # TimeForwardSec se pueden ajustar a otros
        # valores si se requiere
        HashValues = np.array([])
        TFMatAbsMax = np.zeros([np.size(TFMatAbs, 0), np.size(TFMatAbs, 1)])
        FreqLenHz = 50
        FreqLenPoints = np.int(np.round(FreqLenHz / FreqStepHz))
        TimeLenSec = 1.0
        TimeLenSam = np.int(np.round(TimeLenSec * FsHz))
        TimeForwardSec = 0.25
        TimeForwardSam = np.int(np.round(TimeForwardSec * FsHz))
        for i in range(np.size(TFMatAbsMaxBin, 0)):
            for j in range(np.size(TFMatAbsMaxBin, 1)):
                if TFMatAbsMaxBin[i][j] == 0.0:
                    continue
                HashFreqHz = np.int(np.round(FreqTestHz[i]))
                HashTimeMilliSec = np.int(np.round(TimeArray[j] * 1000))
                iF1 = i - FreqLenPoints
                iF2 = i + FreqLenPoints
                jF1 = j + TimeForwardSam
                jF2 = jF1 + TimeLenSam
                if iF1 < 0:
                    iF1 = 0
                if iF2 >= np.size(TFMatAbsMaxBin, 0):
                    iF2 = np.size(TFMatAbsMaxBin, 0) - 1
                if jF1 > np.size(TFMatAbsMaxBin, 1):
                    continue
                if jF2 >= np.size(TFMatAbsMaxBin, 1):
                    jF2 = np.size(TFMatAbsMaxBin, 1) - 1
        
                for i1 in range(iF1, iF2 + 1):
                    for j1 in range(jF1, jF2 + 1):
                        if TFMatAbsMaxBin[i1][j1] == 0.0:
                            continue
                        HashFreqHzAux = np.int(np.round(FreqTestHz[i1]))
                        HashTimeMilliSecAux = np.int(np.round(TimeArray[j1] * 1000))
                        HashTimeDiffMilliSecAux = \
                            np.int(HashTimeMilliSecAux - HashTimeMilliSec)
                        HashLine = np.array([HashTimeMilliSec, HashFreqHz,
                                            HashFreqHzAux, HashTimeDiffMilliSecAux])
                        if np.size(HashValues) == 0:
                            HashValues = np.array([HashLine])
                        else:
                            HashValues = np.append(HashValues, [HashLine], 0)
                                    
        dataset[FileName] = HashValues
    if save == True:
        np.save('dataset.npy', dataset)
    return dataset

