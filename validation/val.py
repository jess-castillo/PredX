## Prueba de validaciÃ³n:
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

#####################Canciones#######################
ann_canciones  =  np.load('an_pistas.npy')
pred_canciones = np.load('pred_pistas.npy')

ann_filtradas = []
for i in range(len(ann_canciones)):
    ann_filtradas.append(ann_canciones[i][1:])
    
#cm1 = confusion_matrix(ann_filtradas,pred_canciones)

ptp = 0
pfn = 0
pfp = 0
for i in range(len(ann_filtradas)):
    if ann_filtradas[i] == pred_canciones[i]:
        ptp +=1
    elif pred_canciones[i] == 'Not found':
        pfn +=1
    else:
        pfp +=1
#cm1 = np.array([[ptp,pfn],[pfp,ptp]])
precision_can = ptp/(ptp+pfp)
recall_can = ptp/(ptp+pfn)
print("Para canciones:")
print('Precisión: ' + str(precision_can))
print('Cobertura: ' + str(recall_can))
###############Cardiacos############################

ann_cardiacos  =  np.load('an_cardio.npy')
pred_cardiacos = np.load('pred_cardio.npy')

ann_filtradas2 = []
for i in range(len(ann_cardiacos)):
    ann_filtradas2.append(ann_cardiacos[i][1:])
    
#cm2 = confusion_matrix(ann_filtradas2,pred_cardiacos)

ctp = 0
cfn = 0
cfp = 0
for i in range(len(ann_filtradas2)):
    if ann_filtradas2[i] == pred_cardiacos[i]:
        ctp +=1
    elif pred_cardiacos[i] == 'Not found':
        cfn +=1
    else:
        cfp +=1
#cm2 = np.array([[ctp,cfn],[cfp,ctp]])
precision_car = ctp/(ctp+cfp)
recall_car = ctp/(ctp+cfn)
print("Para cardiácos:")
print('Precisión: ' + str(precision_car))
print('Cobertura: ' + str(recall_car))
###################################################
