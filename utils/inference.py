import numpy as np
import os
from tqdm import tqdm


def predict (dataset, HashValues):
    keys = [*dataset] # Otiene la lista de los nombres de las canciones
    names = np.array([])
    names2 = np.array([])
    pos_aux = np.array([])
    puntajes ={}

    #pdb.set_trace()
    for i in range(len(dataset)):
        cancion = dataset[keys[i]]
        cancion = cancion[:,1:] # Sacar las tres últimas posiciones de los hashes de la canción i-ésima
        hashito = HashValues[:,1:] # Sacar las tres últimas posiciones de los hashes del segmento
        compa = hashito[0,:] # El primer hash de los hashes del segmento
        pos = np.argwhere(np.all((cancion-compa)==0, axis=1)) # Me saca todas las posiciones donde esté compa
        
        if len(pos)!= 0:
            panita = hashito[1,:] #el siguiente hash de los hashes del segmento
            resta_hashito = abs(compa[2]-panita[2]) # Distancia entre hashes consecutivos de los hashes del segmento
            if len(pos) > 1:
                for j in range(len(pos)):
                    hashc = cancion[pos[j]]
                    siguiente_hashc = cancion[pos[j]+1] # el siguiente hash de los hashes de la cancion
                    resta_hashc = abs(hashc[0][2] - siguiente_hashc[0][2])
                    if resta_hashc != resta_hashito:
                        continue
                    else: 
                        pos_aux = np.append(pos_aux, int(pos[j][0]))
                
                if len(pos_aux) == 1: # Acá seguimos haciendo el análisis en una sola canción
                    names = np.append(names,[keys[i]], 0) 
                elif len(pos_aux) > 1:
                    for h in range(len(pos_aux)):
                        parce = hashito[2,:]
                        resta_p = abs(panita[2] - parce[2])
                        hash_c1 = cancion[int(pos_aux[h])]
                        hash_c2 = cancion[int(pos_aux[h])+1]
                        resta_c2 = abs(hash_c1[2] - hash_c2[2])
                        if resta_c2 != resta_p and parce[0] != hash_c1[0] and parce[1] != hash_c1[1] : # Comparando diferencias entre el 2 y 3 hahs de los hashes del segmento y que coincidan las frecuencias de la segunda posición 
                            continue
                        else:
                            names2 = np.append(names2,[keys[i]], 0)                 
            else: # Este else es por si solo existe una posición
                hashc = cancion[pos]
                siguiente_hashc = cancion[pos+1] # el siguiente hash de los hashes de la cancion
                resta_hashc = abs(hashc[0][0][2] - siguiente_hashc[0][0][2])
                if resta_hashc != resta_hashito:
                        continue
                else:
                    if hashc [0][0][0] == compa[0] and hashc [0][0][1] == compa[1]:
                        names = np.append(names,[keys[i]], 0)
        else:       
            continue    



    if len(names) == 1:
        prediccion = names[0]
    elif len(names2) != 0:
        for y in range(len(names2)):
            lista_nombres = names2.tolist()
            nombre_actual = lista_nombres[y]
            contador = lista_nombres.count(nombre_actual)
            puntajes[nombre_actual] = contador
        lista = [*puntajes.values()]

        llaves = [*puntajes]
        if len(llaves) ==1:
            prediccion = llaves[0]
        else: 
            lista = [*puntajes.values()]
            prediccion = lista_nombres[max(lista)]

    elif len(names) > 1:
        prediccion = names[0]   
    
    elif len(names) == 0:
        prediccion = "Not found"

    return prediccion


def predict_rythm(File_Abnormal,name_predicted):   
    File_Abnormal = File_Abnormal.read()    
    File_Abnormal = File_Abnormal.split()

    if name_predicted in File_Abnormal:
        return "Abnormal rythm"
    else:
        return "Normal rythm"





        
        
        
        
        
        






        
        
        
        
        
        