# PredX: Una aproximación a la detección de ritmos cardiacos anormales

### Luisa Andrade, Jessica Castillo, Juan David García. 

Ingeniería biomédica.

Universidad de los Andes.

## Descripción

Esta aplicación permite obtener la predicción de segmentos de audios, los cuales hacen parte de una base de datos  pre-existente, o creada en el momento; para así obtener el nombre de la pista a la cual pertenece el segmento. Además, si los segmentos de ritmos cardiacos, es posible conocer si son ritmos anormales. Finalmente, la aplicación permite visualizar la señal de audio, junto con su transformada de tiempo-frecuencia.

## Requerimientos de software

Para hacer uso de la aplicación, es necesario tener instaladas las siguientes librerías de Python:

**Para Windows:** Desde alguna consola como Anaconda Prompt:

+ **PyQt5:** `pip install PyQt5`
+ **Numpy:** `pip install numpy`
+ **Scipy:** `pip install scipy`
+ **tqdm:** `pip install tqdm`
+ **PyQt Graph:** `pip install pyqtgraph`

**Para Linux:** Desde la terminal:

+ **PyQt5:** `sudo apt-get install python3-pyqt5`
+ **Numpy:** `sudo apt-get install python3-numpy`
+ **Scipy:** `sudo apt-get install python3-scipy`
+ **PyQt Graph:** `sudo apt-get install python3-pyqtgraph`

## Uso

La aplicación se corre desde el script principal denominado **main**: `python main.py` 

En primer lugar, es necesario cargar la base de datos (**Load database**) guardada previamente como un diccionario en un archivo de numpy; o bien, generar la base de datos (**Create database**), para lo cual, se requerirá ingresar la ruta en la que se encuentra la carpeta con las pistas. Posteriormente, se deberá seleccionar la carpeta con los segmentos a predecir (**Select folder**), en formato `.wav` , para así obtener la lista desplegable de estos. Al seleccionar alguno, se debe cargar (**Load track**) para así obtener una visualización de la señal.  

Para visualizar la transformada tiempo frecuencia, es necesario oprimir el botón **Plot**, y finalmente, para predecir cuál es el nombre de la pista, se debe oprimir **Predict name**. En adición, si el segmento y la base de datos cargados hacen parte de ritmos cardiacos, se puede predecir si el segmento es anormal. Para ello, se presiona el botón **Predict**, bajo el módulo **Cardiac rythm**. La aplicación requiere que se ingrese el archivo de anotaciones de ritmos cardiacos *anormales* para realizar la predicción correctamente. 

