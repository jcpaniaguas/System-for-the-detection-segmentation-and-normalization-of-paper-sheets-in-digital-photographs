# @author: jcpaniaguas
from Trainer import Trainer
from Tester import Tester
import os

def encontrar_folios(directorio_fotos,numero_de_fotos_training,groundtruth,nombre_bd,porcentaje=False):
    """Funcion que busca las esquinas de los folios de testing con una base de datos de entrenamiento dada.
    Si la base de datos no ha sido creada se entrenara y utilizara.

    Args:
        directorio_fotos ([str]): Directorio donde se encuentran las fotos de los folios a localizar.
        numero_de_fotos_training ([int]): Del total de fotos del directorio_fotos, el número de fotos que se van a utilizar
        para training.
        groundtruth ([str]): Archivo csv con las esquinas de los folios de training.
        nombre_bd ([str]): Nombre de la base de datos de entrenamiento. Si no existe el archivo se realizara el 
        entrenamiento. Si existe se utilizará esa base de datos.
        porcentaje (bool, optional): Si porcentaje es False el parámetro numero_de_fotos_training debera ser
        un numero entero. Si porcentaje es True el parámetro numero_de_fotos_training debera ser un porcentaje (0-100),
        el porcentaje de archivos del directorio de fotos que se va a utilizar para training. Por defecto a False.
    """
    archivos = os.listdir(directorio_fotos)
    rango = 0
    print("Nombre bd: "+nombre_bd)

    if not porcentaje:    
        if numero_de_fotos_training <= len(archivos):
            tr = str(numero_de_fotos_training)
            ts = str(len(archivos)-numero_de_fotos_training) 
            print("training: "+tr+", testing: "+ts)
            rango = numero_de_fotos_training
        else:
            print("Error: 'numero_de_fotos_training' debe ser menor al número de fotos del directorio seleccionado 'directorio_fotos'.")
            return 0
    else:
        if numero_de_fotos_training in range(101):
            tr = round(len(archivos)*(numero_de_fotos_training/100))
            ts = len(archivos)-tr
            rango = round(len(archivos)*(numero_de_fotos_training/100))
            print("training: "+str(tr)+", testing: "+str(ts))
        else:
            print("Error: 'numero_de_fotos_training' debe ser un porcentaje comprendido entre [0-100].")
        
    trainer = Trainer(directorio_fotos,groundtruth,rango)
    trainer.entrenar(nombre_bd)
    #Guardar = True -> guarda las imagenes; = False -> se muestran en pantalla
    tester = Tester("./bd/"+nombre_bd,1)
    dic_result = tester.localizar(directorio_fotos)
    return dic_result

DIR_FOTOS = "./img/redim/"
fotos = len(os.listdir(DIR_FOTOS))
groundtruth = "./img/groundtruth_redim_tipo_0.csv"

for i in [28]:
    encontrar_folios(directorio_fotos=DIR_FOTOS,
                    groundtruth=groundtruth,
                    numero_de_fotos_training=i,
                    nombre_bd="bd_train_"+str(i)+"_test_"+str(fotos-i)+"_closing_tr_pequenos.pkl")