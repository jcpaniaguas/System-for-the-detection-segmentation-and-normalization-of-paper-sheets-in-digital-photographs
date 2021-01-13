# @author: jcpaniaguas
import cv2
import math
import numpy as np
import os
import pandas as pd
#import csv

class Analyzer:
    """Clase para analizar la efectividad de una base de datos de entrenamientos
    sobre cada una de las fotos.
    """

    def __init__(self, archivo, primera_col):
        """Constructor de la clase Analyzer.

        Args:
            archivo ([char]): Nombre del path del archivo csv que se va a utilizar como Analyzer,
            para escribir nuevos datos.
            primera_col ([[char]]): Lista de directorio que van a servir como primera columna para
            el csv.
        """
        self.archivo = archivo
        if os.path.isfile(archivo):
            print("Ya existe un archivo con ese nombre. Se ha escogido ese archivo.") 
            return
        contenido = []
        for d in primera_col:
            contenido += os.listdir(d)
        contenido += ['train','test','total']
        col_img = {'img':contenido}
        df = pd.DataFrame(col_img,columns=['img'])
        df.to_csv(archivo,index=False) 


    def escribir_columna(self,nombre_col,contenido):
        """Se añade una nueva columna nueva en el csv.

        Args:
            nombre_col ([char]): Nombre que tendrá la columan en el csv.
            contenido ([[char]]]): Lista de valores de la columna.
        """
        df = pd.read_csv(self.archivo)
        df.insert(1,nombre_col,pd.DataFrame(contenido),True)
        df.to_csv(self.archivo,index=False) 

    def concatenar_a_columna(self,origen,contenido):
        """Concatenar en una columna ya existen nuevos valores.

        Args:
            origen ([char]): Nombre de la columna donde se va a añadir el contenido.
            contenido ([[char]]): Lista de valores de la columna.  
        """
        df = pd.read_csv(self.archivo)
        index = 0  
        for idx,d in enumerate(df[origen]):
            if type(d)==float:
                index = idx
                break
        for idx,c in enumerate(contenido):
            df[origen][idx+index] = c
        df.to_csv(self.archivo,index=False)

    def insertar(self,foto_referencia,origen,contenido):
        """Se inserta el 'contenido' en la columna 'origen' con el indice
         de la 'foto_referencia'  

        Args:
            foto_referencia ([char]): Foto en la fila donde se va a insertar el contenido.
            origen ([char]): Nombre de la columna donde se va a añadir el contenido.
            contenido ([char]): Valor de la columna. 
        """
        df = pd.read_csv(self.archivo)
        index = 0  
        for idx,d in enumerate(df['img']):
            if d==foto_referencia:
                index = idx
                break
        if origen not in df.columns:
            df.insert(1,origen,'',True)
        df[origen][index] = contenido
        df.to_csv(self.archivo,index=False)
    
    def porcentaje_acierto(self,column):
        df = pd.read_csv(self.archivo)
        actual = df[column]

        def is_complete(actual):
            last_cell = 99
            while last_cell >= 0:
                if actual[last_cell] in ['S','s','N','n']:
                    last_cell -= 1
                else:
                    return False
            return True
        
        def sumar_aciertos(actual):
            train = actual[:75].tolist().count('S')
            test = actual[75:].tolist().count('S')
            total = actual.tolist().count('S')
            #return round(train/75, 2),round(test/25, 2),round(total/100, 2)
            return train,test,total

        if is_complete(actual):
            train, test, total = sumar_aciertos(df[column])
            df[column][100] = train
            df[column][101] = test
            df[column][102] = total
        df.to_csv(self.archivo,index=False)
                

"""
a = Analyzer('Analyzer.csv',['./img/training/','./img/testing/'])
a.escribir_columna('primero',['S','S','N'])
a.concatenar_a_columna('primero',['N','N','S','S','S'])
a.insertar('IMG_7228.JPG','base de datos','N')
"""
a = Analyzer('Analyzer.csv',['./img/training/','./img/testing/'])
a.porcentaje_acierto('bd_training_1_foto_v2.pkl')