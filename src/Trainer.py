# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt
import cv2
import csv
import pickle
import os
import numpy as np

class Trainer:
    """Clase que necesita de un directorio de imágenes de entrenamiento y archivo groundtruth 
    con la esquinas reales de cada una de las imágenes.
    El método principal de la clase es entrenar() que iniciará el entrenamiento y creará una base
    de datos con los keypoints y descriptores válidos de cada imagen. 
    """
    
    IMG_H_VIEW = 960
    IMG_W_VIEW = 540
    KP_RANGE = 15
    COTA_MAX_STDR = 200
    COTA_MIN_STDR = 130
    MAX_IMAGEN = 255
    MIN_IMAGEN = 0
    ORB_HARRIS_SCORE = 1.3
    ORB_FAST_SCORE = 4

    def __init__(self, directorio, csv_puntos):
        """Constructor de la clase Trainer

        Args:
            directorio ([str]): Directorio de imagenes
            csv_puntos ([str]): Archivo .csv con las esquinas de cada imagen
        """
        self.dir = directorio
        self.csv_puntos = csv_puntos

    def __redimensionar_puntos(self, archivo, puntos):
        """Se redimesiona, en función de como se va a redimensionar la imagen, los puntos de esta
        para que no se descuadren. La redimensión de esta depende de los valores de IMG_H_VIEW y 
        IMG_W_VIEW

        Args:
            archivo ([str]): Nombre de la imagen
            puntos ([ [(int,int)] ]): Puntos originales de la imagen sin redimensionar

        Returns:
            [ [(int,int)] ]: Puntos después de la redimensión
        """
        img = cv2.imread(self.dir + archivo, 0)
        width, height = img.shape
        redimensionados = []
        for (x, y) in puntos:
            new_x = round((self.IMG_H_VIEW*x)/height)
            new_y = round((self.IMG_W_VIEW*y)/width)
            redimensionados.append((new_x, new_y))
        return redimensionados

    def __cvs_to_dict(self):
        """Con un csv como entrada devuelve un diccionario con sus valores tal que la primera columna será la key
        y las demás los valores

        Returns:
            [dict([(int,int)])]: Diccionario cuya key es el el nombre de la imagen y cuyos values son los puntos
            que se consideran esquinas 
        """
        csv_dict = dict()
        with open(self.csv_puntos) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                actual_id = row['\ufeff"id"']
                actual_list = list()
                idx = 0
                actual_x = "x" + str(idx)
                var = int((len(row)-1)/2)
                while (idx < var) and (row[actual_x] != ''):
                    actual_y = "y" + str(idx)
                    actual_list.append(
                        (int(row[actual_x]), int(row[actual_y])))
                    idx += 1
                    actual_x = "x" + str(idx)
                actual_list = self.__redimensionar_puntos(
                    actual_id, actual_list)
                csv_dict[actual_id] = actual_list

        return csv_dict

    def __lista_to_csv(self, lista):
        """Sirve como funcion de apoyo para mejores_parametros y crea un archivo .csv 
        con los datos de lista. 

        Args:
            lista ([[int]]): incluye una matriz con los mejores parametros
        """
        with open("./img/mejores_parametros.csv", "wt") as f:
            writer = csv.writer(f)
            writer.writerows(lista)

    def __guardar_entrenamiento(self, nombre, bd, truncar):
        """Se crea la base de datos y se guarda el resultado del entrenamiento

        Args:
            nombre ([str]): Nombre del futuro archivo
            bd ([dict()]): Diccionario con la solución
            truncar ([boolean]): Saber si se trunca la base de datos si la hay
        """
        if truncar:
            bd_antiguas = dict()
            bds = os.listdir('./bd/')
            for bd_actual in bds:
                bd_actual = open('./bd/'+bd_actual, 'rb')
                bd_dict = pickle.load(bd_actual)
                for key,value in bd_dict.items():
                    for v in value:
                        bd[key].append(v)
                bd_actual.close()
        file = open('./bd/'+nombre, "wb")
        pickle.dump(bd, file)
        file.close()

    def __criba(self, archivo, kp, des, kp_correctos):
        """Criba los keypoints encontrados en función de KP_RANGE. KP_RANGE es la máximo distancia a la que puede estar
        un posible keypoint correcto de uno de los puntos. Si está más lejos, no es un keypoint válido. 

        Args:
            archivo ([str]): Nombre del archivo actual
            kp ([[Keypoint]]): Keypoints obtenidos del archivo actual
            des ([[int]]): Descriptores obtenidos del archivo actual
            kp_correctos ([[Keypoint,[int]]]): Si el keypoint es válido se guarda también su descriptor correspondiente

        Returns:
            [[Keypoint,[int]]]: Keypoints y descriptores válidos
        """
        for i, k in enumerate(kp):
            (kx, ky) = (round(k.pt[0]), round(k.pt[1]))
            for (img_x, img_y) in self.puntos[archivo]:
                if (abs(kx-img_x) < self.KP_RANGE)and(abs(ky-img_y) < self.KP_RANGE):
                    kp_correctos.append([k, des[i]])
        return kp_correctos

    def __localizar_kp_correctos(self, nkp, punt, archivo, img, kp_correctos):
        """Proceso por el que se obtienen los keypoint validos de una imagen, detectando los posibles keypoints
        tras umbralizar la imagen

        Args:
            nkp ([int]): Número de posibles keypoints a destacar en cada foto
            punt ([int]): Cota del umbral para binarizar la foto 
            archivo ([str]): Nombre del archivo actual
            img ([numpy.ndarray]): Imagen original a umbralizar
            kp_correctos ([[Keypoint,[int]]]): Keypoints y descriptores válidos
        
        Returns:
            [[Keypoint,[int]]]: Keypoints y descriptores válidos
        """
        #ret, thresh = cv2.threshold(img.copy(), punt, self.MAX_IMAGEN, cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(img.copy(), punt, self.MAX_IMAGEN, cv2.THRESH_TOZERO)
        kp = self.orb.detect(thresh, None)
        kp, des = self.orb.compute(thresh, kp)
        kp_correctos = self.__criba(archivo,kp,des,kp_correctos)
        
        #cuatro mejores
        """
        def compare_tuples(tp_compare,kp,des,anterior):
            #más cerca tp1 o tp2 de tp_compare
            import math
            tp1 = kp.pt
            tp2 = anterior[0]
            c1 = ( abs(tp_compare[0]-tp1[0]) , abs(tp_compare[1]-tp1[1]))
            c2 = ( abs(tp_compare[0]-tp2[0]) , abs(tp_compare[1]-tp2[1]))
            h1 = math.sqrt(c1[0]**2 + c1[1]**2)
            h2 = math.sqrt(c2[0]**2 + c2[1]**2)
            return [tp1,kp,des] if h1<h2 else anterior
        
        if len(kp_correctos)>0:
            #[coordenadas del punto, Keypoint, descriptor]
            best_A = [kp_correctos[0][0].pt, kp_correctos[0][0], kp_correctos[0][1]]
            best_B = [kp_correctos[0][0].pt, kp_correctos[0][0], kp_correctos[0][1]]
            best_C = [kp_correctos[0][0].pt, kp_correctos[0][0], kp_correctos[0][1]]
            best_D = [kp_correctos[0][0].pt, kp_correctos[0][0], kp_correctos[0][1]]
            for idx,(kpc,desc) in enumerate(kp_correctos):
                esq = self.puntos[archivo]
                #A
                best_A = compare_tuples(esq[0],kpc,des,best_A)
                #B
                best_B = compare_tuples(esq[1],kpc,des,best_B)
                #C
                best_C = compare_tuples(esq[2],kpc,des,best_C)
                #D
                best_D = compare_tuples(esq[3],kpc,des,best_D)

            dt.mostrar_imagen(img,archivo+" (4 esquinas)",[best_A[0],best_B[0],best_C[0],best_D[0]])
            kp_new = [ best_A[1:3],best_B[1:3],best_C[1:3],best_D[1:3] ]
            kp_correctos = kp_new
        """

        # muestra todos los kp encontrados en la imagen thresh (copia y umbralizado de img)
        #dt.mostrar_kp(thresh,archivo+" (precriba)",kp)
        #dt.mostrar_kp(thresh,archivo+" (postcriba)",[kp[0] for kp in kp_correctos])
        
        return kp_correctos

    def __kp_correctos(self, nkp, archivo, cota_min, cota_max, dif, img, kp_correctos=[]):
        """Se buscan los keypoints correctos de la imagen. Como método de evitar las imágenes con sombras se 
        van a producir sucesivos umbralizados con cota desde cota_max hasta cota_min 

        Args:
            nkp ([int]): Número de posibles keypoints a destacar en cada foto
            archivo ([str]): Nombre del archivo actual
            cota_min ([int]): Cota inferior
            cota_max ([int]): Cota superior
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min
            img ([numpy.ndarray]): Imagen original en la que buscar los keypoints
            kp_correctos ([Keypoint,[int]], optional): Keypoints y descriptores válidos. Por defecto a [].

        Returns:
            [[Keypoint,[int]]]: Keypoints y descriptores válidos
        """
        kp_correctos = []
        cota_max = cota_max if cota_max <= self.MAX_IMAGEN else self.COTA_MAX_STDR
        cota_min = cota_min if cota_min >= self.MIN_IMAGEN else self.COTA_MIN_STDR
        punt = cota_max
        while ((cota_min <= punt)and(punt <= cota_max)):
            #kp_correctos.append(self.__localizar_kp_correctos(nkp,punt,archivo,img,kp_correctos))
            kp_correctos = self.__localizar_kp_correctos(nkp,punt,archivo,img,kp_correctos)
            punt = punt - dif
        
        # muestra todos los puntos que corresponden con img
        #dt.mostrar_kp(img,archivo,[i[0] for i in kp_correctos])
        
        return kp_correctos

    def __fotos_entrenamiento(self,archivos,nkp,cota_min,cota_max,dif):
        """Se van a optener todos los keypoints y descriptores válidos encontrados en cada una de las imagenes de training

        Args:
            archivos ([[str]]]): Una lista con los nombres de todas las imagenes de training
            nkp ([int]): Número de posibles keypoints a destacar en cada foto
            cota_min ([int]): Cota inferior
            cota_max ([int]): Cota superior
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min

        Returns:
            [dict]: Diccionario cuya key es el nombre de cada imagen y value una lista de sus keypoints y descriptores válidos
        """
        bd_validos = dict()
        for idx, archivo in enumerate(archivos):
            img = cv2.imread(self.dir + archivo, 0)
            img = cv2.resize(img, (self.IMG_H_VIEW, self.IMG_W_VIEW))
            
            """
            #Versión 1: median blur y dilatar 
            kernel = np.ones((2,2), np.uint8)
            img = cv2.medianBlur(img,25)
            dt.mostrar_imagen(img, archivo+" - Median") 
            img = cv2.dilate(img, kernel, iterations=5)
            dt.mostrar_imagen(img, archivo+" - Dilate")
            kp_correctos = self.__kp_correctos(nkp, archivo, cota_min, cota_max, dif, img)
            """

            """
            #Versión 2: median blur
            median = cv2.medianBlur(img,25)
            dt.mostrar_imagen(median, archivo)
            kp_correctos = self.__kp_correctos(nkp, archivo, cota_min, cota_max, dif, median)
            """

            dt.mostrar_kp(img,archivo+" (final)",[kp[0] for kp in kp_correctos])
            bd_validos.update({archivo: kp_correctos})
            """
            if idx == 2:
                break
            """
        return bd_validos

    def entrenar(self, nombre_bd, nkp=250, cota_min=130, cota_max=220, dif=2, truncar=False):
        """Función principal del entrenamiento. Formará una base de datos de entrenamiento con 
        todos los keypoints y descriptores de las imágenes de training que se obtendrán con las 
        consecuentes umbralizaciones de la las imágenes. Estas umbralizaciones se van a realizar
        desde las cota_max a la cota_min con una diferencia de dif menos en cada iteración. 

        Args:
            nombre_bd ([str]): Nombre que obtendrá la base de datos de entrenamiento
            nkp ([int]): Número de posibles keypoints a destacar en cada foto. Por defecto a 500.
            cota_min ([int]): Cota inferior. Por defecto a 130.
            cota_max ([int]): Cota superior. Por defecto a 250.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min. Por defecto a 10.
            truncar ([boolean]): Se van a obtener los datos de las bases de datos anteriores y se crea la nueva añadiendo los datos
            nuevos y los anteriores.
        """
        self.orb = cv2.ORB_create(nkp, self.ORB_HARRIS_SCORE, self.ORB_FAST_SCORE)
        if not os.path.isdir(self.dir):
            print("Debe introducir un directorio",self.dir)
            return
        bds = os.listdir('./bd/')
        if nombre_bd in bds:
            print("La base de datos a crear lo debe hacer con un nombre nuevo. Ya existe una base de datos con eso nombre")
            return
        archivos = os.listdir(self.dir)
        self.puntos = self.__cvs_to_dict()
        bd_validos = self.__fotos_entrenamiento(archivos,nkp,cota_min,cota_max,dif)
        self.__guardar_entrenamiento(nombre_bd,bd_validos,truncar)


trainer = Trainer("./img/training/", "./img/groundtruth_train.csv")
trainer.entrenar("bd_training_75_fotos_v1.pkl",truncar=False)
