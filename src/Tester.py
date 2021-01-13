# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt
from Analyzer import Analyzer
import pickle
import cv2
import os
import numpy as np


class Tester:
    """
    Clase que necesita de una base de datos de entrenamiento y el directorio de las imagénes de test
    de las cuales se van a intentar localizar los folios.
    La función que va a iniciar la búsqueda es localizar(), que funciona bien con un directorio bien
    con una sola imagen.
    """

    IMG_H_VIEW = 960
    IMG_W_VIEW = 540
    KP_RANGE = 2
    COTA_MAX_STDR = 200
    COTA_MIN_STDR = 130
    MAX_IMAGEN = 255
    MIN_IMAGEN = 0
    ORB_HARRIS_SCORE = 1.3
    ORB_FAST_SCORE = 4
    MAT_SIZE = 100

    def __init__(self, bd):
        """Constructor de la clase Tester.

        Args:
            bd ([str]): Archivo .pkl con el entrenamiento realizado.
        """
        self.bd = bd
        archivo = open(bd, 'rb')
        self.entrenamiento = pickle.load(archivo)
        archivo.close()

    def __desempaquetar(self):
        """Desempaquetar los valores de entrenamiento en una lista de 
        keypoints y una lista de descriptores.
        """
        self.kp_entrenados = []
        self.des_entrenados = []
        for path, key_des_list in self.entrenamiento.items():
            for key_des in key_des_list:
                kp = key_des[0]
                des = key_des[1]
                self.kp_entrenados.append(kp)
                self.des_entrenados.append(des)

    def localizar(self, path, nkp=250, cota_min=130, cota_max=220, dif=2):
        """Función principal de la clase. Intenta localizar el folio en la imagen
        dependiendo de si ingresa un path con la localización de una imagen
        o de un directorio de imágenes.
        Va a buscar las coincidencias con una base de datos entrenados. Para ello,
        realiza una serie de umbralizaciones en la imagen a comparar, que van de 
        la cota_max a la cota_min con una diferencia de dif menos en cada iteración.

        Args:
            path ([str]): Path donde se localiza la imagen o el directorio de imágenes a analizar.
            nkp ([int]): Número de posibles keypoints a destacar en cada foto. Por defecto a 500.
            cota_min ([int]): Cota inferior. Por defecto a 130.
            cota_max ([int]): Cota superior. Por defecto a 250.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min. Por defecto a 10.
        """
        self.analizador = Analyzer('Analyzer.csv',[path])
        self.orb = cv2.ORB_create(nkp, self.ORB_HARRIS_SCORE, self.ORB_FAST_SCORE)
        if os.path.isdir(path):
            print("Introdujo el directorio", path)
            self.__localizar_dir(path, nkp, cota_min, cota_max, dif)
        elif os.path.isfile(path):
            print("Introdujo el archivo", path)
            (path, archivo) = os.path.split(path)
            self.__localizar_archivo(path, archivo, nkp, cota_min, cota_max, dif)
        else:
            print("Debe introducir un directorio o un archivo")
        self.analizador.porcentaje_acierto(self.bd.split('/')[-1])

    def __localizar_dir(self, path, nkp, cota_min, cota_max, dif):
        """Cuando el path indicado en la función localizar es un directorio
        localiza cada una de las imágenes para trabajar con ellas y las 
        deriva a la función localizar_archivo que trabaja con ella una a una.

        Args:
            path ([str]): Path donde se localiza el directorio de imágenes a analizar.
            nkp ([int]): Número de posibles keypoints a destacar en cada foto.
            cota_min ([int]): Cota inferior.
            cota_max ([int]): Cota superior.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min.
        """
        archivos = os.listdir(path)
        for idx,archivo in enumerate(archivos):
            self.__localizar_archivo(path, archivo, nkp, cota_min, cota_max, dif)
            """
            if idx == 2:
                break
            """

    def __localizar_archivo(self, path, archivo, nkp, cota_min, cota_max, dif):
        """Cuando el path indicado en la función localizar correspondia a una imagen
        se utiliza esta función para traer la imagen y buscar las esquinas.

        Args:
            path ([str]): Path donde se localiza la imagen a analizar.
            archivo ([str]): Nombre de la imagen actual.
            nkp ([int]): Número de posibles keypoints a destacar en cada foto.
            cota_min ([int]): Cota inferior.
            cota_max ([int]): Cota superior.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min.
        """
        print(path+archivo)
        img = cv2.imread(path+'/'+archivo, 0)
        img = cv2.resize(img, (self.IMG_H_VIEW, self.IMG_W_VIEW))
        self.__desempaquetar()
        #"""
        #Versión 1: median blur y dilatar 
        kernel = np.ones((2,2), np.uint8)
        img = cv2.medianBlur(img,25)
        #dt.mostrar_imagen(img, archivo+" - Median") 
        img = cv2.dilate(img, kernel, iterations=5)
        #dt.mostrar_imagen(img, archivo+" - Dilate")
        coincidencias = self.__coincidencias(img, archivo, nkp, cota_min, cota_max, dif)
        #"""
        """
        #Versión 2: median blur
        median = cv2.medianBlur(img,25)
        dt.mostrar_imagen(median, archivo)
        coincidencias = self.__coincidencias(median, archivo, nkp, cota_min, cota_max, dif)
        """
        #dt.mostrar_imagen(img, archivo)
        ###
        #coincidencias = self.__coincidencias(img, archivo, nkp, cota_min, cota_max, dif)
        #mostrar final
        dt.mostrar_imagen(img, archivo, {archivo: [(i[0],i[1]) for i in coincidencias[archivo]]})
        self.__analizar_resultado(archivo)
        from_C = coincidencias[archivo]
        to_C = np.float32([[0, 0],[self.IMG_W_VIEW, 0],[0, self.IMG_H_VIEW],[self.IMG_W_VIEW,self.IMG_H_VIEW]])
        dt.transformar_perspectiva(img, archivo, from_C, to_C)

    def __analizar_resultado(self, archivo):
        """.

        Args:
            archivo ([str]): Nombre de la imagen actual.
        """
        print("¿Se ha encontrado el folio?: S/N")
        respuesta = True
        dato = ""
        while respuesta:
            dato = input()
            if dato in ['s','S','n','N']:
                dato = dato.upper()     
                respuesta = False
            else:
                print("Debe introducir: S/N")
        """
        Analizador: 
            insertar(self,foto_referencia,origen,contenido) -> 
            foto_referencia = archivo
            origen = self.bd
            contenido = dato
        """
        self.analizador.insertar(archivo, self.bd.split('/')[-1], dato)


    def __localizar_kp(self, nkp, punt, archivo, img):
        """Umbraliza la imagen para localizar las coincidencias con el entrenamiento y
        actualizar las puntuaciones.

        Args:
            nkp ([int]): Número de posibles keypoints a destacar en cada foto.
            punt ([int]): Cota del umbralizado.
            archivo ([str]): Nombre de la imagen.
            img ([numpy.ndarray]): Imagen actual.
        """
        ret, thresh = cv2.threshold(img.copy(), punt, self.MAX_IMAGEN, cv2.THRESH_BINARY)
        kp = self.orb.detect(thresh, None)
        kp, des = self.orb.compute(thresh, kp)
        #dt.mostrar_kp(img,archivo+" (prematches)",kp)
        matches = self.__encontrar_matches(des)
        #dt.mostrar_matches(img,archivo+" (postmatches)",kp,matches)
        self.__puntuar_matches(kp, matches)

    def __encontrar_esquinas(self, puntuaciones):
        """Ordena según las coordenadas entrantes las esquinas del folio.

        Args:
            puntuaciones ([[(float,foat),int]]): Las cuatro coordenadas mejor valoradas y su puntuación.

        Returns:
            [numpy.array]: Las cuatro coordenadas asignadas a las cuatro esquinas del folio. 
        """
        A = puntuaciones[0][0]
        B = puntuaciones[1][0]
        C = puntuaciones[2][0]
        D = puntuaciones[3][0]
        coords = [A, B, C, D]
        #dt.mostrar_imagen(img, archivo, puntos=[], circulos=True)
        coords = sorted(coords)
        if coords[0][1] <= coords[1][1]:
            A = coords[0]
            C = coords[1]
        else:
            C = coords[0]
            A = coords[1]

        if coords[2][1] <= coords[3][1]:
            B = coords[2]
            D = coords[3]
        else:
            D = coords[2]
            B = coords[3]
        return np.float32([A, B, C, D])

    def __maximas_puntuaciones(self):
        """Va a actualizar las puntuaciones del diccionario dic_puntuaciones entendiendo cada coordenadas
        a una distancia de KP_RANGE como iguales.

        Returns:
            [[((float,float),int)]]: Lista ordenada de tuplas con las coordenadas y las veces 
            que las encontramos en la imagen.
        """
        sort_puntuaciones = sorted(self.dic_puntuaciones.items(), key=lambda x: x[1], reverse=True)
        new_sort_punt = list()
        for idx1,tupla1 in enumerate(sort_puntuaciones):
            idx2 = 0
            ((x1,y1),_) = tupla1
            for tupla2 in sort_puntuaciones[idx2+idx1+1:]:
                ((x2,y2),p2) = tupla2
                if (abs(x1-x2) <= self.KP_RANGE) & (abs(y1-y2) <= self.KP_RANGE):
                    self.dic_puntuaciones[(x1,y1)] += p2
                    del self.dic_puntuaciones[(x2,y2)]
                    del sort_puntuaciones[idx1+idx2+1]
                else:
                    idx2 += 1
        sort_puntuaciones = sorted(self.dic_puntuaciones.items(), key=lambda x: x[1], reverse=True)
        for (coord,punt) in sort_puntuaciones:
            (x,y) = (int(coord[0]),int(coord[1]))
            trans_x = int((self.IMG_H_VIEW*x)/self.MAT_SIZE)
            trans_y = int((self.IMG_W_VIEW*y)/self.MAT_SIZE)
            new_sort_punt.append( ((trans_x,trans_y),punt) )
        return new_sort_punt

    def __encontrar_matches(self, des):
        """Se van a buscar coincidencias entre la imagen actual y el entrenamiento.

        Args:
            des ([numpy.ndarray]): Descriptores de la imagen en la que buscar coincidencias.

        Returns:
            [[DMatch]]: Las coincidencias encontradas.
        """
        # para la comparacion se transforma el array de descriptores en un numpy array
        self.des_entrenados = np.array(self.des_entrenados)
        # busqueda de coincidencias con BruteForceMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, self.des_entrenados)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def __puntuar_matches(self, kp, matches):
        """Se va a encargar de calcular las puntuaciones. 

        Args:
            kp ([[Keypoint]]): Keypoints de la imagen.
            matches ([[DMatch]]): Las coincidencias encontradas en la imagen.
        """
        for idx, match in enumerate(matches):
            posible_esq = kp[match.queryIdx]
            posible_esq_x = "{0:.2f}".format(round(posible_esq.pt[0]))
            posible_esq_y = "{0:.2f}".format(round(posible_esq.pt[1]))
            coord = (float(posible_esq_x), float(posible_esq_y))
            self.__coord_a_puntuacion(coord)
    
    def __coord_a_puntuacion(self,coord):
        (x,y) = (int(coord[0]),int(coord[1]))
        transf_x = int((x/self.IMG_H_VIEW)*self.MAT_SIZE)
        transf_y = int((y/self.IMG_W_VIEW)*self.MAT_SIZE)
        self.matriz_puntuaciones[transf_x][transf_y] += 1
        if (transf_x,transf_y) in self.dic_puntuaciones:
            self.dic_puntuaciones[(transf_x,transf_y)] += 1
        else:
            self.dic_puntuaciones[(transf_x,transf_y)] = 1
    
    def __coincidencias(self, img, archivo, nkp, cota_min, cota_max, dif):
        """Guardara la puntuacion de cuantos votos tiene cada punto de la imagen
        para ser una esquina y la muestra.

        Args:
            img ([numpy.array]): Imagen actual en la que encontrar coincidencias.
            archivo ([str]): Nombre de la imagen actual.
            nkp ([int]): Número de posibles keypoints a destacar en cada foto.
            cota_min ([int]): Cota inferior.
            cota_max ([int]): Cota superior.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min.

        Returns:
            [dict]: Diccionario con el nombre de la foto como key y las cuatro esquinas ordenadas
            como valor.
        """
        self.dic_puntuaciones = dict()
        self.matriz_puntuaciones = np.zeros((self.MAT_SIZE,self.MAT_SIZE))
        cota_max = cota_max if cota_max <= self.MAX_IMAGEN else self.COTA_MAX_STDR
        cota_min = cota_min if cota_min >= self.MIN_IMAGEN else self.COTA_MIN_STDR
        punt = cota_max
        while ((cota_min <= punt)and(punt <= cota_max)):
            self.__localizar_kp(nkp,punt,archivo,img)
            punt = punt - dif

        puntuaciones = self.__maximas_puntuaciones()
        if len(puntuaciones) == 0:
            print("No se ha encontrado ninguna esquina")
            return -1
        puntos = {archivo: [puntuaciones[0][0],puntuaciones[1][0],puntuaciones[2][0],puntuaciones[3][0]]}
        #dt.mostrar_imagen(img, archivo, puntos)
        from_C = self.__encontrar_esquinas(puntuaciones[:4])
        #to_C = np.float32([[0, 0],[self.IMG_W_VIEW, 0],[0, self.IMG_H_VIEW],[self.IMG_W_VIEW,self.IMG_H_VIEW]])
        #dt.mostrar_imagen(img, archivo, {archivo: [(i[0],i[1]) for i in from_C]})
        #dt.transformar_perspectiva(img, archivo, from_C, to_C)
        return {archivo:from_C}


tester = Tester("./bd/bd_training_75_fotos_v1.pkl")
tester.localizar("./img/training/")
tester.localizar("./img/testing/")
#tester.localizar("./img/testing/2stickies.jpg")
#tester.localizar("./img/training/IMG_7160.JPG")