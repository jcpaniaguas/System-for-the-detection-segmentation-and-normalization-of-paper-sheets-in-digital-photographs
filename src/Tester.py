# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt


class Tester:
    """
    Clase que necesita de una base de datos de entrenamiento y el directorio de las imagénes de test
    de las cuales se van a intentar localizar los folios.
    La función que va a iniciar la búsqueda es localizar(), que funciona bien con un directorio bien
    con una sola imagen.
    """

    IMG_H_VIEW = 960
    IMG_W_VIEW = 540
    KP_RANGE = 10
    COTA_MAX_STDR = 200
    COTA_MIN_STDR = 130

    def __init__(self, bd):
        """Constructor de la clase Tester.

        Args:
            bd ([str]): Archivo .pkl con el entrenamiento realizado.
        """
        import pickle
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

    def localizar(self, path, nkp=500, cota_min=130, cota_max=250, dif=10):
        """Función principal de la clase. Intenta localizar el folio en la imagen
        dependiendo de si ingresa un path con la localización de una imagen
        o de un directorio de imágenes.

        Args:
            path ([str]): Path donde se localiza la imagen o el directorio de imágenes a analizar.
            nkp ([int]): Número de posibles keypoints a destacar en cada foto. Por defecto a 500.
            cota_min ([int]): Cota inferior. Por defecto a 130.
            cota_max ([int]): Cota superior. Por defecto a 250.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min. Por defecto a 10.
        """
        import os
        if os.path.isdir(path):
            print("Introdujo el directorio", path)
            self.__localizar_dir(path, nkp, cota_min, cota_max, dif)
        elif os.path.isfile(path):
            print("Introdujo el archivo", path)
            (path, archivo) = os.path.split(path)
            self.__localizar_archivo(path, archivo, nkp, cota_min, cota_max, dif)
        else:
            print("Debe introducir un directorio o un archivo")

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
        import os
        archivos = os.listdir(path)
        for archivo in archivos:
            self.__localizar_archivo(path, archivo, nkp, cota_min, cota_max, dif)

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
        import cv2
        print(path+archivo)
        img = cv2.imread(path+'/'+archivo, 0)
        img = cv2.resize(img, (self.IMG_H_VIEW, self.IMG_W_VIEW))
        self.__desempaquetar()
        self.__coincidencias(img, archivo, nkp, cota_min, cota_max, dif)

    def __localizar_kp(self, nkp, punt, archivo, img, puntuaciones):
        """Umbraliza la imagen para localizar las coincidencias con el entrenamiento y
        actualizar las puntuaciones.

        Args:
            nkp ([int]): Número de posibles keypoints a destacar en cada foto.
            punt ([int]): Cota del umbralizado.
            archivo ([str]): Nombre de la imagen.
            img ([numpy.ndarray]): Imagen actual.
            puntuaciones ([dict]): Diccionario que lleva las puntuaciones cuantas veces un keypoint.
            se localiza en un sitio. La key es la coordenada del keypoint y el valor las veces encontrado.
        """
        import cv2
        orb = cv2.ORB_create(nkp, 1.3, 4)
        ret, thresh = cv2.threshold(img.copy(), punt, 255, cv2.THRESH_BINARY)
        kp = orb.detect(thresh, None)
        kp, des = orb.compute(thresh, kp)
        matches = self.__encontrar_matches(des)
        self.__puntuar_matches(kp, matches, puntuaciones)

    def __encontrar_esquinas(self, puntuaciones):
        """Ordena según las coordenadas entrantes las esquinas del folio.

        Args:
            puntuaciones ([[(float,foat),int]]): Las cuatro coordenadas mejor valoradas y su puntuación.

        Returns:
            [numpy.array]: Las cuatro coordenadas asignadas a las cuatro esquinas del folio. 
        """
        import numpy as np
        A = puntuaciones[0][0]
        B = puntuaciones[1][0]
        C = puntuaciones[2][0]
        D = puntuaciones[3][0]
        coords = [A, B, C, D]
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

    def __cribar_puntuaciones(self, puntuaciones):
        """Va a actualizar las puntuaciones del diccionario puntuaciones entendiendo cada coordenadas
        a una distancia de KP_RANGE como iguales.

        Args:
            puntuaciones ([dict]): Diccionario que lleva las puntuaciones cuantas veces un keypoint.

        Returns:
            [[((float,float),int)]]: Lista ordenada de tuplas con las coordenadas y las veces 
            que las encontramos en la imagen.
        """
        punt_list = list(puntuaciones.keys())
        for idx1,(x_actual, y_actual) in enumerate(punt_list):
            idx2 = 0
            for (x_sig, y_sig) in punt_list[idx2+idx1+1:]: 
                if (abs(x_actual - x_sig) < self.KP_RANGE) & (abs(y_actual - y_sig) < self.KP_RANGE):
                    puntuaciones[(x_actual,y_actual)] += 1
                    del puntuaciones[(x_sig,y_sig)]
                    del punt_list[idx1+idx2+1]
                else:
                    idx2 += 1
        puntuaciones = sorted(puntuaciones.items(), key=lambda x: x[1], reverse=True)
        return puntuaciones

    def __encontrar_matches(self, des):
        """Se van a buscar coincidencias entre la imagen actual y el entrenamiento.

        Args:
            des ([numpy.ndarray]): Descriptores de la imagen en la que buscar coincidencias.

        Returns:
            [[DMatch]]: Las coincidencias encontradas.
        """
        import numpy as np
        import cv2
        # para la comparacion se transforma el array de descriptores en un numpy array
        self.des_entrenados = np.array(self.des_entrenados)
        # busqueda de coincidencias con BruteForceMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, self.des_entrenados)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def __puntuar_matches(self, kp, matches, puntuaciones):
        """Se va a encargar de calcular las puntuaciones. 

        Args:
            kp ([[Keypoint]]): Keypoints de la imagen.
            matches ([[DMatch]]): Las coincidencias encontradas en la imagen.
            puntuaciones ([dict]): Diccionario que lleva las puntuaciones cuantas veces un keypoint.
        """
        for idx, match in enumerate(matches):
            posible_esq = kp[match.queryIdx]
            posible_esq_x = "{0:.2f}".format(round(posible_esq.pt[0]))
            posible_esq_y = "{0:.2f}".format(round(posible_esq.pt[1]))
            coord = (float(posible_esq_x), float(posible_esq_y))
            if coord in puntuaciones:
                contador = puntuaciones.get(coord)
                contador += 1
                puntuaciones.update({coord: contador})
            else:
                puntuaciones.update({coord: 1})

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
        import cv2
        import numpy as np
        puntuaciones = dict()
        cota_max = cota_max if cota_max < 256 else self.COTA_MAX_STDR
        cota_min = cota_min if cota_min >= 0 else self.COTA_MIN_STDR
        punt = cota_max
        while ((cota_min <= punt)and(punt <= cota_max)):
            self.__localizar_kp(nkp,punt,archivo,img,puntuaciones)
            punt = punt - dif

        puntuaciones = self.__cribar_puntuaciones(puntuaciones)
        puntos = {archivo: [puntuaciones[0][0],puntuaciones[1][0],puntuaciones[2][0],puntuaciones[3][0]]}
        dt.mostrar_imagen(img, archivo, puntos)
        from_C = self.__encontrar_esquinas(puntuaciones[:4])
        to_C = np.float32([[0, 0],[self.IMG_W_VIEW, 0],[0, self.IMG_H_VIEW],[self.IMG_W_VIEW,self.IMG_H_VIEW]])
        dt.transformar_perspectiva(img, archivo, from_C, to_C)
        return {archivo:from_C}


tester = Tester("./bd_training.pkl")
tester.localizar("./img/testing/")
tester.localizar("./img/testing/2stickies.jpg")
