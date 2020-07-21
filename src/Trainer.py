# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt


class Trainer:
    """Clase que necesita de un directorio de imágenes de entrenamiento y archivo groundtruth 
    con la esquinas reales de cada una de las imágenes.
    El método principal de la clase es entrenar() que iniciará el entrenamiento y creará una base
    de datos con los keypoints y descriptores válidos de cada imagen. 
    """
    
    IMG_H_VIEW = 960
    IMG_W_VIEW = 540
    KP_RANGE = 10
    COTA_MAX_STDR = 200
    COTA_MIN_STDR = 130

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
        
        import cv2
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
        import csv
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

    
        """Se dibujan los keypoints en la foto concreta 

        Args:
            img ([numpy.ndarray]): Imagen que se va a mostrar
            archivo ([str]): Nombre de la imagen que se va a mostrar
            kp ([[keypoint]]): Lista con los keypoints encontrados en la imagen actual
        """
        import cv2
        img_kp = cv2.drawKeypoints(
            img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self._mostrar_imagen(img_kp, archivo, None, False)

    def __lista_to_csv(self, lista):
        """Sirve como funcion de apoyo para mejores_parametros y crea un archivo .csv 
        con los datos de lista. 

        Args:
            lista ([[int]]): incluye una matriz con los mejores parametros
        """
        import csv
        with open("./img/mejores_parametros.csv", "wt") as f:
            writer = csv.writer(f)
            writer.writerows(lista)

    def __mejores_parametros(self):
        """Intenta encontrar los mejores parametros con los que ejecutar el entrenamiento
        con el objetivo de encontrar el maximo numero de keypoints.
        En este caso con esta ejecución de ejemplo, la mejor solución corresponde con
        los máximos valores de nkp, cota_min, cota_max.
        """
        mejores_parametros = []
        cota_min = 130
        nkp_list = [100, 200, 300, 400, 500]
        cota_max_list = [150, 170, 190, 210, 230, 250]
        for nkp in nkp_list:
            for cota_max in cota_max_list:
                print(nkp, " ", cota_max)
                kp_encontrados = self.entrenar(
                    nkp=nkp, cota_min=cota_min, cota_max=cota_max)
                mejores_parametros.append(
                    [nkp, cota_max, cota_min, kp_encontrados])
        mejores_parametros.sort(key=lambda p: p[3])
        mejores_parametros.insert(
            0, ['nkp', 'cota_max', 'cota_min', 'kp_encontrados'])
        self.__lista_to_csv(mejores_parametros)

    def __guardar_entrenamiento(self, nombre, bd):
        """Se crea la base de datos y se guarda el resultado del entrenamiento

        Args:
            nombre ([str]): Nombre del futuro archivo
            bd ([dict()]): Diccionario con la solución
        """
        import pickle
        file = open(nombre, "wb")
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
        import cv2
        orb = cv2.ORB_create(nkp, 1.3, 4)
        ret, thresh = cv2.threshold(img.copy(), punt, 255, cv2.THRESH_BINARY)
        kp = orb.detect(thresh, None)
        kp, des = orb.compute(thresh, kp)
        kp_correctos = self.__criba(archivo,kp,des,kp_correctos)
        # muestra todos los kp encontrados en la imagen thresh (copia y umbralizado de img)
        # self.mostrar_kp(thresh,archivo,kp)
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
        cota_max = cota_max if cota_max < 256 else self.COTA_MAX_STDR
        cota_min = cota_min if cota_min >= 0 else self.COTA_MIN_STDR
        punt = cota_max
        while ((cota_min <= punt)and(punt <= cota_max)):
            kp_correctos = self.__localizar_kp_correctos(nkp,punt,archivo,img,kp_correctos)
            punt = punt - dif
        # muestra todos los puntos que corresponden con img
        # self.mostrar_kp(img,archivo,[i[0] for i in kp_correctos])
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
        import cv2
        bd_validos = dict()
        for idx, archivo in enumerate(archivos):
            img = cv2.imread(self.dir + archivo, 0)
            img = cv2.resize(img, (self.IMG_H_VIEW, self.IMG_W_VIEW))
            # muestra las imagenes con los puntos señalados en el csv
            # self.mostrar_imagen(img,archivo,puntos)
            orb = cv2.ORB_create(nkp, 1.3, 4)
            kp_correctos = self.__kp_correctos(nkp, archivo, cota_min, cota_max, dif, img)

            bd_validos.update({archivo: kp_correctos})
        return bd_validos

    def entrenar(self, nombre_bd, nkp=500, cota_min=130, cota_max=250, dif=10):
        """Función principal del entrenamiento. Formará una base de datos de entrenamiento con 
        todos los keypoints y descriptores de las imágenes de training 

        Args:
            nombre_bd ([str]): Nombre que obtendrá la base de datos de entrenamiento
            nkp ([int]): Número de posibles keypoints a destacar en cada foto. Por defecto a 500.
            cota_min ([int]): Cota inferior. Por defecto a 130.
            cota_max ([int]): Cota superior. Por defecto a 250.
            dif ([int]): Cuanto se le va a restar a la cota_max para que disminuya en cada iteración hasta la cota_min. Por defecto a 10.
        """
        import os
        if not os.path.isdir(self.dir):
            print("Debe introducir un directorio",self.dir)
            return 0
        archivos = os.listdir(self.dir)
        self.puntos = self.__cvs_to_dict()
        bd_validos = self.__fotos_entrenamiento(archivos,nkp,cota_min,cota_max,dif)
        self.__guardar_entrenamiento(nombre_bd,bd_validos)


trainer = Trainer("./img/training/", "./img/groundtruth.csv")
trainer.entrenar("bd_training.pkl")
