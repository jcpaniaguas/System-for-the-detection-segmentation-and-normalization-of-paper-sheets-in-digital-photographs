# @author: jcpaniaguas
import cv2
import csv
import pickle
import os
import numpy as np
import math


class Trainer:
    """Clase que necesita de un directorio de imágenes de entrenamiento y archivo groundtruth 
    con la esquinas reales de cada una de las imágenes (redimensionadas previamente a 1700x1700).
    El método principal de la clase es entrenar() que iniciará el entrenamiento y creará una base
    de datos con los keypoints y descriptores válidos de cada imagen. 
    """
    
    ORB_EDGE_THRESHOLD = 7
    ORB_SCALE_FACTOR = 1.5
    ORB_NLEVELS = 15
    RESIZE = (1700,1700)


    def __init__(self, directorio, csv_puntos, rango=-1):
        """Constructor de la clase Trainer.

        Args:
            directorio ([str]): Directorio de imagenes o lista de imágenes.
            csv_puntos ([str]): Archivo .csv con las esquinas de cada imagen.
            rango (int, optional): Número de imágenes del directorio que se 
            van a entrenar. Por defecto a -1, es decir, se entrenan todas las
            del directorio.
        """
        self.dir = directorio
        self.rango = rango
        self.csv_puntos = csv_puntos

    def entrenar(self, nombre_bd):
        """Función principal del entrenamiento. Formará una base de datos de entrenamiento con 
        los cuatro keypoints y descriptores correspondientes a las esquinas de los folios en las 
        imágenes de training. El proceso se realizará con una búsqueda de los keypoints correspondientes
        tras el ecualizado y sucesivas ventanaciones de cada imagen.

        Args:
            nombre_bd ([str]): Nombre que obtendrá la base de datos de entrenamiento.
            y se crea la nueva añadiendo los datos nuevos y los anteriores. Por defecto a False.
        """
        self.orb = cv2.ORB_create(scaleFactor=self.ORB_SCALE_FACTOR, nlevels=self.ORB_NLEVELS, edgeThreshold=self.ORB_EDGE_THRESHOLD)
        if not os.path.isdir(self.dir):
            print("Debe introducir un directorio",self.dir)
            return
        bds = os.listdir('./bd/')
        if nombre_bd in bds:
            print("La base de datos a crear lo debe hacer con un nombre nuevo. Ya existe una base de datos con eso nombre.")
            return
        archivos = os.listdir(self.dir)
        img_entrenadas = "./img/Imagenes_Entrenadas_"+str(self.rango)+".txt"
        with open(img_entrenadas,'w') as f:
            for a in archivos[:self.rango]:
                f.write(a+'\n')
        self.puntos = self.__cvs_to_dict()
        bd_validos = self.__fotos_entrenamiento(archivos)
        self.__guardar_entrenamiento(nombre_bd,bd_validos)
    
    def __cvs_to_dict(self):
        """Con un csv como entrada devuelve un diccionario con sus valores tal que la primera columna será la key
        y las demás los valores.

        Returns:
            [dict([(int,int)])]: Diccionario cuya key es el el nombre de la imagen y cuyos values son los puntos
            que se consideran esquinas. 
        """
        csv_dict = dict()
        with open(self.csv_puntos) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                actual_id = row["id"]
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
                csv_dict[actual_id] = actual_list

        return csv_dict

    def __fotos_entrenamiento(self,archivos):
        """Se van a optener todos los keypoints y descriptores válidos encontrados en cada una de las imagenes de training.

        Args:
            archivos ([[str]]]): Una lista con los nombres de todas las imagenes de training.

        Returns:
            [dict]: Diccionario cuya key es el nombre de cada imagen y value una lista de sus keypoints y descriptores válidos.
        """
        bd_validos = dict()
        self.rango = len(archivos) if self.rango==-1 else self.rango
        pos_actual = 0

        while pos_actual!=self.rango:
            print(pos_actual)
            archivo = archivos[pos_actual]
            img = cv2.imread(self.dir + archivo, 0)
            if not (img.shape == self.RESIZE):
                img = cv2.resize(img,self.RESIZE)
                print("Imagen redimensionada a ",img.shape)

            i90 = np.rot90(img)
            i180 = np.rot90(img,2)
            i270 = np.rot90(img,3)
            iespec = np.fliplr(img)
            
            kp_correctos = self.__encontrar_kp(archivo,img)
            kp_c90 = self.__encontrar_kp(archivo,i90,t=90)
            kp_c180 = self.__encontrar_kp(archivo,i180,t=180)
            kp_c270 = self.__encontrar_kp(archivo,i270,t=270)
            kp_espec = self.__encontrar_kp(archivo,iespec,t=-1)
            
            todos = []

            for kp in [kp_correctos,kp_c90,kp_c180,kp_c270,kp_espec]:
                l = list(kp.values())
                for kd in l:
                    todos.append(kd) 

            bd_validos.update({archivo: todos})
            pos_actual += 1

        return bd_validos

    def __encontrar_kp(self, archivo, img, t=0):
        """En la imagen actual closing se lleva un proceso de ecualizado y
        de búsqueda de los keypoints por ventanas (trozos de la imagen orginal)
        para obtener como resultado los cuatro puntos correspondientes con las
        esquinas del folio.

        Args:
            archivo ([str]): Nombre del archivo actual.
            img ([numpy.ndarray]): Imagen original en escala de grises.

        Returns:
            [dict]: Diccionario cuya key es una de las cuatro esquinas y value una lista
            de sus keypoints y descriptores válidos.
        """
        kernel = np.ones((35,35),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        closing = cv2.erode(dilation,kernel,iterations = 1)
        (height, width) = closing.shape
        kp_list = []   
        kp = self.orb.detect(closing,None)
        kp_list, des_list = self.orb.compute(closing, kp)
        puntos = {archivo:[(kp.pt[0],kp.pt[1]) for kp in kp_list]}
        todos = cv2.drawKeypoints(img, kp_list, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        nombre = "./img/giros_kp/"+str(t)+"_"+archivo
        cv2.imwrite(nombre,todos)
        kp_correctos = dict()
        return self.__criba(archivo,img,t,kp_list,des_list,kp_correctos)

    def __criba(self, archivo, img, t, kp, des, kp_correctos):
        """Criba los keypoints devolviendo los más cercanos a los puntos
        descritos en el archivo csv_puntos.

        Args:
            archivo ([str]): Nombre del archivo actual.
            kp ([[Keypoint]]): Keypoints obtenidos del archivo actual.
            des ([[int]]): Descriptores obtenidos del archivo actual.
            kp_correctos ([[Keypoint,[int]]]): Si el keypoint es válido se 
            guarda también su descriptor correspondiente.

        Returns:
            [[Keypoint,[int]]]: Keypoints y descriptores válidos.
        """
        

        esquinas = self.puntos[archivo]
        nuevas = []
        h,w = img.shape
        for i,e in enumerate(esquinas):
            nuevas.append(self.__transf_puntos(e[0],e[1],h,w,t))         

        for i, k in enumerate(kp):
            (kx, ky) = (round(k.pt[0]), round(k.pt[1]))
            for idx,(img_x, img_y) in enumerate(nuevas):
                kp_correctos = self.__insertar_mejor(kp_correctos,idx,(img_x,img_y),k,des[i])
        return kp_correctos

    def __insertar_mejor(self,kp_correctos,idx,punto,k,d):
        """Se seleccionan los mejores keypoints que se acerquen a los puntos del groundtruth.

        Args:
            kp_correctos ([{Keypoint}]): Diccionario de keypoints de los mejores keypoints.
            idx ([int]): Numero de la esquina.
            punto ([(int,int)]): Punto al que acercarse.
            k ([[Keypoint]]): Keypoint actual.
            d ([[int]]): Descriptor actual.

        Returns:
            [{Keypoint}]: Diccionario de keypoints de los mejores keypoints.
        """
        if idx in kp_correctos.keys():
            actual = kp_correctos[idx][0].pt
            distancia_act = math.sqrt( ((punto[0]-round(actual[0]))**2)+((punto[1]-round(actual[1]))**2) )
            distancia_new = math.sqrt( ((punto[0]-round(k.pt[0]))**2)+((punto[1]-round(k.pt[1]))**2) )
            if distancia_new < distancia_act:
                kp_correctos[idx] = [k,d]
            elif distancia_new == distancia_act:
                if kp_correctos[idx][0].size > k.size:
                    k_old = kp_correctos[idx][0].size
                    k_new = k.size
                    kp_correctos[idx] = [k,d]
        else:
            kp_correctos[idx] = [k,d]
        return kp_correctos

    def __transf_puntos(self,x,y,h,w,t):
        """Rota los puntos.

        Args:
            x ([int]): Coordenada X del punto original.
            y ([int]): Coordenada Y del punto original.
            h ([int]): Altura de la imagen.
            w ([int]): Anchura de la imagen.
            t ([int]): Angulo de rotacion: 90,180,270,-1(especular),diferente(original)

        Returns:
            [(int,int)]: Punto rotado.
        """
        if t==90:
            return (y,h-x)
        elif t==180:
            return (w-x,h-y)
        elif t==270:
            return (w-y,x)
        elif t==-1:
            return (w-x,y)
        else:
            return (x,y)


    def __guardar_entrenamiento(self, nombre, bd):
        """Se crea la base de datos y se guarda el resultado del entrenamiento.

        Args:
            nombre ([str]): Nombre del futuro archivo.
            bd ([dict()]): Diccionario con la solución.
        """
        file = open('./bd/'+nombre, "wb")
        pickle.dump(bd, file)
        file.close()