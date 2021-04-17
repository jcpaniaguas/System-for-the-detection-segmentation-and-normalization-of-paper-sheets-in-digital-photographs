# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt
from matplotlib import pyplot as plt
import pickle
import cv2
import os
import numpy as np
import math

class Tester:
    """
    Clase que necesita de una base de datos de entrenamiento y el directorio de las imagénes de test
    de las cuales se van a intentar localizar los folios.
    La función que va a iniciar la búsqueda es localizar(), que funciona bien con un directorio bien
    con una sola imagen.
    """

    ORB_EDGE_THRESHOLD = 7
    ORB_SCALE_FACTOR = 1.5
    ORB_NLEVELS = 15
    RESIZE = (1700,1700)
    KP_RANGE = 200
    DIST_PAR = 300
    ANG_MAX = 120
    ANG_MIN = 60

    def __init__(self, bd, guardar=0):
        """Constructor de la clase Tester.

        Args:
            bd ([str]): Archivo .pkl con el entrenamiento realizado.
            guardar (int, optional): Si guardar es 0 no se guardan fotos de resultado. 
            Si guardar es 1 se va crear la imagen resultado. Si guardar es 2 se 
            imprime la imagen por pantalla. Por defecto a 0.
        """
        self.guardar = guardar
        self.bd = bd
        archivo = open(bd, 'rb')
        self.entrenamiento = pickle.load(archivo)
        archivo.close()

    def localizar(self, path):
        """Función principal de la clase. Intenta localizar el folio en la imagen
        dependiendo de si ingresa un path con la localización de una imagen
        o de un directorio de imágenes.
        Se realizará el ecualizado y sucesivas ventanaciones de cada imagen 
        obteniendo keypoints y buscando las coincidencias con una base de datos entrenados.

        Args:
            path ([str]): Path donde se localiza la imagen o el directorio de imágenes a analizar.
        """
        self.orb = cv2.ORB_create(scaleFactor=self.ORB_SCALE_FACTOR, nlevels=self.ORB_NLEVELS, edgeThreshold=self.ORB_EDGE_THRESHOLD)
        if os.path.isdir(path):
            print("Introdujo el directorio", path)
            return self.__localizar_dir(path)
        elif os.path.isfile(path):
            print("Introdujo el archivo", path)
            (path, archivo) = os.path.split(path)
            k,v = self.__localizar_archivo(path, archivo)
            return {k:v}
        else:
            print("Debe introducir un directorio o un archivo")

    def __localizar_dir(self, path):
        """Cuando el path indicado en la función localizar es un directorio
        localiza cada una de las imágenes para trabajar con ellas y las 
        deriva a la función localizar_archivo que trabaja con ella una a una.

        Args:
            path ([str]): Path donde se localiza el directorio de imágenes a analizar.
        """
        archivos = os.listdir(path)
        result = {}
        for idx,archivo in enumerate(archivos):
            k,v = self.__localizar_archivo(path, archivo)
            result[k] = v
        return result

    def __localizar_archivo(self, path, archivo):
        """Cuando el path indicado en la función localizar correspondia a una imagen
        se utiliza esta función para traer la imagen y buscar las esquinas.

        Args:
            path ([str]): Path donde se localiza la imagen a analizar.
            archivo ([str]): Nombre de la imagen actual.
        """
        img = cv2.imread(path+'/'+archivo, 0)
        if not (img.shape == self.RESIZE):
                img = cv2.resize(img,self.RESIZE)
                print("Imagen redimensionada a ",img.shape)
        self.__desempaquetar()
        self.dic_puntuaciones = dict()
    
        kernel = np.ones((35,35),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        closing = cv2.erode(dilation,kernel,iterations = 1)

        cuatro = self.__buscar_coincidencias(archivo,closing,img)
        return archivo,cuatro

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

    def __buscar_coincidencias(self,archivo,closing,img):
        """En la imagen actual closing se lleva un proceso de ecualizado y
        de búsqueda de los keypoints por ventanas (trozos de la imagen orginal).
        Después, se buscan coincidencias con la base de datos de aprendizaje y
        se intenta encontrar los cuatro puntos correspondientes con las
        esquinas del folio.

        Args:
            archivo ([str]): Nombre del archivo actual.
            closing ([numpy.ndarray]): Imagen en la que se van a buscar las esquinas.
            img ([numpy.ndarray]): Imagen original en escala de grises.

        Returns:
            [[(int,int)]]: Lista solución con las coordenadas de las cuatro esquinas.
        """
        (height, width) = closing.shape
        kp_list = []
        kp = self.orb.detect(closing,None)
        kp_list, des_list = self.orb.compute(closing, kp)

        todos = cv2.drawKeypoints(img, kp_list, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        #KP
        spl = self.bd.split('_')
        archivo_entrenado = self.__archivo_entrenado(archivo,spl[2])
        nombre_test = spl[2]+"/"+spl[4].split('.')[0]+"_"+archivo
        nombre_test_entrenado = spl[2]+"/Entrenados/"+spl[4].split('.')[0]+"_"+archivo

        if self.guardar==1:
            nombre = "./img/kp_circle/"+nombre_test
            cv2.imwrite(nombre,todos)
        elif self.guardar==2:
            plt.figure(figsize = (8,8))
            plt.imshow(todos)
            plt.show()

        matches = self.__encontrar_matches(np.array([list(d) for d in des_list]))
        m1 = dt.mostrar_matches(img,archivo,kp_list,matches)

        #MATCHES
        if self.guardar==1:
            nombre = ""
            if archivo_entrenado:
                nombre = "./img/match_circle/"+nombre_test_entrenado
            else:
                nombre = "./img/match_circle/"+nombre_test
            cv2.imwrite(nombre,m1)
        elif self.guardar==2:
            plt.figure(figsize = (8,8))
            plt.imshow(m1)
            plt.show()
        
        cuatro = self.__cuatro_mejores(matches,kp_list)
        found,a = self.__folio_valido(archivo,cuatro,kp_list)

        #4 ESQUINAS
        img_match = dt.dibujar_puntos({archivo: a},img,archivo,color=(0,0,255))
        if self.guardar==1:
            nombre = ""
            if archivo_entrenado:
                nombre = "./img/sol_circle/"+nombre_test_entrenado
            else:
                nombre = "./img/sol_circle/"+nombre_test
            cv2.imwrite(nombre,img_match)
        elif self.guardar==2:
            plt.figure(figsize = (8,8))
            plt.imshow(img_match)
            plt.show()

        return cuatro

    def __archivo_entrenado(self,archivo,n_test):
        """Si el archivo pertenece a los que se han utilizado en el entrenamiento.

        Args:
            archivo ([str]): Archivo actual.
            n_test ([int]): Numero de imagenes. Coincira con el nombre del archivo 
            'Imagenes_Entrenadas_{n_test}.txt', archivo con los nombres de las imagenes
            que se han utilizado para el entrenamiento.

        Returns:
            [bool]: True si el archivo ha formado parte del entrenamiento. False si no.
        """
        path = "./img/Imagenes_Entrenadas_"+str(n_test)+".txt"
        entrenados = []
        with open(path,'r') as f:
            e = f.read()
            entrenados = e.split('\n')
        if archivo in entrenados[:-1]:
            return True
        else:
            return False

    def __encontrar_matches(self, des):
        """Se van a buscar coincidencias entre la imagen actual y el entrenamiento.

        Args:
            des ([numpy.ndarray]): Descriptores de la imagen en la que buscar coincidencias.

        Returns:
            [[DMatch]]: Lista de coincidencias encontradas y ordenadas por grado de coincidencia (distancia).
        """
        self.des_entrenados = np.array(self.des_entrenados)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, self.des_entrenados)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def __cuatro_mejores(self,matches,kp):
        """Busca en los matches los cuatro primeros distintos.
        Se entiende que dos matches son distintos cuando sus
        coordenadas estan a más de una distancia KP_RANGE.

        Args:
            matches ([DMatch]): Lista de coincidencias encontradas y ordenadas por grado de coincidencia (distancia). 
            kp ([[Keypoint]]): Keypoints obtenidos del archivo actual.

        Returns:
            [[(int,int)]]: Lista solución con las coordenadas de las cuatro esquinas.
        """
        cuatro = {}
        for m in matches:
            if len(cuatro)==4:
                break
            match_pt = (round(kp[m.queryIdx].pt[0]),round(kp[m.queryIdx].pt[1]))
            if match_pt not in cuatro.keys():
                igual = False
                for c in cuatro.keys():
                    if (abs(c[0]-match_pt[0]) <= self.KP_RANGE) & (abs(c[1]-match_pt[1]) <= self.KP_RANGE):
                        igual = True
                        break    
                if not igual:
                    cuatro[match_pt] = kp[m.queryIdx]
        return list(cuatro.values())
    
    def __regla_paralelogramo(self,a,b,c):
        """Regla del paralelogramo. Con tres puntos dado calcula donde estaria el cuarto
        teniendo en cuenta que este estara enfrente de B.

        Args:
            a ([(int,int)]): Punto A.
            b ([(int,int)]): Punto B.
            c ([(int,int)]): Punto C.

        Returns:
            [(int,int)]: Punto D. El punto enfrente de B.
        """
        a1,a2 = a
        b1,b2 = b
        c1,c2 = c
        return a1+c1-b1,a2+c2-b2

    def __devolver_cuatro(self,p1,p2,p3):
        """Se piden tres puntos y devuelve los cuatro utilizando la regla
        del paralelogramo.

        Args:
            p1 ([(int,int)]): Punto 1.
            p2 ([(int,int)]): Punto 2.
            p3 ([(int,int)]): Punto 3.

        Returns:
            [[(int,int)]]: Lista con los cuatro puntos.
        """
        d12 = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        d23 = math.sqrt(((p3[0]-p2[0])**2)+((p3[1]-p2[1])**2))
        d31 = math.sqrt(((p1[0]-p3[0])**2)+((p1[1]-p3[1])**2))
        mayor = [d12,d23,d31]
        mayor.sort(reverse=True)        
        if mayor[0]==d12:
            #enfrente de p3
            d1,d2 = self.__regla_paralelogramo(p1,p3,p2)
        elif mayor[0]==d23:
            #enfrente de p1
            d1,d2 = self.__regla_paralelogramo(p2,p1,p3)
        else:
            #enfrente de p2
            d1,d2 = self.__regla_paralelogramo(p1,p2,p3)
        return [p1,p2,p3,(d1,d2)]
    
    def __distancia(self,A,B):
        """Distancia entre dos puntos.

        Args:
            A ([(int,int)]): Punto A.
            B ([(int,int)]): Punto B.

        Returns:
            [float]: Distancia entre A y B.
        """
        return math.sqrt(((A[0]-B[0])**2)+((A[1]-B[1])**2))

    def __vector(self,A,B):
        """Obtener el vector AB.

        Args:
            A ([(int,int)]): Punto A.
            B ([(int,int)]): Punto B.

        Returns:
            [(int,int)]: Vector AB.
        """
        return (B[0]-A[0],B[1]-A[1])

    def __analizar_distancia(self,a):
        """Analiza el posible paralelogramo.

        Args:
            a ([[(int,int)]]): Lista de los cuatro puntos.

        Returns:
            [dict()]: Diccionario analizando los cuatro puntos. Devuelve:
                Punto A
                Punto B
                Punto C
                Punto D
                Distancia AB 
                Distancia BD
                Distancia DC
                Distancia CA
                Mitad del segmento AD
                Mitad del segmento BC
                Angulo ABC
                Angulo DBC
                Angulo CDA
                Angulo BDA
                Perimetro 
        """
        found = False
        a = sorted(a, key=lambda p: float(p[1]))
        AB = a[0:2]
        CD = a[2:]
        AB = sorted(AB, key=lambda p: float(p[0]))
        CD = sorted(CD, key=lambda p: float(p[0]))
        A = AB[0]
        B = AB[1]
        C = CD[0]
        D = CD[1]
        A = (int(A[0]),int(A[1]))
        B = (int(B[0]),int(B[1]))
        C = (int(C[0]),int(C[1]))
        D = (int(D[0]),int(D[1]))
        distAB = self.__distancia(A,B)
        distBD = self.__distancia(B,D)
        distDC = self.__distancia(D,C)
        distCA = self.__distancia(C,A)
        distBC = self.__distancia(B,C)
        distAD = self.__distancia(A,D)
        distBC = self.__distancia(B,C)
        ABC = self.__angulos(A,B,C)
        DBC = self.__angulos(D,B,C)
        CDA = self.__angulos(C,D,A)
        BDA = self.__angulos(B,D,A)
        perimetro = 2*distAB*distBC
        d = dict()
        d['A'] = A
        d['B'] = B
        d['C'] = C
        d['D'] = D
        d['distAB'] = distAB 
        d['distBD'] = distBD
        d['distDC'] = distDC
        d['distCA'] = distCA
        d['mitadAD'] = int((A[0]+D[0])/2),int((A[1]+D[1])/2)
        d['mitadBC'] = int((B[0]+C[0])/2),int((B[1]+C[1])/2)
        d['ABC'] = ABC
        d['DBC'] = DBC
        d['CDA'] = CDA
        d['BDA'] = BDA
        d['perimetro'] = perimetro 
        return d

    def __discriminar_punto(self,c,kp):
        """Va a analizar los cuatro puntos dados y los va a dividir en cuatro ternas tal que:
                [A,B,C],
                [A,B,D],
                [B,C,D],
                [A,C,D].
            Con estas cuatro ternas, con la regla del paralelogramo obtiene el cuarto punto y selecciona
            la terna y el nuevo punto mejor. 

        Args:
            c ([[(int)]]): Lista con los cuatro puntos.
            kp ([[Keypoint]]): Lista de los keypoint de la imagen.

        Returns:
            [[(int,int)]]: Tres puntos y el cuarto nuevo en una lista.
        """
        t1 = self.__devolver_cuatro(c[0],c[1],c[2])
        t2 = self.__devolver_cuatro(c[0],c[1],c[3])
        t3 = self.__devolver_cuatro(c[0],c[2],c[3])
        t4 = self.__devolver_cuatro(c[1],c[2],c[3])
        tres = [t1,t2,t3,t4]
        dist = {}
        p = []

        for idx,posibles in enumerate(tres):
            p.append(posibles[3])
            d = self.__analizar_distancia(posibles)
            angulos = [ d['ABC'],d['DBC'],d['CDA'],d['BDA'] ]  
            correcto = True
            for a in angulos:
                if (a < self.ANG_MIN) or (self.ANG_MAX < a):
                    correcto = False
                    break
            distAB = d['distAB']  
            distBD = d['distBD']
            distDC = d['distDC']
            distCA = d['distCA']
            if distAB < 400:
                correcto = False
            if distCA < 400:
                correcto = False
            if distBD < 400:
                correcto = False
            if distDC < 400:
                correcto = False
            if correcto:
                cercano = 10000
                for k in kp:
                    if self.__distancia(k.pt,posibles[3]) < cercano:
                        cercano = self.__distancia(k.pt,posibles[3])
                dist[cercano] = idx
    
        for d,valid in dist.items():
            if valid==0:
                t1[3] = p[valid]
                d = self.__analizar_distancia(t1)
                distCA = d['distCA']
                distBD = d['distBD']
                distAB = d['distAB']
                distDC = d['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return t1
            elif valid==1:
                t2[3] = p[valid]
                d = self.__analizar_distancia(t2)
                distCA = d['distCA']
                distBD = d['distBD']
                distAB = d['distAB']
                distDC = d['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return t2
            elif valid==2:
                t3[3] = p[valid]
                d = self.__analizar_distancia(t3)
                distCA = d['distCA']
                distBD = d['distBD']
                distAB = d['distAB']
                distDC = d['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return t3
            else:
                t4[3] = p[valid]
                d = self.__analizar_distancia(t4)
                distCA = d['distCA']
                distBD = d['distBD']
                distAB = d['distAB']
                distDC = d['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return t4
        
    def __angulos(self,a,b,c):
        """Se pasan tres puntos y se selecciona 'a' devolviendo alfa, el ángulo entre AB y AC.

        Args:
            a ([(int,int)]): Punto A. Donde se encuentra el angulo.
            b ([(int,int)]): Punto B.
            c ([(int,int)]): Punto C.

        Returns:
            [float]: Grados del angulo alfa, entre AB y AC.
        """
        AB = (b[0]-a[0],b[1]-a[1])
        AC = (c[0]-a[0],c[1]-a[1])
        escalar = AB[0]*AC[0]+AB[1]*AC[1]
        modAB = math.sqrt((AB[0]**2) + (AB[1]**2))
        modAC = math.sqrt((AC[0]**2) + (AC[1]**2))
        alfa = math.acos(escalar/(modAB*modAC))
        return math.degrees(alfa)

    def __folio_valido(self,archivo,cuatro,kp):
        """[summary]

        Args:
            archivo ([str]): Nombre del archivo.
            cuatro ([[(int,int)]]): Lista con las cuatro posibles esquinas del folio.
            kp ([[Keypoint]]): Lista de keypoints de la imagen.

        Returns:
            [(bool,[(int,int)])]: True o False si el posible folio encontrado es valido y los
            cuatro puntos correspondientes con las esquinas.
        """
        c = cuatro
        found = False
        a = [(c.pt[0],c.pt[1]) for c in cuatro]

        # 1) Si el folio tiene solo tres puntos: regla del paralelo gramo
        if len(c)==3:
            #que punto falta
            #los tres puntos que tenemos formarán un triángulo rectángulo
            p1 = c[0].pt
            p2 = c[1].pt
            p3 = c[2].pt
            a = self.__devolver_cuatro(p1,p2,p3)
        elif len(c) < 3:
            print("Error: se han encontrado menos de dos esquinas en ",archivo)
            return False,cuatro

        # Analizar el folio: 
        # A = menos y, menos x;
        # B = menos y, más x; 
        # C = más y, menos x; 
        # D = más y, más x; 
        esq = self.__analizar_distancia(a)
        distAB = esq['distAB']
        distBD = esq['distBD']
        distDC = esq['distDC']
        distCA = esq['distCA']

        # 2) Si los lados paralelos son parecidos en longitud se considera encontrado el folio
        if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
            found = True
        
        print(archivo,": ",found)
        
        # 3) Si no se encuentra el folio se busca, de los 4, el punto a discriminar: 
        # 3.1) Por cada 3 de los 4 puntos se genera un nuevo cuarto punto: regla de paralelogramo
        # 3.2) Se valora los tres puntos y el cuarto nuevo: los cuatro puntos con el menor perímetro
        # 3.3) Se busca el kp más cercano al nuevo cuarto  
        if not found:
            a = self.__discriminar_punto(a,kp)
            cercano = (0,0)
            nuevo = a[3]
            for k in kp:
                if cercano==(0,0):
                    cercano = k.pt
                else:
                    if self.__distancia(nuevo,k.pt) < self.__distancia(nuevo,cercano):
                        cercano = k.pt
            a[3] = cercano

        return True,a