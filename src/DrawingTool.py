# @author: jcpaniaguas
import cv2
import math
import numpy as np

class DrawingTool:
    """Clase de apoyo para mostrar directamente la imagen
    con los puntos o los keypoints dibujados. 
    """

    @staticmethod
    def __dibujar_puntos(puntos, img, archivo):
        """Dibuja los puntos en la imagen 

        Args:
            puntos ([dict([(int,int)])]): Diccionario cuya key es el nombre de la imagen y cuyos values son los puntos
            que se consideran esquinas
            img ([numpy.ndarray]): Imagen en la que se van a dibujar los puntos
            archivo ([str]): Nombre de la imagen que se va a mostrar

        Returns:
            [numpy.ndarray]: Imagen con los puntos dibujados
        """
        img_copia = img.copy()
        p = puntos if(type(puntos) == type(list())) else puntos[archivo]
        for x, y in p:
            img_copia = cv2.circle(img_copia, (int(x), int(
                y)), radius=20, thickness=1, color=(0, 0, 255))
        return img_copia

    @staticmethod
    def mostrar_imagen(img, archivo, puntos=[], circulos=True):
        """Muestra la imagen, bien con los puntos dibujados, bien la original

        Args:
            img ([numpy.ndarray]): Imagen que se va a mostrar
            archivo ([str]): Nombre de la imagen que se va a mostrar
            puntos ([dict([(int,int)])]): Diccionario cuya key es el el nombre de la imagen y cuyos values son los puntos que se consideran esquinas
            circulos (bool, optional): Si es true, se van a dibujar circulos; si es false, se imprime la imagen original. Por defecto a True.
        """
        if circulos:
            img = DrawingTool.__dibujar_puntos(puntos, img, archivo)
        cv2.imshow(archivo, img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @staticmethod
    def mostrar_kp(img, archivo, kp):
        """Se dibujan los keypoints en la foto concreta 

        Args:
            img ([numpy.ndarray]): Imagen que se va a mostrar
            archivo ([str]): Nombre de la imagen que se va a mostrar
            kp ([[keypoint]]): Lista con los keypoints encontrados en la imagen actual
        """
        img_kp = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        DrawingTool.mostrar_imagen(img_kp, archivo, None, False)

    @staticmethod
    def mostrar_matches(img, archivo, kp, matches):
        puntos = []
        for m in matches:
            punto = kp[m.queryIdx].pt
            punto = (int(punto[0]),int(punto[1]))
            puntos.append(punto)
        DrawingTool.mostrar_imagen(img, archivo, puntos)

    @staticmethod
    def transformar_perspectiva(img, archivo, desde_esq, hasta_esq, tamaño=(540,960)):
        """Muestra la transformación de una imagen con 
        perspectiva a una en 2D

        Args:
            img ([numpy.ndarray]): Imagen que se va a mostrar
            archivo ([str]): Nombre de la imagen que se va a mostrar
            desde_esq ([numpy.ndarray]): Numpy array con las cuatro esquinas actuales
            hasta_esq ([numpy.ndarray]): Numpy array con las cuatro esquinas a conseguir
        """
        desde_esq = DrawingTool.__ordenar_origen(desde_esq)
        M = cv2.getPerspectiveTransform(desde_esq, hasta_esq)
        transformado = cv2.warpPerspective(img, M, tamaño)
        #cv2.imwrite(archivo+' - transformado.png',transformado)
        cv2.imshow("Transformado",transformado)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def __ordenar_origen(desde_esq):
        """Función que analiza la distancia desde el primer punto A (esquina arriba-izq) al
        punto B (arriba-der) y C (abajo-izq) para que la transformación sea la adecuada.
        En el caso de que el lado más largo fuese el A-B querrá decir que el folio está 
        de lado a la cámara y habría que variar el orden de la tranformación. Que el 
        más largo fuese de A-C implicaría que el folio está de frente a la cámara.

        Args:
            desde_esq ([numpy.ndarray]): Numpy array con las cuatro esquinas actuales
        """
        A = desde_esq[0]
        B = desde_esq[1]
        C = desde_esq[2]
        D = desde_esq[3]

        primer_lado = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
        segundo_lado = math.sqrt((A[0]-C[0])**2 + (A[1]-C[1])**2)
        
        if primer_lado < segundo_lado:
            return desde_esq
        else:
            return np.float32([A, C, B, D])
