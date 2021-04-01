# @author: jcpaniaguas
import cv2
import math
import numpy as np

class DrawingTool:
    """Clase de apoyo para mostrar directamente la imagen
    con los puntos o los keypoints dibujados. 
    """

    @staticmethod
    def dibujar_puntos(puntos, img, archivo,radio=20,grosor=1,color=(255,0,0)):
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
                y)), radius=radio, thickness=grosor, color=color)
        return img_copia

    @staticmethod
    def mostrar_matches(img, archivo, kp, matches):
        puntos = []
        for m in matches:
            puntos.append(kp[m.queryIdx])
        return cv2.drawKeypoints(img, puntos, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
