# @author: jcpaniaguas


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
        import cv2
        img_copia = img.copy()
        p = puntos if(type(puntos) == type(list())) else puntos[archivo]
        for x, y in p:
            img_copia = cv2.circle(img_copia, (int(x), int(
                y)), radius=10, thickness=25, color=(0, 0, 255))
        return img_copia

    @staticmethod
    def mostrar_imagen(img, archivo, puntos, circulos=True):
        """Muestra la imagen, bien con los puntos dibujados, bien la original

        Args:
            img ([numpy.ndarray]): Imagen que se va a mostrar
            archivo ([str]): Nombre de la imagen que se va a mostrar
            puntos ([dict([(int,int)])]): Diccionario cuya key es el el nombre de la imagen y cuyos values son los puntos que se consideran esquinas
            circulos (bool, optional): Si es true, se van a dibujar circulos; si es false, se imprime la imagen original. Por defecto a True.
        """
        import cv2
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
        import cv2
        img_kp = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        DrawingTool.mostrar_imagen(img_kp, archivo, None, False)

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
        import cv2
        M = cv2.getPerspectiveTransform(desde_esq, hasta_esq)
        transformado = cv2.warpPerspective(img, M, tamaño)
        cv2.imshow("Transformado",transformado)
        cv2.waitKey()
        cv2.destroyAllWindows()