# Sistema para la deteccion, segmentacion y normalizacion de hojas de papel en fotografias digitales

Descripción: El sistema es capaz de, con una fotografía inicial, encontrar un folio y pasar su contenido de tridimensional a bidimensional. 

Para la ejecución:

0- El clase Src/Entrenador.py se utiliza para que el usuario busque los keypoints de la imagen correspondientes con esquinas. Sólo se ejecuta (por ahora) en el directorio dirFotos/unPapel/fondoDiferente/camaraMovil/cerca/ y sólo utiliza las tres primeras fotos para minimizar el problema. El resultado es guardado en Src/para_match.pkl con pickle, por lo que, al estar ya hecho, no es necesario ejecutar el fichero (Src/Entrenador.py).

1- Ejecutar el fichero Src/match.py. Este va a buscar las esquinas que pueden ser bordes de una foto de la carpeta Muestra y encontrar coincidencias con el entrenamiento almacenado en Src/para_match.pkl hasta dibujar los bordes del folio. La imagen utilizada es una de las tres que participan en el entrenamiento.

Mejoras:

1- Diferenciar la muestra para que las imagenes de entrenamiento (dirFotos) y de prueba (Muestra) no sean las mismas
2- Ampliar el algoritmo para que estime donde pueden estar las esquinas en el caso de que encuentre <4 esquinas correctas
3- El algoritmo tarda en dibujar las lineas (a mayor número de kp_postcriba más tarda), ¿puede haber una mejor manera de encontrar las líneas válidas?
