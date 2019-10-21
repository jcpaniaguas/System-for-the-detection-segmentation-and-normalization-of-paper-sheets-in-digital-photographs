import numpy as np
import cv2
import pickle

"""
Busca las coincidencias entre la imagen definida por el path y los keypoints
entrenados para hallar las esquinas y encuentra las correctas, dibujando los
bordes del papel   
"""
def coincidencias(path1):
    
    """Guardara la puntuacion de cuantos votos tiene cada punto de la imagen
    para ser una esquina"""
    
    puntuaciones = dict()
    kp_match = []
    
    """Cargamos la foto de la cual buscamos los puntos de interes"""
    img1 = cv2.imread(path1,0)
    
    orb = cv2.ORB_create(50, 1.3, 4)
    
    """Cargamos los datos del entrenamiento: keypoints y descriptores de cada
    una de las esquinas de los folios de muestra"""
    
    bd = open("para_match.pkl",'rb')
    diccionario_match = pickle.load(bd)
    bd.close()
    
    cota_max = 200
    cota_min = 130
    punt = cota_max
                
    while ((cota_min<=punt)and(punt<=cota_max)):
        
        ret,thresh = cv2.threshold(img1.copy(),punt,255,cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, (960,540))
        kp = orb.detect(thresh,None)
        kp1, des1 = orb.compute(thresh,kp)
        punt = punt - 10
        
        """Se descomponen los datos de entrenamiento: es un diccionario cuya claves es
        el path de la fotografia del folio y cuyo valor es una lista de keypoints
        y su respectivo descriptores. Los keypoints se descomponen por necesidades
        al utilizar la libreria pickle, que no los acepta"""
        
        for path2,v in diccionario_match.items():
            kp_comparar = []
            des_comparar = []
            #desempaquetar 
            for e in v:
                temp_kp = cv2.KeyPoint(x=e[0][0],y=e[0][1],_size=e[1], _angle=e[2], _response=e[3], _octave=e[4], _class_id=e[5]) 
                temp_des = e[6]
                kp_comparar.append(temp_kp)
                des_comparar.append(temp_des)
        
            #para la comparacion se transforma el array de descriptores en un numpy array
            des_comparar = np.array(des_comparar)
            #busqueda de coincidencias con BruteForceMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1,des_comparar)
            matches = sorted(matches, key = lambda x:x.distance)
            
            ###dibujar kps correspondientes
            
            #imprimir matches
            for idx,match in enumerate(matches):
                #keypoint
                posible_esq = kp1[match.queryIdx]
                #coordenadas del punto
                posible_esq_x = "{0:.2f}".format(round(posible_esq.pt[0],2))
                posible_esq_y = "{0:.2f}".format(round(posible_esq.pt[1],2))
                coord = (float(posible_esq_x),float(posible_esq_y))
                kp_match.append(posible_esq)
                if coord in puntuaciones:
                    contador = puntuaciones.get(coord)
                    contador+= 1
                    puntuaciones.update({ coord : contador})
                else:
                    puntuaciones.update({ coord : 1})
                
            img_match_keypoints = thresh.copy()
            img_match_keypoints = cv2.drawKeypoints(thresh,kp_match,thresh.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            print("Diccionario de puntuaciones=",puntuaciones)
            cv2.imshow("Foto con los aciertos de esquina",img_match_keypoints)
            cv2.waitKey()
            cv2.destroyAllWindows()    
            
            ###
            
            """
            #cargamos la actual imagen de entrenamiento
            img2 = cv2.imread(path2,0)
            ret,img2 = cv2.threshold(img2,170,255,cv2.THRESH_BINARY)
            img2 = cv2.resize(img2, (960,540))
            
            dibuja1 = cv2.drawKeypoints(thresh,kp,thresh.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Foto en busqueda de coincidencias",dibuja1)
            cv2.waitKey()
            cv2.destroyAllWindows()
            ###
            dibuja2 = cv2.drawKeypoints(img2,kp_comparar,img2.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Foto de entrenamiento",dibuja2)
            cv2.waitKey()
            cv2.destroyAllWindows()
            ###
            img3 = cv2.drawMatches(thresh,kp,img2,kp_comparar,matches,None,flags=2)
            img3 = cv2.resize(img3, (960,540))
            ###
            cv2.imshow("Coincidencias",img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
            """
    ###cribar kp_match
    print("Kp_match pre-criba=",len(kp_match))
    pre = cv2.drawKeypoints(cv2.resize(img1,(960,540)),kp_match,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Kp_match pre-criba",pre)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    for idx,kp_actual in enumerate(kp_match):
        (x_actual,y_actual) = kp_actual.pt
        for kp_siguiente in kp_match[idx+1:]:
            (x_siguiente,y_siguiente) = kp_siguiente.pt
            if (abs(x_actual - x_siguiente) < 4)&(abs(y_actual - y_siguiente) < 4):
                kp_match.remove(kp_siguiente)
            
    """
    print("Kp_match post-criba=",len(kp_match))
    for f in kp_match:
        print("F ",f.pt)        
    """
    post = cv2.drawKeypoints(cv2.resize(img1,(960,540)),kp_match,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Kp_match post-criba",post)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    ###dibujar lineas correctas
    lineas = cv2.resize(img1.copy(), (960,540))
    for idx,inicio in enumerate(kp_match): 
        for otros in kp_match[idx+1:]:
            pt_ini = (int(inicio.pt[0]),int(inicio.pt[1]))
            pt_fin = (int(otros.pt[0]),int(otros.pt[1]))
            if es_linea_valida(cv2.resize(img1, (960,540)),pt_ini,pt_fin):
                lineas = cv2.line(lineas,pt_ini,pt_fin,(0, 255, 0),2)
            
            
    cv2.imshow("Lineas",lineas)
    cv2.waitKey()
    cv2.destroyAllWindows()        

def es_linea_valida(img,ini,fin):
    x=0
    y=0
    w=0
    h=0
    if(ini[0]<fin[0]):
        if ini[0]-5 >= 0:
            x=ini[0]-5
        else:
            x=ini[0]
    else:
        if fin[0]-5 >= 0:
            x=fin[0]-5
        else:
            x=fin[0]
    
    if(ini[1]<fin[1]): 
        if ini[1]-5 >= 0:
            y=ini[1]-5
        else:
            y=ini[1]
    else: 
        if fin[1]-5 >= 0:
            y=fin[1]-5
        else:
            y=fin[1]

    w = abs(fin[0]-ini[0])+10
    h = abs(fin[1]-ini[1])+10

    recorte = img[y:(y+h), x:(x+w)]
    comparar = recorte.copy()
    cv2.line(comparar,ini,fin,(0,0,255),2)     
    
    solucion = comparar-recorte
    
    ### si el resultado tiene blanco o es muy pequeño no es valido
    tamano = len(solucion)*len(solucion[0]) 
    negro = True
    for fil in solucion:
        for col in solucion:
            if not np.all(solucion==0):
                negro = False
                break
    
    return negro & (tamano > 200)


coincidencias('../Muestra/IMG_7158.JPG')