# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:43:56 2019

@author: jcpaniaguas
"""

"""
Clase Entrenador:
La clase entrenador se encargara de analizar las imagenes de entrenamiento y
crear una base de datos con los descriptores que correspondan con las esquinas
de esta muestra en sus diferentes umbralizados
"""
class Entrenador: 
    
    """
    Constructor:
    Al constructor se le pasa el directorio donde estaran las imagenes de entrenamiento
    """    
    def __init__(self, directorio):
        self.dir = directorio
        
        
    """
    Metodo auxiliar mostrar_img(self,path,img):        
    Muestra la imagen img con cv2 
    """
    def mostrar_img(self,path,img):
        import cv2
        cv2.imshow(path,img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    
    """
    Metodo principal entrenar:
    Encontrar las esquinas de los folios a traves de los puntos de interes y
    con distintas etapas de umbralizado. Despues almacenarlos en con pickle para
    su posterior uso
    """
    def entrenar(self):
        import os
        import cv2
        import pickle 
        archivos = os.listdir(self.dir)
        
        des_parecidos = []
        kp_parecidos = []
        
        for i,path in enumerate(archivos):
            print(path)
            
            color = cv2.imread(self.dir+path)
            gris = cv2.imread(self.dir+path, 0)
        
            color = cv2.resize(color, (960,540))
            gris = cv2.resize(gris, (960,540))
            
            cota_max = 200
            cota_min = 130
            punt = cota_max
            
            orb = cv2.ORB_create(20, 1.3, 4)
                
            while ((cota_min<=punt)and(punt<=cota_max)):
                
                ret,thresh = cv2.threshold(gris.copy(),punt,255,cv2.THRESH_BINARY)
                thresh = cv2.resize(thresh, (960,540))
                t = thresh.copy()
                
                punt = punt - 10
                kp = orb.detect(t,None)
                kp, des = orb.compute(t,kp)
                
                todos = cv2.drawKeypoints(thresh,kp,color.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                self.mostrar_img(path,todos)
                
                if input()=="s":
                    for idx,k in enumerate(kp):
                        uno = cv2.drawKeypoints(thresh,[k],color.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        self.mostrar_img(path,uno)
                        if input()=="s":
                            des_parecidos.append(des[idx])
                            kp_parecidos.append(k)
                            print("KPs introducidos: ",len(des_parecidos))
                
                print("Parecidos: ",des_parecidos)
                print("Kp de parecidos", kp_parecidos)
                print(len(des_parecidos)," ",len(kp_parecidos))
                
                parecidos = cv2.drawKeypoints(thresh,kp_parecidos,color.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                self.mostrar_img("Parecidos",parecidos)
        
        file = open("entrenamiento_2.pkl","wb")
        pickle.dump(des_parecidos,file)    
        file.close()
        
    def para_match(self):
        import os
        import cv2
        import pickle
        import numpy as np
        
        archivos = os.listdir(self.dir)
        
        des_parecidos = []
        kp_parecidos = []
        kp_parecidos_ser = []
        prueba = []
        diccionario_match = dict()
        
        for i,path in enumerate(archivos):
            print(path)
            
            directorio = self.dir+path
            
            color = cv2.imread(directorio)
            gris = cv2.imread(directorio,0)
        
            color = cv2.resize(color, (960,540))
            gris = cv2.resize(gris, (960,540))
            
            cota_max = 200
            cota_min = 130
            punt = cota_max
            
            orb = cv2.ORB_create(20, 1.3, 4)
                
            while ((cota_min<=punt)and(punt<=cota_max)):
                ret,thresh = cv2.threshold(gris.copy(),punt,255,cv2.THRESH_BINARY)
                thresh = cv2.resize(thresh, (960,540))
                t = thresh.copy()
                
                punt = punt - 10
                kp = orb.detect(t,None)
                kp, des = orb.compute(t,kp)
                
                todos = cv2.drawKeypoints(thresh,kp,color.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                self.mostrar_img(path,todos)
        
                
                
                if input()=="s":
                    for idx,k in enumerate(kp):
                        uno = cv2.drawKeypoints(thresh,[k],color.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        self.mostrar_img(path,uno)
                        if input()=="s":
                            kp_parecidos.append(k)
                            lista = [k.pt,k.size,k.angle,k.response,k.octave,k.class_id]
                            kp_parecidos_ser.append(lista)
                            des_parecidos.append(des[idx])
                            prueba.append([k.pt,k.size,k.angle,k.response,k.octave,k.class_id,des[idx]])
                            print("KPs introducidos: ",len(des_parecidos))
                
                print("Parecidos: ",des_parecidos)
                print("Kp de parecidos", kp_parecidos)
                print(len(des_parecidos)," ",len(kp_parecidos))
                parecidos = cv2.drawKeypoints(thresh,kp_parecidos,color.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                self.mostrar_img("Parecidos",parecidos)
                print("Prueba: ",prueba)
            
            
            diccionario_match.update({directorio: prueba})
            print("Dict= ",diccionario_match)
            prueba = []
            des_parecidos = []
            kp_parecidos = []
            kp_parecidos_ser = []
            print("Nuevos kp y des: ",kp_parecidos," ",des_parecidos)
            if i==2:
                break
            
        file = open("para_match.pkl","wb")
        pickle.dump(diccionario_match,file)    
        file.close()
        
    def probar_kp(self):
        import os
        import cv2
        import pickle 
        archivos = os.listdir(self.dir)
        
        diccionario_match = dict()
        
        path = archivos[0]
        
        print(path)
            
        directorio = self.dir+path
        
        color = cv2.imread(directorio)
        gris = cv2.imread(directorio,0)
        
        color = cv2.resize(color, (960,540))
        gris = cv2.resize(gris, (960,540))
            
        orb = cv2.ORB_create(20, 1.3, 4)
            
        ret,thresh = cv2.threshold(gris.copy(),127,255,cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, (960,540))
        t = thresh.copy()
            
        kp = orb.detect(t,None)
        kp, des = orb.compute(t,kp)
        
        point = kp[0]
        keyOfKeys = (point.pt, point.size, point.angle, point.response, point.octave, 
        point.class_id, des)
        
        diccionario_match.update({directorio: keyOfKeys})
        
        file = open("prueba_kp.pkl","wb")
        pickle.dump(diccionario_match,file)    
        file.close()
            
        
path_1 = "../dirFotos/unPapel/fondoDiferente/camaraMovil/cerca/"    
entrenador = Entrenador(path_1)
entrenador.para_match()