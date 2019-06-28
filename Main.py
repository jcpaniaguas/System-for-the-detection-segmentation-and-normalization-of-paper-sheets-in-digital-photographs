# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 16:52:20 2018

@author: jcpaniaguas

Desc: estudio de las técnicas posibles para encontrar el folio en la imagen (ORB,findContours,Hough)
y elección de ORB como mejor opción
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

direc1 = "../Folios/Internet/"
direc2 = "../Folios/Fotos/"
direc3 = "../dirFotos/unPapel/fondoParecido/"
direc4 = "../dirFotos/unPapel/fondoDiferente/camaraMovil/cerca/"

#cargar imagenes
def cargar_fotos(d):
    archivos = os.listdir(d)
    for path in archivos:
        print(path)
        folio_gris = cv2.imread(d+path, 0)
        folio_color = cv2.imread(d+path)
        
        #"""
        ret,thresh = cv2.threshold(folio_gris,127,255,cv2.THRESH_BINARY)
        #thresh = cv2.resize(thresh, (960,540))
        cv2.imshow("Thresh",thresh)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
        cv2.drawContours(folio_color,contours,-1,(0,0,255),3)    
        folio_color = cv2.resize(folio_color, (960,540))
        cv2.imshow(path,folio_color)
        
        cv2.waitKey()
        cv2.destroyAllWindows()
        #break
        #"""

def puntos_de_interes(d):        
    archivos = os.listdir(d)
    for path in archivos:
        print(path)
        folio_gris = cv2.imread(d+path, 0)
        folio_color = cv2.imread(d+path)
        folio_color = cv2.resize(folio_color, (960,540))
        folio_gris = cv2.resize(folio_gris, (960,540))
        
        cota_max = 200
        cota_min = 130
        punt = cota_max
        
        desc = []
        idx = 0
        
        parecidos = []
        
        while ((cota_min<=punt)&(punt<=cota_max)):
            ret,thresh = cv2.threshold(folio_gris.copy(),punt,255,cv2.THRESH_BINARY)
            folio_nueva = cv2.resize(thresh, (960,540))
            cv2.imshow("1", folio_nueva)
            cv2.waitKey()
            cv2.destroyAllWindows()
            punt = punt - 10
            orb = cv2.ORB_create(20, 1.3, 4)
            kp = orb.detect(folio_nueva,None)
            kp, des = orb.compute(folio_nueva, kp)
            ##print(kp[0])
            
            for k in kp:
                print("PT: ",(round(k.pt[0]),round(k.pt[1])))
                
            desc.append(des)
            actual = desc[idx]
            print("En ",idx," el descriptor es ",actual," con tamaño ",len(actual))
            idx += 1
            for i, k in enumerate(kp):
                print("PT: ",(round(k.pt[0]),round(k.pt[1])))
                folio_color = cv2.drawKeypoints(folio_nueva,[kp[i]],folio_color,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow(path, folio_color)
                cv2.waitKey()
                cv2.destroyAllWindows()
            
            
            if not parecidos:
                parecidos.append(descriptores_parecidos(des))
            else:
                p = descriptores_parecidos(des)
                parecidos = concatenar(parecidos,p)
                
            print("Parecidos: ",parecidos)
            
            """
            print("Parecidos")
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des,des)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)

            # Draw first 10 matches.
            img3 = thresh.copy()
            img3 = cv2.drawMatches(thresh,kp,thresh,kp,matches[:10],img3,flags=2)

            plt.imshow(img3),plt.show()
            """
            if idx==4:
                break
        punt = cota_max
        break 

#si sombreado es true, se buscan los bordes con canny tras el threshold, si no no se aplica threshold
def hough(d,sombreado):
    archivos = os.listdir(d)
    for path in archivos:
        print(path)
        img = cv2.imread(d+path,0)
        img = cv2.resize(img, (960,540))
        #cv2.imshow("Original",img)
        ###
        ret,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        thresh = cv2.resize(thresh, (960,540))
        cv2.imshow("Thresh",thresh)
        cv2.waitKey()
        cv2.destroyAllWindows()
        ###
        print("Canny")
        if sombreado:
            edges = cv2.Canny(thresh,50,200)
        else:
            edges = cv2.Canny(img,50,200)
        plt.imshow(edges)
        lines = cv2.HoughLines(edges,1,np.pi/180,150)
        color = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        print(lines)
        if lines is not None:
            for line in lines:
                rho,theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)
            plt.figure(figsize=(15,15))
            color = cv2.resize(color, (960,540))
            plt.imshow(color)
            plt.show()
        #break

def descriptores_parecidos(des):
    #print("Dentro: ",des)
    solucion = []
    for i, d in enumerate(des):
        #print("D1: ",d)
        if(i != len(des)-1):
            for j, d2 in enumerate(des[i+1:]):
                #print("DEMAS: ",d2)
                p = arrays_parecidos(d,d2)
                #print(p)
                if(p >= 18):
                    if not esta_en_el_array(solucion,d):
                        solucion.append(d)
                    if not esta_en_el_array(solucion,d2):
                        solucion.append(d2)
    return solucion
            
    
def arrays_parecidos(d,d2):
    aciertos = d==d2
    num = np.sum(aciertos)
    return num   

def esta_en_el_array(s,a):
    for s1 in s:
        sol = s1==a
        if(np.sum(sol)==len(s1)):
            return True
    return False

#se pasa dos array de arrays "s" y "a". Si algún array de "a" no está en "s", se añade
def concatenar(s,a):
    for a2 in a:
        if not esta_en_el_array(s,a2):
            #l = len(s)
            s[0].append(a2)
    return s

puntos_de_interes(direc4)        
#cargar_fotos(direc4)
#hough(direc4,False)
