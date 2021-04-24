# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt
from ParallelogramTool import ParallelogramTool as plg
from Sheet import Sheet
from matplotlib import pyplot as plt
import pickle
import cv2
import numpy as np
import math


class SheetLocator:
    """Class to locate a sheet.
    """

    ORB_EDGE_THRESHOLD = 7
    ORB_SCALE_FACTOR = 1.5
    ORB_NLEVELS = 15
    RESIZE = (1700,1700)
    KP_RANGE = 200
    DIST_PAR = 300
    ANG_MAX = 120
    ANG_MIN = 60

    def __init__(self, model="./models/train_28_test_14.pkl"):
        """Class constructor without a trained model.

        Args:
            model (str, optional): Trained model of type dictionary
            or name of the model to be trained.
            Defaults to "./models/train_28_test_14.pkl".
        """
        if type(model)==str:
            self.__load_model(model)
        elif type(model)==dict:
            self.model_name = "./models/train_28_test_14.pkl"
            self.training = model
        else:
            print("Error: the parameter 'model' must be of type string or SheetLocator.")

    def __load_model(self, model):
        """Method that is used by the builder to load a previously trained model.

        Args:
            model ([str]): Trained model to be used.
        """
        self.model_name = model
        current_file = open(model, 'rb')
        self.training = pickle.load(current_file)
        current_file.close()

    def locate(self, file_name, image, save=0):
        """Function that locates the sheet in the given image.

        Args:
            file_name ([str]): Image name.
            image ([numpy.ndarray]): The image in which the sheet is to be located.
            save (int, optional): If save is 0 no result images are saved. 
            If save is 1 the result image will be created. 
            If save is 2, the image is printed on the screen. Default to 0.

        Returns:
            [Sheet]: Sheet object found.
        """
        self.save = save
        self.orb = cv2.ORB_create(scaleFactor=self.ORB_SCALE_FACTOR, nlevels=self.ORB_NLEVELS, edgeThreshold=self.ORB_EDGE_THRESHOLD)
        if not (image.shape == self.RESIZE):
                image = cv2.resize(image,self.RESIZE)
                print("Image resized to ",image.shape)
        self.__unpack()
    
        kernel = np.ones((35,35),np.uint8)
        dilation = cv2.dilate(image,kernel,iterations = 1)
        closing = cv2.erode(dilation,kernel,iterations = 1)

        four = self.__search_coincidences(file_name,closing,image)
        sheet = Sheet(file_name,image,four[0],four[1],four[2],four[3])
        return sheet

    def __unpack(self):
        """Unpack the training values into a list of keypoints and a
        list of descriptors.
        """
        self.kp_trained = []
        self.des_trained = []
        for path, key_des_list in self.training.items():
            for key_des in key_des_list:
                kp = key_des[0]
                des = key_des[1]
                self.kp_trained.append(kp)
                self.des_trained.append(des)

    def __search_coincidences(self,file_name,closing,original):
        """Function that searches for the four corners of the sheet
        based on the matches between the image and the training model.

        Args:
            file_name ([str]): Image name.
            closing ([numpy.ndarray]): Image in which the closure
            has been applied and the matches will be searched. 
            original ([numpy.ndarray]): Original image.

        Returns:
            [[(int,int)]]: Solution list with the coordinates of the four corners.
        """
        (height, width) = closing.shape
        kp_list = []
        kp = self.orb.detect(closing,None)
        kp_list, des_list = self.orb.compute(closing, kp)
        all_image = cv2.drawKeypoints(original, kp_list, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        spl = self.model_name.split('_')
        file_trained = self.__trained_file(file_name,spl[1])
        test_name = spl[1]+"/"+spl[3].split('.')[0]+"_"+file_name
        trained_test_name = spl[1]+"/Trained/"+spl[3].split('.')[0]+"_"+file_name

        if self.save==1:
            name = "./img/kp_circle/"+test_name
            cv2.imwrite(name,all_image)
        elif self.save==2:
            plt.figure(figsize = (8,8))
            plt.imshow(all_image)
            plt.show()

        matches = self.__find_matches(np.array([list(d) for d in des_list]))
        matches_image = dt.show_matches(original,kp_list,matches)

        if self.save==1:
            name = ""
            if file_trained:
                name = "./img/match_circle/"+trained_test_name
            else:
                name = "./img/match_circle/"+test_name
            cv2.imwrite(name,matches_image)
        elif self.save==2:
            plt.figure(figsize = (8,8))
            plt.imshow(matches_image)
            plt.show()
        
        four = self.__four_best(matches,kp_list)
        found,points = self.__valid_sheet(file_name,four,kp_list)

        solution_image = dt.draw_points(points,original,color=(0,0,255))
        if self.save==1:
            name = ""
            if file_trained:
                name = "./img/sol_circle/"+trained_test_name
            else:
                name = "./img/sol_circle/"+test_name
            cv2.imwrite(name,solution_image)
        elif self.save==2:
            plt.figure(figsize = (8,8))
            plt.imshow(solution_image)
            plt.show()

        return points

    def __trained_file(self,file_name,n_test):
        """If the file belongs to those used in the training.

        Args:
            file_name ([str]): Image name.
            n_test ([int]): Number of images. It will match the
            name of the file 'Trained_Images_{n_test}.txt',
            file with the names of the images that have been used
            for the training.

        Returns:
            [bool]: True si el archivo ha formado parte del entrenamiento. False si no.
        """
        path = "./img/Trained_Images_"+str(n_test)+".txt"
        trained = []
        with open(path,'r') as f:
            e = f.read()
            trained = e.split('\n')
        if file_name in trained[:-1]:
            return True
        else:
            return False

    def __find_matches(self, des):
        """The function searches for matches between the current image and the training.

        Args:
            des ([numpy.ndarray]): Image descriptors in which to search for matches.

        Returns:
            [[DMatch]]: List of matches found and sorted by degree of coincidence (distance).
        """
        self.des_trained = np.array(self.des_trained)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, self.des_trained)
        return sorted(matches, key=lambda x: x.distance)
    
    def __four_best(self,matches,kp):
        """Search the matches for the first four different matches.
        It is understood that two matches are distinct when their
        coordinates are more than one KP_RANGE distance apart.

        Args:
            matches ([DMatch]): List of matches found and sorted by degree of coincidence (distance).  
            kp ([[Keypoint]]): Keypoints obtained from the current image.

        Returns:
            [[(int,int)]]: Solution list with the coordinates of the four corners.
        """
        four = {}
        for m in matches:
            if len(four)==4:
                break
            match_point = (round(kp[m.queryIdx].pt[0]),round(kp[m.queryIdx].pt[1]))
            if match_point not in four.keys():
                equal = False
                for c in four.keys():
                    if (abs(c[0]-match_point[0]) <= self.KP_RANGE) & (abs(c[1]-match_point[1]) <= self.KP_RANGE):
                        equal = True
                        break    
                if not equal:
                    four[match_point] = kp[m.queryIdx]
        return list(four.values())

    def __return_four(self,p1,p2,p3):
        """It asks for three points and returns all four using 
        the parallelogram rule.

        Args:
            p1 ([(int,int)]): Point 1.
            p2 ([(int,int)]): Point 2.
            p3 ([(int,int)]): Point 3.

        Returns:
            [[(int,int)]]: List with the four points.
        """
        d12 = plg.distance(p1,p2)
        d23 = plg.distance(p3,p2)
        d31 = plg.distance(p1,p3)
        bigger = [d12,d23,d31]
        bigger.sort(reverse=True)        
        if bigger[0]==d12:
            #in front of p3
            d1,d2 = plg.parallelogram_rule(p1,p3,p2)
        elif bigger[0]==d23:
            #in front of p1
            d1,d2 = plg.parallelogram_rule(p2,p1,p3)
        else:
            #in front of p2
            d1,d2 = plg.parallelogram_rule(p1,p2,p3)
        return [p1,p2,p3,(d1,d2)]
    
    def __discriminate_point(self,point_list,kp):
        """You will analyze the four given points and divide them into four triads such that:
                [A,B,C],
                [A,B,D],
                [B,C,D],
                [A,C,D].
        With these four triads, with the parallelogram rule obtain the fourth point
        and select the triad and the new best point. 

        Args:
            point_list ([[(int)]]): List with the four points.
            kp ([[Keypoint]]): List of keypoints in the image.

        Returns:
            [[(int,int)]]: Tres puntos y el cuarto nuevo en una lista.
        """
        triad_1 = self.__return_four(point_list[0],point_list[1],point_list[2])
        triad_2 = self.__return_four(point_list[0],point_list[1],point_list[3])
        triad_3 = self.__return_four(point_list[0],point_list[2],point_list[3])
        triad_4 = self.__return_four(point_list[1],point_list[2],point_list[3])
        triads = [triad_1,triad_2,triad_3,triad_4]
        dist = {}
        possible = []

        for idx,possible_triad in enumerate(triads):
            possible.append(possible_triad[3])
            analyzed_data = plg.analyze_distance(possible_triad)
            angles = [ analyzed_data['ABC'],analyzed_data['DBC'],analyzed_data['CDA'],analyzed_data['BDA'] ]  
            correct = True
            for angle in angles:
                if (angle < self.ANG_MIN) or (self.ANG_MAX < angle):
                    correct = False
                    break
            distAB = analyzed_data['distAB']  
            distBD = analyzed_data['distBD']
            distDC = analyzed_data['distDC']
            distCA = analyzed_data['distCA']
            if distAB < 400:
                correct = False
            if distCA < 400:
                correct = False
            if distBD < 400:
                correct = False
            if distDC < 400:
                correct = False
            if correct:
                nearby = 10000
                for k in kp:
                    if plg.distance(k.pt,possible_triad[3]) < nearby:
                        nearby = plg.distance(k.pt,possible_triad[3])
                dist[nearby] = idx
    
        for analyzed_data,valid in dist.items():
            if valid==0:
                triad_1[3] = possible[valid]
                analyzed_data = plg.analyze_distance(triad_1)
                distCA = analyzed_data['distCA']
                distBD = analyzed_data['distBD']
                distAB = analyzed_data['distAB']
                distDC = analyzed_data['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return triad_1
            elif valid==1:
                triad_2[3] = possible[valid]
                analyzed_data = plg.analyze_distance(triad_2)
                distCA = analyzed_data['distCA']
                distBD = analyzed_data['distBD']
                distAB = analyzed_data['distAB']
                distDC = analyzed_data['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return triad_2
            elif valid==2:
                triad_3[3] = possible[valid]
                analyzed_data = plg.analyze_distance(triad_3)
                distCA = analyzed_data['distCA']
                distBD = analyzed_data['distBD']
                distAB = analyzed_data['distAB']
                distDC = analyzed_data['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return triad_3
            else:
                triad_4[3] = possible[valid]
                analyzed_data = plg.analyze_distance(triad_4)
                distCA = analyzed_data['distCA']
                distBD = analyzed_data['distBD']
                distAB = analyzed_data['distAB']
                distDC = analyzed_data['distDC']
                if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
                    return triad_4
        
    def __valid_sheet(self,file_name,four,kp):
        """We analyze whether a foil is valid based on its characteristics
        as a parallelogram.

        Args:
            file_name ([str]): Image name.
            four ([[(int,int)]]): List with the four possible corners of the sheet.
            kp ([[Keypoint]]): List of image keypoints.

        Returns:
            [(bool,[(int,int)])]: True or False if the possible foil found is valid
            and the four dots corresponding to the corners.
        """
        found = False
        current_points = [(f.pt[0],f.pt[1]) for f in four]

        # 1) If the sheet has only three points: parallelogram rule
        if len(four)==3:
            #which point is missing
            #the three points we have will form a right triangle
            p1 = four[0].pt
            p2 = four[1].pt
            p3 = four[2].pt
            current_points = self.__return_four(p1,p2,p3)
        elif len(four) < 3:
            print("Error: less than two corners have been found in ",file_name)
            return False,four

        # Analyze the sheet: 
        # A = smaller y, smaller x;
        # B = smaller y, larger x; 
        # C = larger y, smaller x; 
        # D = larger y, larger x; 
        corner = plg.analyze_distance(current_points)
        distAB = corner['distAB']
        distBD = corner['distBD']
        distDC = corner['distDC']
        distCA = corner['distCA']

        # 2) If the parallel sides are similar in length, the sheet is considered to be found.
        if (abs(distCA-distBD) < self.DIST_PAR) & (abs(distAB-distDC) < self.DIST_PAR):
            found = True
        
        print(file_name,": ",found)
        
        # 3) If the folio is not found, the point of the four to be discriminated is searched:
        # 3.1) For each 3 of the 4 points a new fourth point is generated: parallelogram rule.
        # 3.2) The three points and the fourth new point are valued.
        # 3.3) The kp closest to the new room is searched.
        if not found:
            current_points = self.__discriminate_point(current_points,kp)
            nearby = (0,0)
            new_point = current_points[3]
            for k in kp:
                if nearby==(0,0):
                    nearby = k.pt
                else:
                    if plg.distance(new_point,k.pt) < plg.distance(new_point,nearby):
                        nearby = k.pt
            current_points[3] = nearby

        return True,current_points
    
    def save_training(self, model_name):
        """The model can be saved if it has been trained.

        Args:
            model_name ([str]): Name under which the model will be saved.
        """
        if self.training:
            print("Saving model: ",model_name)
            file = open(model_name, "wb")
            pickle.dump(self.training, file)
            file.close()
        else:
            print("Error: no model has been trained.")