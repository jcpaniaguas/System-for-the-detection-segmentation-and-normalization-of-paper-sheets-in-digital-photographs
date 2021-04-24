# @author: jcpaniaguas
from ParallelogramTool import ParallelogramTool as plg
from SheetLocator import SheetLocator
import cv2
import csv
import pickle
import os
import numpy as np
import math


class Trainer:
    """Class that needs a directory of training images and groundtruth file with
    the real corners of each of the images (resized to 1700x1700).
    The main method of the class is 'train' which will start the training and
    create a model with the valid keypoints and descriptors of each image.
    """
    
    ORB_EDGE_THRESHOLD = 7
    ORB_SCALE_FACTOR = 1.5
    ORB_NLEVELS = 15
    RESIZE = (1700,1700)


    def __init__(self, directory, groundtruth, range_number=-1):
        """Trainer class constructor.

        Args:
            directory ([str]): Image directory or list of images.
            groundtruth ([str]): .csv file with the corners of each image.
            rango (int, optional): Number of images in the directory to
            be trained. Default is -1, i.e. all images in the directory are trained.
        """
        self.directory = directory
        self.range_number = range_number
        self.groundtruth = groundtruth
        self.training = None

    def train(self):
        """Main function of the training. It will create a training model with the
        four keypoints and descriptors corresponding to the corners of the sheets
        in the training images. The process will be performed with a search for
        the corresponding keypoints after closing of each image.
        
        Returns:
            [SheetLocator]: SheetLocator with trained model. 
        """
        self.orb = cv2.ORB_create(scaleFactor=self.ORB_SCALE_FACTOR, nlevels=self.ORB_NLEVELS, edgeThreshold=self.ORB_EDGE_THRESHOLD)
        if not os.path.isdir(self.directory):
            print("Error: you must enter a directory: ",self.directory)
            return
        models = os.listdir('./models/')
        files = os.listdir(self.directory)
        trained_images = "./img/Trained_Images_"+str(self.range_number)+".txt"
        with open(trained_images,'w') as f:
            for actual in files[:self.range_number]:
                f.write(actual+'\n')
        self.points = self.__cvs_to_dict()
        self.training = self.__training_photos(files)
        return SheetLocator(self.training)
    
    def __cvs_to_dict(self):
        """With a csv as input it returns a dictionary with its values such that
        the first column will be the key and the others the values.

        Returns:
            [dict([(int,int)])]: Dictionary whose key is the name of the image
            and whose values are the points that are considered corners. 
        """
        csv_dict = dict()
        with open(self.groundtruth) as csv_file:
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

    def __training_photos(self,images):
        """All valid keypoints and descriptors found in each of the training images will be obtained.

        Args:
            images ([[str]]]): A list with the names of all the training images.

        Returns:
            [dict]: Dictionary whose key is the name of each image and value a list of
            its keypoints and valid descriptors.
        """
        valid_data = dict()
        self.range_number = len(images) if self.range_number==-1 else self.range_number
        current_position = 0

        while current_position!=self.range_number:
            print(current_position)
            files = images[current_position]
            img = cv2.imread(self.directory + files, 0)
            if not (img.shape == self.RESIZE):
                img = cv2.resize(img,self.RESIZE)
                print("Image resized to ",img.shape)

            i90 = np.rot90(img)
            i180 = np.rot90(img,2)
            i270 = np.rot90(img,3)
            imirror = np.fliplr(img)
            
            kp_correct = self.__find_kp(files,img)
            kp_c90 = self.__find_kp(files,i90,turn_point=90)
            kp_c180 = self.__find_kp(files,i180,turn_point=180)
            kp_c270 = self.__find_kp(files,i270,turn_point=270)
            kp_mirror = self.__find_kp(files,imirror,turn_point=-1)
            
            all_data = []

            for kp in [kp_correct,kp_c90,kp_c180,kp_c270,kp_mirror]:
                values = list(kp.values())
                for kd in values:
                    all_data.append(kd) 

            valid_data.update({files: all_data})
            current_position += 1

        return valid_data

    def __find_kp(self, file_name, img, turn_point=0):
        """In the current image the closing and search of the keypoints
        is applied to obtain as a result the four points corresponding to
        the corners of the sheet.

        Args:
            file_name ([str]): Name of the current image.
            img ([numpy.ndarray]): Original image in grayscale.
            turn_point (int, optional): The point will be rotated as many degrees as the image is rotated.
            Defaults to 0.

        Returns:
            [dict]: Valid keypoints and descriptors.
        """
        kernel = np.ones((35,35),np.uint8)
        dilation = cv2.dilate(img,kernel,iterations = 1)
        closing = cv2.erode(dilation,kernel,iterations = 1)
        (height, width) = closing.shape
        kp_list = []   
        kp = self.orb.detect(closing,None)
        kp_list, des_list = self.orb.compute(closing, kp)
        kp_correct = dict()
        return self.__pruning(file_name,img,turn_point,kp_list,des_list,kp_correct)

    def __pruning(self, file_name, img, turn_point, kp, des, kp_correct):
        """Prune the keypoints returning those closest to the points described in the groundtruth file.

        Args:
            file_name ([str]): Name of the current image.
            img ([numpy.ndarray]): Closing image.
            turn_point ([int]): The point will be rotated as many degrees as the image is rotated.
            kp ([[Keypoint]]): Keypoints obtained from the current image.
            des ([[int]]): Descriptors obtained from the current image.
            kp_correct ([[Keypoint,[int]]]): If the keypoint is valid,
            its corresponding descriptor is also saved.

        Returns:
            [[Keypoint,[int]]]: Valid keypoints and descriptors.
        """
        corners = self.points[file_name]
        news = []
        h,w = img.shape
        for corner in corners:
            news.append(self.__rotate_points(corner[0],corner[1],h,w,turn_point))         

        for i, k in enumerate(kp):
            (kx, ky) = (round(k.pt[0]), round(k.pt[1]))
            for idx,(img_x, img_y) in enumerate(news):
                kp_correct = self.__insert_best(kp_correct,idx,(img_x,img_y),k,des[i])
        return kp_correct

    def __insert_best(self,kp_correct,idx,point,k,d):
        """The best keypoints that are close to the groundtruth points are selected.

        Args:
            kp_correct ([{Keypoint}]): Keypoint dictionary of the best keypoints.
            idx ([int]): Corner number.
            point ([(int,int)]): Point to approach.
            k ([[Keypoint]]): Current Keypoint.
            d ([[int]]): Current descriptor.

        Returns:
            [{Keypoint}]: Keypoint dictionary with the best keypoints.
        """
        if idx in kp_correct.keys():
            current = kp_correct[idx][0].pt
            current_distance = plg.distance(point,(round(current[0]),round(current[1])))
            new_distance = plg.distance(point,(round(k.pt[0]),round(k.pt[1])))
            if new_distance < current_distance:
                kp_correct[idx] = [k,d]
            elif new_distance == current_distance:
                if kp_correct[idx][0].size > k.size:
                    k_old = kp_correct[idx][0].size
                    k_new = k.size
                    kp_correct[idx] = [k,d]
        else:
            kp_correct[idx] = [k,d]
        return kp_correct

    def __rotate_points(self,x,y,h,w,turn_point):
        """Rotate the points.

        Args:
            x ([int]): X coordinate of the original point.
            y ([int]): Y coordinate of the original point.
            h ([int]): Image height.
            w ([int]): Image width.
            turn_point ([int]): The point will be rotated as many degrees as the image is rotated.

        Returns:
            [(int,int)]: Rotated point.
        """
        if turn_point==90:
            return (y,h-x)
        elif turn_point==180:
            return (w-x,h-y)
        elif turn_point==270:
            return (w-y,x)
        elif turn_point==-1:
            return (w-x,y)
        else:
            return (x,y)