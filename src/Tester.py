# @author: jcpaniaguas
from SheetLocator import SheetLocator
import cv2
import os


class Tester:
    """
    Class that needs a training model and the directory of the test images 
    from which we are going to try to locate the sheets.
    The function that will initiate the search is locate(), which works either 
    with a directory or with a single image.
    This function will use the SheetLocator class to find the sheet.
    """

    def __init__(self, model_name="", save=0):
        """Tester class constructor.

        Args:
            model_name ([str]): .pkl file with the training performed.
            save (int, optional): If save is 0 no result images are saved. 
            If save is 1 the result image will be created. 
            If save is 2, the image is printed on the screen. Default to 0.
        """
        self.save = save
        self.model_name = model_name

    def locate(self, path):
        """Main function of the class. Attempts to locate the sheet in the image depending on 
        whether you enter a path with the location of an image or an image directory.

        Args:
            path ([str]): Path where the image or directory of images to be analyzed is located.

        Returns:
            [[Sheet]]: List of Sheet objects.
        """
        if os.path.isdir(path):
            print("A directory has been entered: ", path)
            return self.__locate_dir(path)
        elif os.path.isfile(path):
            print("A file has been entered: ", path)
            (path, file_name) = os.path.split(path)
            return [self.__locate_sheet(path, file_name)]
        else:
            print("Error: you must enter a directory or file.")

    def __locate_dir(self, path):
        """When the path indicated in the 'locate' function is a directory,
        it locates each one of the images to work with them and it derives
        them to the function 'locate_sheet' that works with it one by one.

        Args:
            path ([str]): Path where the directory of images to be analyzed is located.

        Returns:
            [[Sheet]]: List of Sheet objects.
        """
        files = os.listdir(path)
        sheets = []
        for idx,file_name in enumerate(files):
            sheet = self.__locate_sheet(path, file_name)
            sheets.append(sheet)
        return sheets

    def set_model_name(self,model_name):
        """Set model name. 

        Args:
            model_name ([str]): Model name of trained model.
        """
        if type(model_name)==str:
            self.model_name = model_name
        else:
            print("Error: 'model_name' must be of type str.")


    def set_locator(self, locator):
        """Set locator.

        Args:
            locator ([SheetLocator]): Locator with trained model.
        """
        if type(locator)==SheetLocator:
            self.locator = locator
        else:
            print("Error: 'locator' must be of type SheetLocator.")


    def __locate_sheet(self, path, file_name):
        """When the path indicated in the 'locate' function corresponds to an image,
        this function is used to fetch the image and search for corners.

        Args:
            path ([str]): Path where the image to be analyzed is located.
            file_name ([str]): Name of the current image.
        
        Returns:
            [[Sheet]]: Sheet object.
        """
        img = cv2.imread(path+'/'+file_name, 0)
        if self.locator:
            return self.locator.locate(file_name,img,self.save)
        elif self.model_name:
            sl = SheetLocator(self.model_name)
            return sl.locate(file_name,img,self.save)
        else:
            print("Error: there must be a 'locator' or 'model_name' attribute.")
            return