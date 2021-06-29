# @author: jcpaniaguas
from DrawingTool import DrawingTool as dt
from ParallelogramTool import ParallelogramTool as plg
from matplotlib import pyplot as plt
import numpy as np


class Sheet:
    """Sheet class.
    """

    def __init__(self,name,original,A,B,C,D,size=(2900,2100)):
        """Sheet class initializer.

        Args:
            name ([str]): Image file name.
            original ([numpy.ndarray]): Image where the sheet is located.
            A ([(float,float)]): Corner A of sheet found.
            B ([(float,float)]): Corner B of sheet found
            C ([(float,float)]): Corner C of sheet found
            D ([(float,float)]): Corner D of sheet found
            size (tuple, optional): Size to which the image can be resized. Defaults to (1700,1700).
        """
        self.name = name
        self.original_image = original
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.size = size

    def show_image(self):
        """Show the original image.
        """
        image = self.get_image()
        plt.figure(figsize = (8,8))
        plt.imshow(image,cmap='gray')
        plt.show()
    
    def show_corner_image(self):
        """show the image with the corners drawn.
        """
        image = self.get_image()
        corner_image = dt.draw_points([self.A,self.B,self.C,self.D],image)
        plt.figure(figsize = (8,8))
        plt.imshow(corner_image,cmap='gray')
        plt.show()

    def show_sheet(self):
        """Shows the sheet found in the image. It will be shown with the transformed perspective.
        """
        image = self.get_sheet()
        plt.figure(figsize = (8,8))
        plt.imshow(image,cmap='gray')
        plt.show()

    def get_image(self):
        """Obtain the original image.

        Returns:
            [numpy.ndarray]: Image with the corners drawn.
        """
        return self.original_image

    def get_corner_image(self):
        """Obtain the image with the corners drawn.

        Returns:
            [numpy.ndarray]: Original image.
        """
        image = self.get_image()
        corner_image = dt.draw_points([self.A,self.B,self.C,self.D],image)
        return corner_image

    def get_sheet(self):
        """Get the sheet found in the image. It will be transformed from perspective to 2D.

        Returns:
            [numpy.ndarray]: Found sheet.
        """
        image = self.original_image
        width, height = self.size
        from_c = np.array([self.A,self.B,self.C,self.D])
        to_c = np.float32([[0,0],[width,0],[0,height],[width,height]])
        return dt. transform_perspective(image,from_c,to_c,self.size)