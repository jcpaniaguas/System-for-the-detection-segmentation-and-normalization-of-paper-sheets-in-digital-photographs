# @author: jcpaniaguas
from ParallelogramTool import ParallelogramTool as plg
import cv2
import math
import numpy as np

class DrawingTool:
    """Support class with drawing functions. 
    """

    @staticmethod
    def draw_points(points, img, radio=20,thickness=1,color=(255,0,0)):
        """Draw the points on the image.

        Args:
            points ([dict([(int,int)])]): Dictionary whose key is the name of the image and 
            whose values are the points that are considered corners.
            img ([numpy.ndarray]): Image on which the points are to be drawn.

        Returns:
            [numpy.ndarray]: Image with points drawn.
        """
        img_copy = img.copy()
        for x, y in points:
            img_copy = cv2.circle(img_copy,(int(x),int(y)), radius=radio, thickness=thickness, color=color)
        return img_copy

    @staticmethod
    def show_matches(img, kp, matches):
        """Draw the matches on the image.

        Args:
            img ([numpy.ndarray]): Image on which the matches are to be drawn.
            kp ([[Keypoint]]): Keypoints of the new image.
            matches ([[DMatch]]): Matches between trained and current image descriptors.

        Returns:
            [numpy.ndarray]: Image with matches drawn.
        """
        points = []
        for m in matches:
            points.append(kp[m.queryIdx])
        return cv2.drawKeypoints(img, points, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    @staticmethod
    def transform_perspective(img, from_points, to_points, tamaño=(540,960)):
        """Displays the transformation from a perspective image to a 2D image.

        Args:
            img ([numpy.ndarray]): Image to transform the perspective.
            from_points ([numpy.ndarray]): Numpy array with the current four corners.
            to_points ([numpy.ndarray]): Numpy array with the four corners to be obtained.
        
        Returns:
            [numpy.ndarray]: Transformed image.
        """
        from_points = DrawingTool.__sort_origin(from_points)
        M = cv2.getPerspectiveTransform(from_points, to_points)
        return cv2.warpPerspective(img, M, tamaño)

    def __sort_origin(from_points):
        """Function that sorts the points so that A is in the top-left corner,
        B in the top-right corner and C in the bottom-left corner.

        Args:
            from_points ([numpy.ndarray]): Numpy array with the current four corners.

        Returns:
            [numpy.ndarray]: Numpy array with the four ordered corners.
        """
        d = plg.analyze_distance(from_points)
        A = d['A']
        B = d['B']
        C = d['C']
        D = d['D']
        return np.float32([A, B, C, D])
