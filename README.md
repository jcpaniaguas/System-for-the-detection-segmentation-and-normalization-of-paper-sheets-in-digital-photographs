# System for detection, segmentation and normalization of sheets of paper in digital photographs

In this project we try, given an image of a sheet of paper, to locate and cut out the sheet by locating with a trained model its corners.

The directories are described below according to their use:

* img: directory of images used in the project:
  * kp_circle: all keypoints that have been found in the training images are displayed.
  * match_circle: all matches that have been found in the training images are displayed.
  * redim: original images resized to 1700x1700 and used in training and testing.
  * sol_circle: result with the 4 corners found on the sheets.
* models: the result of training, a model trained with 28 images and tested with 14.
* src: main code of the detector properly commented to select the classes you need to use. 
* web: web application code that makes use of the main classes in the 'src' directory (Python, Flask, HTML, CSS).
