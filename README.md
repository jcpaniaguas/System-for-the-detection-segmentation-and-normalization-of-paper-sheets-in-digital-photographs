# System-for-the-detection-segmentation-and-normalization-of-paper-sheets-in-digital-photographs
Computer Vision system that is able to find the corners of a sheet of paper in a photograph. Then, it allows to transform the sheet from 3D to 2D plane.
The process was performed on a set of 28 training images and 14 test images.

The directories correspond to:
* img: directories of images used.
  * kp_circle: all corners found by the algorithm in the training images.
  * match_circle: all matches found between training and test images.
  * redim: all images used in training and testing.
  * sol_circle: images solution with localized corners.
* models: training model with 28 training and 14 test images.
* src: directory with the classes of the main algorithm.
* web: directory with the Flask code of the web interface.
