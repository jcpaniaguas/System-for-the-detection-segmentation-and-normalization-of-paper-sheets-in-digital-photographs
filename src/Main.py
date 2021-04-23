# @author: jcpaniaguas
from Trainer import Trainer
from Tester import Tester
import os

def find_sheets(photo_directory,number_of_training_photos,groundtruth,model_name,percentage=False):
    """Function that searches the corners of the testing folios with a given training database.
    If the database has not been created it will be trained and used.

    Args:
        photo_directory ([str]): Directory where you can find the photos of the sheets to be located.
        number_of_training_photos ([int]): Of the total number of photos in the photo_directory, the number of photos 
        to be used for training.
        groundtruth ([str]): Csv file with the corners of the training sheets.
        model_name ([str]): Name of the training model. If the model does not exist, 
        the training will be performed. If it exists, this model will be used.
        percentage (bool, optional): If percentage is False the parameter number_of_training_photos should be
        an integer. If percentage is True the parameter number_of_training_photos should be a percentage (0-100),
        the percentage of files in the photo directory to be used for training. Default to False.
    """
    files = os.listdir(photo_directory)
    range_number = 0
    print("Model name: "+model_name)

    if not percentage:    
        if number_of_training_photos <= len(files):
            tr = str(number_of_training_photos)
            ts = str(len(files)-number_of_training_photos) 
            print("Training: "+tr+", Testing: "+ts)
            range_number = number_of_training_photos
        else:
            print("Error: 'number_of_training_photos' must be less than the number of photos in the selected directory 'photo_directory'.")
            return 0
    else:
        if number_of_training_photos in range_number(101):
            tr = round(len(files)*(number_of_training_photos/100))
            ts = len(files)-tr
            range_number = round(len(files)*(number_of_training_photos/100))
            print("Training: "+str(tr)+", Testing: "+str(ts))
        else:
            print("Error: 'number_of_training_photos' must be a percentage between [0-100].")
        
    trainer = Trainer(photo_directory,groundtruth,range_number)
    trainer.train(model_name)
    tester = Tester("./models/"+model_name,save=1)
    sheets = tester.locate(photo_directory)
    for sheet in sheets:
        sheet.show_sheet()
    return sheets

PHOTO_DIR = "./img/redim/"
photos = len(os.listdir(PHOTO_DIR))
groundtruth = "./img/groundtruth_redim_tipo_0.csv"

for i in [28]:
    find_sheets(photo_directory=PHOTO_DIR,
                    groundtruth=groundtruth,
                    number_of_training_photos=i,
                    model_name="train_"+str(i)+"_test_"+str(photos-i)+".pkl")
