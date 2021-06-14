# @author: jcpaniaguas
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import sys
import os
import urllib.request
import cv2
import tempfile
from werkzeug.utils import secure_filename
sys.path.insert(1,'./src')
from SheetLocator import SheetLocator as locator

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'tmp/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = "secret-key"
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT,UPLOAD_FOLDER)

locator = locator()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('home.html')

@app.route('/',methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # delete old images
        images = os.listdir(app.config['UPLOAD_FOLDER'])
        for delete_image in images:
            delete_path = os.path.join(app.config['UPLOAD_FOLDER'],delete_image)
            if os.path.isfile(delete_path) or os.path.islink(delete_path):
                os.unlink(delete_path)
                print('Image delete:',delete_path)
        # save original image
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # load original image
        image_name = app.config['UPLOAD_FOLDER']+filename
        img = cv2.imread(image_name,0)
        # locate sheet
        sheet = locator.locate(filename,img)
        # save corner image and sheet
        corner_name = app.config['UPLOAD_FOLDER']+'corner_'+filename
        sheet_name = app.config['UPLOAD_FOLDER']+'sheet_'+filename
        cv2.imwrite(corner_name,sheet.get_corner_image())
        cv2.imwrite(sheet_name,sheet.get_sheet())
        flash('Image successfully uploaded and displayed below')
        return render_template('home.html',image_name=filename,corner_image='corner_'+filename,sheet='sheet_'+filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return redirect(request.url)

@app.route('/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)