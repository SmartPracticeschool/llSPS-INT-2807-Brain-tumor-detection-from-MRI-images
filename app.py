from __future__ import division, print_function

import os


import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, url_for,jsonify, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

global model,graph
import tensorflow as tf
graph = tf.get_default_graph()


app = Flask(__name__)

MODEL_PATH = 'body.h5'

model = load_model(r'C:\Users\Sarthak\Desktop\final\body.h5')



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            preds = model.predict_classes(x)
        index = ["The paitent is Normal","The paitent has Tumor"]
        text = "prediction : "+ str(index[preds[0]])
        return (text)
    


if __name__ == '__main__':
    app.run(debug=True,threaded = False)