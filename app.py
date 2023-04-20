from flask import Flask, render_template ,request
import joblib
import os
import numpy as np
from waitress import serve

from PIL import Image
from numpy import asarray
from sklearn.model_selection import train_test_split


# app = Flask(__name__)
def create_app():
    app = Flask(__name__)
    # add configuration, blueprints, etc. to app
    return app

app = create_app()  # create the Flask app object



@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html' ,message= None)

@app.route('/',methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    im = Image.open(image_path)
    im=im.convert('L')
    newsize = (128, 128)
    im = im.resize(newsize)

    numpydata = asarray(im)

    newarr = np.resize(numpydata,(1,16384))

    # Load the trained SVM model
    model = joblib.load('model.pkl')
    yhat = model.predict(newarr)
    print(yhat)

    classification = yhat[0]
    print(classification)
    os.remove(image_path)
    message=""
    if(classification):
        message="gram positive"
    else:
        message="gram negative"

    return render_template('index.html',prediction=message)
    


if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=True)
    serve(app, host='0.0.0.0', port=8080, threads=4)
