
import numpy as np 
np.random.seed(42)
import random 
import os
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='./model/model_vgg16_rmsprop0.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    ## Scaling
    # x=x/255
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)[0]
    # preds=np.argmax(preds, axis=1)
    
    if preds>0.5:
        preds="Dog"
    else:
        preds="Cat"

    return str(preds)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request
    f = request.files['file']
    # Save the file to ./uploads
    if not os.path.isdir("uploads"):os.makedirs("uploads")
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    try:
        result = model_predict(file_path, model)# Make prediction
        os.remove(file_path)
    except:
        result="Error"
        os.remove(file_path)
    return result


if __name__ == '__main__':
    app.run(debug=True)
