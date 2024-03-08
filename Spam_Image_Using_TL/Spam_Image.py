
from flask import Flask,render_template,request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the trained model
model = tf.keras.models.load_model('spam_image_filter_model.h5')

# Function to preprocess uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict whether the image contains spam or not
def predict_spam(image_path,model):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
# Return the prediction value (probability)

# Threshold for considering if it's spam or not (adjust as needed)
    spam_threshold = 0.5

    if prediction >= spam_threshold:
        a= "The uploaded image is predicted as SPAM."
    else:
        a= "The uploaded image is predicted as NOT SPAM."
    return a


app=Flask(__name__)
import os
app.config['UPLOAD_PATH']='static/images'
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload',methods=['POST'])
def upload():
    
    file=request.files['file']
    filename = file.filename
    filepath=os.path.join(app.config['UPLOAD_PATH'],filename)
    file.save(filepath)
    a=predict_spam(filepath,model)
    return render_template('index.html',result=a)
    

if __name__ == '__main__':
    app.run(debug=True,port=5001)