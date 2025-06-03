from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
model = tf.keras.models.load_model('plant_classification_model (3).h5')

# Class names - replace with your actual class names
class_names = ['Healthy', 'Diseased']  # Example - update with your classes

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")
    
    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_names[int(np.round(prediction)[0][0])]
            confidence = float(np.max(prediction))
            
            return render_template('index.html', 
                                prediction=predicted_class,
                                confidence=confidence,
                                image_path=filename)
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")
    
    return render_template('index.html', prediction="Invalid file type")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)