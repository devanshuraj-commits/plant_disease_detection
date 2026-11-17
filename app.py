import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Reassemble model if missing (e.g. on fresh clone or cloud deploy)
if not os.path.exists(MODEL_PATH):
    import glob
    parts = sorted(glob.glob(os.path.join(BASE_DIR, 'model.h5.part*')))
    if parts:
        print(f'Reassembling model from {len(parts)} parts...')
        with open(MODEL_PATH, 'wb') as outfile:
            for part in parts:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())

model = load_model(MODEL_PATH)
print('Model loaded successfully.')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return redirect(url_for('index'))

    if 'file' not in request.files:
        return 'No file part in request', 400

    f = request.files['file']

    if f.filename == '':
        return 'No file selected', 400

    if not allowed_file(f.filename):
        return 'Invalid file type. Please upload a PNG, JPG, or JPEG image.', 400

    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
    f.save(file_path)

    try:
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    except Exception as e:
        return f'Error during prediction: {str(e)}', 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
