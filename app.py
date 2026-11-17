import os
import gc
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TFLITE = os.path.join(BASE_DIR, 'model.tflite')
MODEL_H5 = os.path.join(BASE_DIR, 'model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Lazy model loader to ensure the home page and health checks load in milliseconds
_interpreter = None
_input_idx = None
_output_idx = None
_keras_model = None


def get_model():
    global _interpreter, _input_idx, _output_idx, _keras_model
    if _interpreter is not None or _keras_model is not None:
        return

    if os.path.exists(MODEL_TFLITE):
        import tensorflow as tf
        _interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
        _interpreter.allocate_tensors()
        _input_idx = _interpreter.get_input_details()[0]['index']
        _output_idx = _interpreter.get_output_details()[0]['index']
        print('TFLite model loaded successfully.')
    else:
        if not os.path.exists(MODEL_H5):
            import glob
            parts = sorted(glob.glob(os.path.join(BASE_DIR, 'model.h5.part*')))
            if parts:
                print(f'Reassembling model from {len(parts)} parts...')
                with open(MODEL_H5, 'wb') as outfile:
                    for part in parts:
                        with open(part, 'rb') as infile:
                            outfile.write(infile.read())
        from tensorflow.keras.models import load_model
        _keras_model = load_model(MODEL_H5)
        print('Keras model loaded successfully.')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getResult(image_path):
    get_model()
    img = Image.open(image_path).convert('RGB').resize((225, 225))
    x = np.array(img, dtype='float32') / 255.0
    x = np.expand_dims(x, axis=0)

    if _interpreter is not None:
        _interpreter.set_tensor(_input_idx, x)
        _interpreter.invoke()
        return _interpreter.get_tensor(_output_idx)[0]
    else:
        return _keras_model(x, training=False).numpy()[0]


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
        if os.path.exists(file_path):
            os.remove(file_path)
        gc.collect()
        return str(predicted_label)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return f'Error during prediction: {str(e)}', 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
