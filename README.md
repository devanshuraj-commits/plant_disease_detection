# Plant Disease Detection using Deep Learning

A Flask web application that classifies plant leaves into three health categories (**Healthy**, **Powdery Mildew**, and **Rust**) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Features
- **Deep Learning Model**: Custom CNN model trained on plant leaf images.
- **Web UI**: Interactive interface allowing users to preview and classify leaf images.
- **RESTful Prediction API**: `/predict` endpoint for multipart image uploads.
- **Render Ready**: Includes `render.yaml` blueprint for one-click cloud deployment.

## Classes
- `Healthy` (Healthy Leaf)
- `Powdery` (Powdery Mildew)
- `Rust` (Leaf Rust)

## Quick Start (Local)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chando0185/plant_disease_detection.git
   cd plant_disease_detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   Navigate to `http://127.0.0.1:5000/`.

## Deploying to Render

This repository is configured for deployment on [Render](https://render.com).

### Option 1: Blueprint Deployment (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** -> **Blueprint**.
3. Connect this repository (`Chando0185/plant_disease_detection`).
4. Render will read `render.yaml` and set up the Web Service automatically.
5. Click **Apply**.

### Option 2: Manual Web Service
1. Click **New +** -> **Web Service**.
2. Connect your GitHub repository.
3. Set the following configuration:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --threads 2 --timeout 120`
4. Add Environment Variables:
   - `PYTHON_VERSION`: `3.11.9`
   - `TF_ENABLE_ONEDNN_OPTS`: `0`
5. Click **Create Web Service**.
