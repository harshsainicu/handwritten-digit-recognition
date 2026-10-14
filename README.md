# Handwritten Digit Recognition using Deep Learning

## Project overview
This beginner-friendly project uses a Convolutional Neural Network (CNN) to classify handwritten digits from 0 to 9. It uses the public MNIST dataset available through TensorFlow/Keras.

## Technologies
- Python
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy and Pillow

## How it works
1. The MNIST dataset is loaded.
2. Pixel values are normalized to the range 0–1.
3. A CNN learns visual patterns from the training images.
4. The model is evaluated on test images.
5. A Streamlit interface lets a user upload a digit image for prediction.

## Run locally
Install Python 3.10 or 3.11, then run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Training downloads the MNIST dataset and creates `mnist_cnn.keras`. The web app requires this trained model file.

## Note
This repository contains the source code. The trained model file is generated when `train_model.py` completes successfully. The first training run needs an internet connection to download MNIST.
