from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

model = load_model("model.h5", compile=False)

for image_file in os.listdir():
    if image_file.endswith(".png"):

        img = Image.open(image_file).convert("L")

        # Make background white and digit black
        img = ImageOps.invert(img)

        # Resize to MNIST size
        img = img.resize((28, 28))

        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img, verbose=0)
        digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"{image_file} --> Predicted Digit: {digit} | Confidence: {confidence:.2f}%")