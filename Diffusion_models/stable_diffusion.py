# To run this file, please install the exact same versions of the libraries
# pip install tensorflow==2.15.0 keras==2.15.0 keras-cv==0.6.0 tensorflow-datasets==4.9.6 keras-core==0.1.7

import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

images = model.text_to_image("Lion as the king of the forest", batch_size=3)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)