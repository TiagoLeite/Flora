import tensorflow as tf
from keras import Model
from keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

image = Image.open('5.jpg')
image = image.resize((224, 224), Image.ANTIALIAS)
plt.imshow(image)
plt.show()
image_arr = (np.array(image)[None, ...])/255.0

print(image_arr)

model = load_model('saved_models/saved_model_2.h5')

print(np.shape(image_arr))

# image_arr = image_arr.reshape([224, 224, 3])

pred = model.predict(image_arr)

print(pred[0])
print(np.argmax(pred[0]))


