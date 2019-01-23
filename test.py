import tensorflow as tf
from keras import Model
from keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import keras.backend as K


def recall_score(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


image = Image.open('5.jpg')
image = image.resize((224, 224))
plt.imshow(image)
plt.show()
image_arr = (np.array(image)[None, ...]) / (255.0)

# print(image_arr)

model = load_model('saved_models/new_saved_model.h5',
                   custom_objects={'recall_score': recall_score,
                                   'precision_score': precision_score})

print(np.shape(image_arr))


# image_arr = image_arr.reshape([224, 224, 3])

pred = model.predict(image_arr)[0]

args_pred = np.argsort(-pred)

print(-1*np.sort(-pred))

print(args_pred[:3])




