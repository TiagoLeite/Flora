import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.layers import Dense, Reshape, Dropout
from keras.models import Model
import argparse
import tensorflow as tf
from keras import backend as K
from metrics import Metrics

train_path = '65_flowers'
# test_path = '../data/test'
# valid_path = '../data/valid'

FLAGS = None
CLASSES_NUM = 65


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


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


def load_trained_model(h5_file_path):
    model = load_model(h5_file_path)
    return model


def get_new_model():
    mobile = keras.applications.mobilenet.MobileNet(weights='imagenet')
    # print(mobile.summary())
    x = mobile.layers[FLAGS.layer_to_append].output
    reshaped = Reshape(target_shape=[1024], name='tiago_reshape')(x)
    # intermediate = Dense(512, activation='relu')(reshaped)
    # drop = Dropout(0.5)(intermediate)
    pred = Dense(CLASSES_NUM, activation='softmax')(reshaped)
    model = Model(inputs=mobile.input, outputs=pred)
    return model


def main():
    EPOCHS = FLAGS.EPOCHS
    BATCH_SIZE = FLAGS.BATCH_SIZE
    SAVED_MODEL_PATH = FLAGS.saved_model_path
    # preprocessing_function = keras.applications.mobilenet.preprocess_input

    train_datagen = ImageDataGenerator(preprocessing_function=None,
                                       rescale=1.0/255.0,
                                       rotation_range=180,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.1,
                                       horizontal_flip=True,
                                       zoom_range=[0.9, 1.25],
                                       brightness_range=[0.5, 1.5],
                                       validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(train_path,
                                                  target_size=(224, 224),
                                                  batch_size=BATCH_SIZE,
                                                  subset='training')
    val_gen = train_datagen.flow_from_directory(train_path,
                                                target_size=(224, 224),
                                                batch_size=BATCH_SIZE,
                                                subset='validation')

    # test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
    #    flow_from_directory(test_path, target_size=(224, 224), batch_size=100, shuffle=False)

    # valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
    #    flow_from_directory(valid_path, target_size=(224, 224), batch_size=100)

    if SAVED_MODEL_PATH == 'null':
        model = get_new_model()
    else:
        print('Restoring model from ', SAVED_MODEL_PATH)
        model = load_trained_model(SAVED_MODEL_PATH)

    print(model.summary())

    print('Layers:', len(model.layers))

    #for layer in model.layers[:FLAGS.layer_to_train]:
    #    layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy', recall_score, precision_score])

    label_map = train_gen.class_indices
    print(label_map)

    model.fit_generator(train_gen,
                        steps_per_epoch=train_gen.samples // BATCH_SIZE,
                        validation_data=val_gen,
                        validation_steps=val_gen.samples // BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2)

    model.save('saved_models/new_saved_model.h5')
    #frozen_graph = freeze_session(K.get_session(),
    #                              output_names=[out.op.name for out in model.outputs])
    # tf.train.write_graph(frozen_graph, logdir='saved_models', name="saved_model.pb", as_text=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--layer_to_append',
        type=int,
        default=-4,
    )
    parser.add_argument(
        '--layer_to_train',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--BATCH_SIZE',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--EPOCHS',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--saved_model_path',
        type=str,
        default='null',
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()