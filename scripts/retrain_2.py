import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Dropout
from keras.models import Model
import argparse
import tensorflow as tf
from keras import backend as K

train_path = '../65_flowers'
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


def main():

    EPOCHS = FLAGS.EPOCHS
    BATCH_SIZE = FLAGS.BATCH_SIZE

    train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                       rescale=1.0 / 255.0,
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

    mobile = keras.applications.mobilenet.MobileNet(weights='imagenet')
    # print(mobile.summary())
    x = mobile.layers[FLAGS.layer_to_append].output
    reshaped = Reshape(target_shape=[1024], name='tiago_reshape')(x)
    #intermediate = Dense(512, activation='relu')(reshaped)
    #drop = Dropout(0.5)(intermediate)
    pred = Dense(CLASSES_NUM, activation='softmax')(reshaped)
    model = Model(inputs=mobile.input, outputs=pred)

    print(model.summary())

    print('Layers:', len(model.layers))

    for layer in model.layers[:FLAGS.layer_to_train]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(train_gen,
                        steps_per_epoch=train_gen.samples//BATCH_SIZE,
                        validation_data=val_gen,
                        validation_steps=val_gen.samples//BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=2)

    model.save('saved_model.h5')
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, logdir='tf_files', name="saved_model.pb", as_text=False)


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
        default=64,
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
