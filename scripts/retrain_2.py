import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape, Dropout
from keras.models import Model
import argparse
import tensorflow as tf
from keras import backend as K

train_path = '../data/train'
test_path = '../data/test'
valid_path = '../data/valid'

FLAGS = None


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
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
    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                       rotation_range=180, width_shift_range=0.2, height_shift_range=0.2,
                                       horizontal_flip=True, zoom_range=[0.9, 1.25], brightness_range=[0.5, 1.5]). \
        flow_from_directory(train_path, target_size=(224, 224), batch_size=100)

    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(test_path, target_size=(224, 224), batch_size=100, shuffle=False)

    valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(valid_path, target_size=(224, 224), batch_size=100)

    mobile = keras.applications.mobilenet.MobileNet(weights='imagenet')

    # print(mobile.summary())

    x = mobile.layers[FLAGS.layer_to_append].output
    reshaped = Reshape(target_shape=[1024], name='tiago_reshape')(x)
    intermediate = Dense(512, activation='relu')(reshaped)
    drop = Dropout(0.5)(intermediate)
    pred = Dense(16, activation='softmax')(drop)
    model = Model(inputs=mobile.input, outputs=pred)

    print(model.summary())

    print('Layers:', len(model.layers))

    for layer in model.layers[:FLAGS.layer_to_train]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=46, validation_data=valid_batches,
                        validation_steps=537 // 75, epochs=100, verbose=2)
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
    FLAGS, unparsed = parser.parse_known_args()
    main()
