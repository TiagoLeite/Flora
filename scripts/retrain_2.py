import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Reshape
from keras.models import Model
import argparse

train_path = '../data/train'
test_path = '../data/test'
valid_path = '../data/valid'

FLAGS = None


def main():
    train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(train_path, target_size=(224, 224), batch_size=100)
    test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(test_path, target_size=(224, 224), batch_size=100, shuffle=False)
    valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input). \
        flow_from_directory(valid_path, target_size=(224, 224), batch_size=100)

    mobile = keras.applications.mobilenet.MobileNet(weights='imagenet')

    print(mobile.summary())

    x = mobile.layers[FLAGS.layer_to_append].output
    reshaped = Reshape(target_shape=[1000], name='tiago_reshape')(x)
    pred = Dense(16, activation='softmax')(reshaped)
    model = Model(inputs=mobile.input, outputs=pred)

    print(model.summary())

    print('Layers:', len(model.layers))

    for layer in model.layers[:FLAGS.layer_to_train]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(lr=10e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(train_batches, steps_per_epoch=46, validation_data=valid_batches,
                        validation_steps=537//100, epochs=500, verbose=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--layer_to_append',
        type=int,
        default=-3,
    )
    parser.add_argument(
        '--layer_to_train',
        type=int,
        default=0,
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()
