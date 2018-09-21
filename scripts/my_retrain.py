import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint


IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
EPOCHS = 10
BATCH_SIZE = 64
FULLY_CONN_SIZE = 1024
# NB_IV3_LAYERS_TO_FREEZE = 172

def get_nb_files(directory):

    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune(model):

    # NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    for layer in model.layers[:len(model.layers)-2]:
        layer.trainable = False
    for layer in model.layers[len(model.layers)-2:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # data prep
    train_datagen = ImageDataGenerator(rotation_range=30,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(args.train_dir,
                                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(args.val_dir,
                                                            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                            batch_size=batch_size)

    # base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model_base = MobileNet(weights='imagenet', include_top=False,
                           input_shape=(224, 224, 3))

    x = model_base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FULLY_CONN_SIZE, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=model_base.input, output=predictions)

    # model = Model(inputs=model_base.input, outputs=model_top(model_base.output))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=2),
        ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                        save_best_only=True, verbose=2)]

    # Transfer learning:
    setup_to_transfer_learn(model, model_base)
    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        callbacks=callbacks,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    # fine-tuning
    setup_to_finetune(model)
    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_tl)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=EPOCHS)
    a.add_argument("--batch_size", default=BATCH_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
