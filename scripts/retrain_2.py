import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Model

train_path = '../data/train'
test_path = '../data/test'
valid_path = '../data/valid'

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).\
    flow_from_directory(train_path, target_size=(224, 224), batch_size=100)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).\
    flow_from_directory(test_path, target_size=(224, 224), batch_size=16, shuffle=False)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).\
    flow_from_directory(valid_path, target_size=(224, 224), batch_size=16)

mobile = keras.applications.mobilenet.MobileNet()

print(mobile.summary())

x = mobile.layers[-1].output
pred = Dense(16, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=pred)
print(model.summary())

print('Layers:', len(model.layers))

for layer in model.layers[:-1]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(lr=10e-4), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=46, validation_data=valid_batches,
                    validation_steps=537//16, epochs=500, verbose=2)















