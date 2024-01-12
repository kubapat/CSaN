import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10


# creating new model on top of the base model
def create_transfer_learning_model(base_model):
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00000001) # play around with lower learning rate + different batch size
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def hardware_check():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found. Check TensorFlow-GPU support.")
        raise RuntimeError("GPU is not available")
    else:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Available GPU devices:", gpus)


def preprocessing():
    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    #y_train = tf.keras.utils.to_categorical(y_train, 10)
    # using ImageDataGenerator for on-the-fly image resizing and augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True
    )
    # generator for training augmented data
    train_generator = datagen.flow(x_train, y_train, batch_size=256) # play around with lower learning rate + different batch size
    return x_train, y_train, train_generator


def train_model(base_model, resource, train_generator, x_train_len):
    model = create_transfer_learning_model(base_model)
    if resource == 'CPU':
        print("Model Device Placement:")
        model.summary()
    start_time = time.time()
    with tf.device(f'/{resource}:0'):
        model.fit(train_generator, epochs=10, steps_per_epoch=x_train_len//256)  # play around with lower learning rate + different batch size
    end_time = time.time()
    return end_time - start_time


def main():
    hardware_check()
    x_train, y_train, train_generator = preprocessing()
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #cpu_time = train_model(base_model, 'CPU', train_generator, len(x_train)) # took away for faster impoving model learning 
    gpu_time = train_model(base_model, 'GPU', train_generator, len(x_train))

    print(f"Time taken to train on CPU: {cpu_time} seconds")
    print(f"Time taken to train on GPU: {gpu_time} seconds")


if __name__ == "__main__":
    main()
