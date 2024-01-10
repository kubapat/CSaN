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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def hardware_check():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found. Check TensorFlow-GPU support.")
        raise RuntimeError("GPU is not available")
    else:
        print("Available GPU devices:", gpus)


def preprocessing():
    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    return x_train, y_train


def train_model(base_model, resource, x_train, y_train):
    model = create_transfer_learning_model(base_model)
    start_time = time.time()
    with tf.device(f'/{resource}:0'):
        model.fit(x_train, y_train, epochs=5, batch_size=64)  # maybe lower epochs due to time consumption
    end_time = time.time()
    return end_time - start_time


def main():
    hardware_check()
    x_train, y_train = preprocessing()
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    cpu_time = train_model(base_model, 'CPU', x_train, y_train)
    gpu_time = train_model(base_model, 'GPU', x_train, y_train)

    print(f"Time taken to train on CPU: {cpu_time} seconds")
    print(f"Time taken to train on GPU: {gpu_time} seconds")


if __name__ == "__main__":
    main()
