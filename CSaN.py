import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

def hardware_check():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found. Check TensorFlow-GPU support.")
        raise RuntimeError("GPU is not available")
    else:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Available GPU devices:", gpus)

# creating new model on top of the base model
def create_transfer_learning_model(base_model):
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00000001) # play around with lower learning rate + different batch size
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocessing_image():
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


def train_model_image(base_model, resource, train_generator, x_train_len):
    model = create_transfer_learning_model(base_model)
    if resource == 'CPU':
        print("Model Device Placement:")
        model.summary()
    start_time = time.time()
    with tf.device(f'/{resource}:0'):
        model.fit(train_generator, epochs=2, steps_per_epoch=x_train_len//256)  # play around with lower learning rate + different batch size
    end_time = time.time()
    return end_time - start_time

def preprocessing_text(maxlen):
    (train_data, train_labels), _ = tf.keras.datasets.imdb.load_data(num_words=10000)
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
    train_labels = (train_labels >= 7).astype(int)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([str(sentence) for sentence in train_data])

    x_train_text = tokenizer.texts_to_sequences([str(sentence) for sentence in train_data])
    x_train_text = tf.keras.preprocessing.sequence.pad_sequences(x_train_text, maxlen=maxlen)

    y_train_text = train_labels

    return x_train_text, y_train_text

# simplier version of what we used
def create_text_model(maxlen):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=16, input_length=maxlen))
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_complex_text_model(maxlen):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=32, input_length=maxlen))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))  # Adjust output classes
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model_text(x_train, y_train, train_generator, x_train_len, resource):
    model = create_complex_text_model(100)
    if resource == 'CPU':
        print("Model Device Placement:")
        model.summary()

    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=10)  # Adjust output classes

    start_time = time.time()
    with tf.device(f'/{resource}:0'):
        model.fit(x_train, y_train_onehot, epochs=2, steps_per_epoch=x_train_len//256)
    end_time = time.time()
    return end_time - start_time

def main():

    hardware_check()

    # images
    x_train, y_train, train_generator = preprocessing_image()
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    cpu_time_image = train_model_image(base_model, 'CPU', train_generator, len(x_train))
    gpu_time_image = train_model_image(base_model, 'GPU', train_generator, len(x_train))

    # text
    x_train_text, y_train_text = preprocessing_text(100)
    cpu_time_text = train_model_text(x_train_text, y_train_text, train_generator, len(x_train), 'CPU')
    gpu_time_text = train_model_text(x_train_text, y_train_text, train_generator, len(x_train), 'GPU')

    print(f"Time taken to train image model on CPU: {cpu_time_image} seconds")
    print(f"Time taken to train text model on CPU: {cpu_time_text} seconds")
    print(f"Time taken to train image model on GPU: {gpu_time_image} seconds")
    print(f"Time taken to train text model on GPU: {gpu_time_text} seconds")



if __name__ == "__main__":
    main()
