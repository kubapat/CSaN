import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import tensorflow_datasets as tfds

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
        model.fit(train_generator, epochs=2, steps_per_epoch=x_train_len//256)  # play around with lower learning rate + different batch size
    end_time = time.time()
    return end_time - start_time

def preprocessing_text(maxlen):
    # IMDB movie reviews dataset
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words=10000)

    # truncating and paddding the input sequences
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)
    test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=maxlen)

    # converting the labels to binary format (positive or negative sentiment
    train_labels = (train_labels >= 7).astype(int)
    test_labels = (test_labels >= 7).astype(int)

    # tokenize and pad sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([str(sentence) for sentence in train_data])  # Convert each sentence to a string

    # convert texts to sequences
    x_train_text = tokenizer.texts_to_sequences([str(sentence) for sentence in train_data])
    x_test_text = tokenizer.texts_to_sequences([str(sentence) for sentence in test_data])

    # padding the sequences
    x_train_text = tf.keras.preprocessing.sequence.pad_sequences(x_train_text, maxlen=maxlen)
    x_test_text = tf.keras.preprocessing.sequence.pad_sequences(x_test_text, maxlen=maxlen)

    # binary labels
    y_train_text = train_labels
    y_test_text = test_labels

    return x_train_text, y_train_text, x_test_text, y_test_text

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
    model.add(layers.Dense(2, activation='softmax'))  # Change output to 2 classes for binary classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model_text(x_train, y_train, x_test, y_test, maxlen, resource):
    model = create_complex_text_model(maxlen)
    if resource == 'CPU':
        print("Model Device Placement:")
        model.summary()

    # Convert labels to one-hot encoding for binary classification
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=2)

    start_time = time.time()
    with tf.device(f'/{resource}:0'):
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train_onehot, epochs=2, validation_data=(x_test, y_test_onehot))
    end_time = time.time()
    return end_time - start_time

def main():
    hardware_check()
    # images
    x_train, y_train, train_generator = preprocessing()
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    cpu_time_image = train_model(base_model, 'CPU', train_generator, len(x_train))
    gpu_time_image = train_model(base_model, 'GPU', train_generator, len(x_train))

    # text
    maxlen = 100 
    x_train_text, y_train_text, x_test_text, y_test_text = preprocessing_text(maxlen)
    cpu_time_text = train_model_text(x_train_text, y_train_text, x_test_text, y_test_text, maxlen, 'CPU')
    gpu_time_text = train_model_text(x_train_text, y_train_text, x_test_text, y_test_text, maxlen, 'GPU')


    print(f"Time taken to train image model on CPU: {cpu_time_image} seconds")
    print(f"Time taken to train text model on CPU: {cpu_time_text} seconds")
    print(f"Time taken to train image model on GPU: {gpu_time_image} seconds")
    print(f"Time taken to train text model on GPU: {gpu_time_text} seconds")



if __name__ == "__main__":
    main()
