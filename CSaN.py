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

    """
    Checks and prints information about available GPU devices.
    Raises an error if no GPU is found.
    """

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found. Check TensorFlow-GPU support.")
        raise RuntimeError("GPU is not available")
    else:
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Available GPU devices:", gpus)


def create_transfer_learning_model(base_model):

    """
    Creates a transfer learning model by adding layers on top of a base model.
    """

    model = models.Sequential()
    
    # Base model (pre-trained)
    model.add(base_model)
    
    # Global average pooling layer
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense layer with ReLU activation
    model.add(layers.Dense(256, activation='relu'))
    
    # Dropout layer for regularization
    model.add(layers.Dropout(0.5))
    
    # Output layer with softmax activation for classification
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model with a very low learning rate, you can increase it if it's needed
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00000001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def preprocessing_image():

    """
    Preprocesses CIFAR-10 image data, including data augmentation: rescaling and horizontal flip.
    """

    (x_train, y_train), _ = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0

    # Image data generator for on-the-fly data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,            # Normalize pixel values to the range [0, 1]
        zoom_range=0.2,            # Randomly zoom into images by 20%
        horizontal_flip=True       # Randomly flip images horizontally
    )

    # Generator for training augmented data
    train_generator = datagen.flow(x_train, y_train, batch_size=256)

    return x_train, y_train, train_generator


def train_model_image(base_model, resource, train_generator, x_train_len):

    """
    Trains an image model using transfer learning on either CPU or GPU.
    """

    model = create_transfer_learning_model(base_model)

    if resource == 'CPU':
        print("Model Device Placement:")
        model.summary()
    
    # Training the model on the generator
    start_time = time.time()

    with tf.device(f'/{resource}:0'):
        model.fit(train_generator, epochs=2, steps_per_epoch=x_train_len//256)

    end_time = time.time()

    return end_time - start_time


def preprocessing_text(maxlen):

    """
    Preprocesses IMBD text data, including padding (model requires the same dimension of input data) 
    and tokenization (convert words to unique numbers for further training).
    """

    # Loading IMDB text data
    (train_data, train_labels), _ = tf.keras.datasets.imdb.load_data(num_words=10000)

    # Padding and preprocessing text data
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen)  # Pad sequences to a fixed length

    # Convert labels to binary (0 or 1) based on a threshold (e.g., 7 for positive sentiment)
    train_labels = (train_labels >= 7).astype(int)

    # Tokenizing and padding text data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')  # Tokenize words, keep 10,000 most frequent
    tokenizer.fit_on_texts([str(sentence) for sentence in train_data])  # Fit tokenizer on text data

    x_train_text = tokenizer.texts_to_sequences([str(sentence) for sentence in train_data])  # Convert text to sequences
    x_train_text = tf.keras.preprocessing.sequence.pad_sequences(x_train_text, maxlen=maxlen)  # Pad sequences to a fixed length

    y_train_text = train_labels  # Assigning labels to y_train_text

    return x_train_text, y_train_text


def create_text_model(maxlen):

    """
    Creates a simple text classification model.
    """

    model = models.Sequential()
    
    # Embedding layer for word embeddings
    model.add(layers.Embedding(input_dim=10000, output_dim=16, input_length=maxlen))
    
    # Global average pooling layer
    model.add(layers.GlobalAveragePooling1D())
    
    # Dense layer with ReLU activation
    model.add(layers.Dense(16, activation='relu'))
    
    # Output layer with sigmoid activation for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_complex_text_model(maxlen):

    """
    Creates a complex text classification model with bidirectional LSTM layers.
    """

    model = models.Sequential()
    
    # Embedding layer for word embeddings (convert word indices into dense vectors to capture semantic relationships between words)
    model.add(layers.Embedding(input_dim=10000, output_dim=32, input_length=maxlen))
    
    # Bidirectional LSTM layer with return sequences (Capture contextual information in both forward and backward directions = dependencies and relationships in both past and future contexts)
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    
    # Bidirectional LSTM layer
    model.add(layers.Bidirectional(layers.LSTM(32)))
    
    # Dense layer with ReLU activation
    model.add(layers.Dense(64, activation='relu'))
    
    # Dropout layer for regularization
    model.add(layers.Dropout(0.5))
    
    # Output layer with softmax activation for multiclass classification
    model.add(layers.Dense(10, activation='softmax'))  # Adjust output classes
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_model_text(x_train, y_train, train_generator, x_train_len, resource):

    """
    Trains a text classification model on either CPU or GPU.
    """

    model = create_complex_text_model(100)

    if resource == 'CPU':
        print("Model Device Placement:")
        model.summary()

    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=10)  # Adjust output classes

    start_time = time.time()

    with tf.device(f'/{resource}:0'):
        model.fit(x_train, y_train_onehot, epochs=2, steps_per_epoch=x_train_len//256)

    end_time = time.time()

    return end_time - start_time


def main():

    """
    Main function to execute the training and measure time consumption.
    """

    hardware_check()

    # Images
    x_train, y_train, train_generator = preprocessing_image()
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    cpu_time_image = train_model_image(base_model, 'CPU', train_generator, len(x_train))
    gpu_time_image = train_model_image(base_model, 'GPU', train_generator, len(x_train))

    # Text
    x_train_text, y_train_text = preprocessing_text(100)
    cpu_time_text = train_model_text(x_train_text, y_train_text, train_generator, len(x_train), 'CPU')
    gpu_time_text = train_model_text(x_train_text, y_train_text, train_generator, len(x_train), 'GPU')

    print(f"Time taken to train image model on CPU: {cpu_time_image} seconds")
    print(f"Time taken to train text model on CPU: {cpu_time_text} seconds")
    print(f"Time taken to train image model on GPU: {gpu_time_image} seconds")
    print(f"Time taken to train text model on GPU: {gpu_time_text} seconds")



if __name__ == "__main__":
    main()
