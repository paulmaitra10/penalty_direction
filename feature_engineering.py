import tensorflow as tf

def preprocess_images(X):
    return X / 255.0

def one_hot_encode(y, num_classes):
    return tf.keras.utils.to_categorical(y, num_classes=num_classes)

