import tensorflow as tf


class CNN_Model(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu')
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        output = self.outputs(x)

        return output
