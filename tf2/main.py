import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
from hashing_memory import HashingMemory, AttrDict

import argparse

parser = argparse.ArgumentParser("Minimal tensorflow2 PKM layer example")
parser.add_argument('--pkm_off', default=False, action="store_true", help='turn off pkm memory')
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 5GB of memory on the first GPU, getting crashing otherwise
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    # maybe an issue with a non-headless node?
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self, pkm_on):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.conv2 = Conv2D(64, 3)
        self.pool = MaxPool2D()
        self.dropout1 = Dropout(.25)
        self.dropout2 = Dropout(.5)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')

        params = AttrDict({
                "sparse": False,
                "k_dim": 128,
                "heads": 4,
                "knn": 32,
                "n_keys": 512,  # the memory will have (n_keys ** 2) values
                "query_batchnorm": True,
                "input_dropout": 0,
                "query_dropout": 0,
                "value_dropout": 0,
            })

        self.memory_on = pkm_on
        if self.memory_on:
            print("PKM ON")
            self.memory = HashingMemory(128, 128, params)
        else:
            print("PKM OFF")
            self.memory = Dense(128)

        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = x + self.memory(x)
        x = self.dropout2(x)
        return self.d2(x)

# Create an instance of the model
model = MyModel(not args.pkm_off)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adadelta(learning_rate=1, rho=.90, epsilon=1e-06)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 10

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                            train_loss.result(),
                            train_accuracy.result()*100,
                            test_loss.result(),
                            test_accuracy.result()*100))