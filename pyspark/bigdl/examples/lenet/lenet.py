#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.models.lenet.utils import *
from bigdl.nn.keras.layer import *
from bigdl.dataset.transformer import *


def load_data(location="/tmp/mnist"):
    (train_images, train_labels) = mnist.read_data_sets(location, "train")
    (test_images, test_labels) = mnist.read_data_sets(location, "test")
    X_train = normalizer(train_images, mnist.TRAIN_MEAN, mnist.TRAIN_STD)
    X_test = normalizer(test_images, mnist.TRAIN_MEAN, mnist.TRAIN_STD)
    Y_train = train_labels + 1
    Y_test = test_labels + 1
    return (X_train, Y_train), (X_test, Y_test)


def build_model(class_num):
    model = Sequential()
    model.add(Reshape((1, 28, 28), input_shape=(28, 28, 1)))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(Convolution2D(32, 3, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation="softmax"))
    return model


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = load_data()

    model = build_model(10)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=12,
              validation_data=(X_test, Y_test))
