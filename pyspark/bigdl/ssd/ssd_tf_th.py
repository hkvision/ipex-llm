import keras
import numpy as np
from bigdl.ssd.ssd import SSD300

# SSD in th dim_ordering
keras.backend.set_image_dim_ordering("th")
kmodel1 = SSD300(input_shape=(3, 300, 300))
# kmodel1.load_weights("/home/kai/weights_SSD300_th.hdf5")

# SSD in tf dim_ordering with pre-trained weights
keras.backend.set_image_dim_ordering("tf")
kmodel2 = SSD300(input_shape=(300, 300, 3))
kmodel2.load_weights("/home/kai/weights_SSD300_tf.hdf5")

# Convert pre-trained tf weights to th
kweights1 = kmodel1.get_weights()
kweights2 = kmodel2.get_weights()
kw = []
for i in range(0, len(kweights2)):
    if len(kweights2[i].shape) == 4:
        w = np.transpose(kweights2[i], (3, 2, 0, 1))
        kw.append(w)
    else:
        kw.append(kweights2[i])
kmodel1.set_weights(kw)
kmodel1.save_weights("/home/kai/weights_SSD300_th.hdf5")

input_data1 = np.random.random([1, 3, 300, 300])
input_data2 = np.transpose(input_data1, (0, 2, 3, 1))
koutput1 = kmodel1.predict(input_data1)
koutput2 = kmodel2.predict(input_data2)
np.testing.assert_allclose(koutput1, koutput2, 1e-5, 1e-5)

# Load Keras SSD th as a BigDL model
from keras.utils.generic_utils import CustomObjectScope
from bigdl.ssd.ssd_layers import Normalize, PriorBox
from bigdl.keras.converter import *
from bigdl.nn.layer import Model
with CustomObjectScope({"Normalize": Normalize, 'PriorBox': PriorBox}):
    bmodel = DefinitionLoader.from_kmodel(kmodel1)
    WeightLoader.load_weights_from_kmodel(bmodel, kmodel1)
    np.testing.assert_allclose(koutput1, bmodel.forward(input_data1), 1e-2, 1e-2)
    bmodel.save("/home/kai/ssd.bigdl", True)
    bmodel2 = Model.load("/home/kai/ssd.bigdl")
