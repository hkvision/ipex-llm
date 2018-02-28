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

from bigdl.util.common import *


class Training(JavaValue):
    def __init__(self, bigdl_type="float"):
        self.bigdl_type = bigdl_type

    def compile(self, optimizer, loss, metrics=None):
        callBigDlFunc(self.bigdl_type, "compile",
                      self.value,
                      optimizer,
                      loss,
                      metrics)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, validation_data=None):
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            training_data = to_sample_rdd(x, y)
            validation_data = to_sample_rdd(*validation_data)
        elif isinstance(x, RDD) and not y:
            training_data = x
        else:
            raise TypeError("Unsupported training data type: %s" % type(x))
        callBigDlFunc(self.bigdl_type, "fit",
                      self.value,
                      training_data,
                      batch_size,
                      nb_epoch,
                      validation_data)
