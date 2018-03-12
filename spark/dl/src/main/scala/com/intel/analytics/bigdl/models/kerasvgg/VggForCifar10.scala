/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.models.kerasvgg

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.numeric.NumericFloat

object VggForCifar10 {
  def apply(classNum: Int, hasDropout: Boolean = true): Sequential[Float] = {
    val vggBnDo = Sequential[Float]()
    vggBnDo.add(InputLayer(inputShape = Shape(3, 32, 32)))

    def convBNReLU(nOutPutPlane: Int)
    : Sequential[Float] = {
      vggBnDo.add(Convolution2D(nOutPutPlane, 3, 3, borderMode = "same"))
      vggBnDo.add(BatchNormalization())
      vggBnDo.add(Activation("relu"))
      vggBnDo
    }
    convBNReLU(64)
    if (hasDropout) vggBnDo.add(Dropout(0.3))
    convBNReLU(64)
    vggBnDo.add(MaxPooling2D())

    convBNReLU(128)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(128)
    vggBnDo.add(MaxPooling2D())

    convBNReLU(256)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(256)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(256)
    vggBnDo.add(MaxPooling2D())

    convBNReLU(512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512)
    vggBnDo.add(MaxPooling2D())

    convBNReLU(512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512)
    vggBnDo.add(MaxPooling2D())
    vggBnDo.add(Reshape(Array(512)))

    val classifier = Sequential[Float]()
    classifier.add(InputLayer(inputShape = Shape(512)))
    if (hasDropout) classifier.add(Dropout(0.5))
    classifier.add(Dense(512))
    classifier.add(BatchNormalization(epsilon = 1e-5, momentum = 0.1))
    classifier.add(Activation("relu"))
    if (hasDropout) classifier.add(Dropout(0.5))
    classifier.add(Dense(classNum))
    classifier.add(Activation("softmax"))
    vggBnDo.add(classifier)

    vggBnDo
  }

  def graph(classNum: Int, hasDropout: Boolean = true)
  : Model[Float] = {
    val input = Input(inputShape = Shape(3, 32, 32))

    def convBNReLU(nOutPutPlane: Int)(input: ModuleNode[Float])
    : ModuleNode[Float] = {
      val conv = Convolution2D(nOutPutPlane, 3, 3, borderMode = "same").inputs(input)
      val bn = BatchNormalization().inputs(conv)
      Activation("relu").inputs(bn)
    }
    val relu1 = convBNReLU(64)(input)
    val drop1 = if (hasDropout) Dropout(0.3).inputs(relu1) else relu1
    val relu2 = convBNReLU(64)(drop1)
    val pool1 = MaxPooling2D().inputs(relu2)

    val relu3 = convBNReLU(128)(pool1)
    val drop2 = if (hasDropout) Dropout(0.4).inputs(relu3) else relu3
    val relu4 = convBNReLU(128)(drop2)
    val pool2 = MaxPooling2D().inputs(relu4)

    val relu5 = convBNReLU(256)(pool2)
    val drop3 = if (hasDropout) Dropout(0.4).inputs(relu5) else relu5
    val relu6 = convBNReLU(256)(drop3)
    val drop4 = if (hasDropout) Dropout(0.4).inputs(relu6) else relu6
    val relu7 = convBNReLU(256)(drop4)
    val pool3 = MaxPooling2D().inputs(relu7)

    val relu8 = convBNReLU(512)(pool3)
    val drop5 = if (hasDropout) Dropout(0.4).inputs(relu8) else relu8
    val relu9 = convBNReLU(512)(drop5)
    val drop6 = if (hasDropout) Dropout(0.4).inputs(relu9) else relu9
    val relu10 = convBNReLU(512)(drop6)
    val pool4 = MaxPooling2D().inputs(relu10)

    val relu11 = convBNReLU(512)(pool4)
    val drop7 = if (hasDropout) Dropout(0.4).inputs(relu11) else relu11
    val relu12 = convBNReLU(512)(drop7)
    val drop8 = if (hasDropout) Dropout(0.4).inputs(relu12) else relu12
    val relu13 = convBNReLU(512)(drop8)
    val pool5 = MaxPooling2D().inputs(relu13)
    val reshape = Reshape(Array(512)).inputs(pool5)

    val drop9 = if (hasDropout) Dropout(0.5).inputs(reshape) else reshape
    val linear1 = Dense(512).inputs(drop9)
    val bn = BatchNormalization(epsilon = 1e-5, momentum = 0.1).inputs(linear1)
    val relu = Activation("relu").inputs(bn)
    val drop10 = if (hasDropout) Dropout(0.5).inputs(relu) else relu
    val linear2 = Dense(classNum).inputs(drop10)
    val output = Activation("softmax").inputs(linear2)
    Model(input, output)
  }

}
