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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.example.keras.LeNet
import com.intel.analytics.bigdl.models.vgg.{VggForCifar10 => VGG1}
import com.intel.analytics.bigdl.models.kerasvgg.{VggForCifar10 => VGG2}
import com.intel.analytics.bigdl.models.inception.{Inception_v1_NoAuxClassifier => Inception1}
import com.intel.analytics.bigdl.models.kerasinception.{Inception_v1_NoAuxClassifier => Inception2}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class LeNetSpec extends FlatSpec with Matchers {

  "LeNet" should "generate the correct outputShape" in {
    val cnn = LeNet()
    cnn.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "LeNet forward and backward" should "work properly" in {
    val cnn = LeNet()
    val input = Tensor[Float](Array(2, 28, 28, 1)).rand()
    val output = cnn.forward(input)
    val gradInput = cnn.backward(input, output)
  }

  "VGG sequential" should "be correct" in {
    val vgg1 = VGG1(10, hasDropout = false)
    val vgg2 = VGG2(10, hasDropout = false)
    vgg2.setWeightsBias(vgg1.parameters()._1)
    val input = Tensor[Float](Array(2, 3, 32, 32)).rand()
    val output1 = vgg1.forward(input)
    val output2 = vgg2.forward(input)
    val gradInput1 = vgg1.backward(input, output1)
    val gradInput2 = vgg2.backward(input, output2)
    output1.toTensor[Float].almostEqual(output2.toTensor[Float], 1e-5) should be(true)
    gradInput1.toTensor[Float].almostEqual(gradInput2.toTensor[Float], 1e-5) should be(true)
  }

  "VGG graph" should "be correct" in {
    val vgg1 = VGG1.graph(10, hasDropout = false)
    val vgg2 = VGG2.graph(10, hasDropout = false)
    vgg2.setWeightsBias(vgg1.parameters()._1)
    val input = Tensor[Float](Array(2, 3, 32, 32)).rand()
    val output1 = vgg1.forward(input)
    val output2 = vgg2.forward(input)
    val gradInput1 = vgg1.backward(input, output1)
    val gradInput2 = vgg2.backward(input, output2)
    output1.toTensor[Float].almostEqual(output2.toTensor[Float], 1e-5) should be(true)
    gradInput1.toTensor[Float].almostEqual(gradInput2.toTensor[Float], 1e-5) should be(true)
  }

  "InceptionV1" should "be correct" in {
    val inception1 = Inception1(1000)
    val inception2 = Inception2(1000)
    inception2.setWeightsBias(inception1.parameters()._1)
    val input = Tensor[Float](Array(2, 3, 224, 224)).rand()
    val output1 = inception1.forward(input)
    val output2 = inception2.forward(input)
    val gradInput1 = inception1.backward(input, output1)
    val gradInput2 = inception1.backward(input, output2)
    output1.toTensor[Float].almostEqual(output2.toTensor[Float], 1e-5) should be(true)
    gradInput1.toTensor[Float].almostEqual(gradInput2.toTensor[Float], 1e-5) should be(true)
  }

}
