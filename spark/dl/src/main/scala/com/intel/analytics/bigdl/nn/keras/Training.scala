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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.optim.{OptimMethod, Optimizer, Trigger, ValidationMethod}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.LoggerFilter
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

case class TrainingConfig[T](optimMethod: OptimMethod[T],
                            criterion: Criterion[T],
                            vMethods: Array[ValidationMethod[T]])

trait Training[T]{

  var model: Module[T]

  private var compile: TrainingConfig[T] = null

  /**
   * Configures the learning process.
   * Must be called before fit.
   */
  def compile(optimizer: OptimMethod[T], loss: Criterion[T],
              metrics: Array[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    this.compile = TrainingConfig(optimizer, loss, metrics)
  }

  /**
   * Trains the model for a fixed number of epochs.
   */
  def fit(x: RDD[Sample[T]], batchSize: Int = 32, epochs: Int = 10,
          verbose: Boolean = false, validationData: RDD[Sample[T]] = null)
    (implicit ev: TensorNumeric[T], ct: ClassTag[T]): Module[T] = {
    // TODO: local optimizer
    require(this.compile != null, "compile must be called before fit")
    if (!verbose) {
      LoggerFilter.redirectSparkInfoLogs()
    }
    val optimizer = Optimizer(
      model = model,
      sampleRDD = x,
      criterion = this.compile.criterion,
      batchSize = batchSize)
    optimizer.setOptimMethod(this.compile.optimMethod)
      .setEndWhen(Trigger.maxEpoch(epochs))
    if (validationData != null) {
      require(this.compile.vMethods != null, "Validation metrics haven't been set yet")
      optimizer.setValidation(trigger = Trigger.everyEpoch,
        sampleRDD = validationData,
        vMethods = this.compile.vMethods,
        batchSize = batchSize)
    }
    optimizer.optimize()
  }

  def fit[D: ClassTag](x: DataSet[D], epochs: Int,
                       verbose: Boolean, validationData: DataSet[MiniBatch[T]])
    (implicit ev: TensorNumeric[T], ct: ClassTag[T]): Module[T] = {
    require(this.compile != null, "compile must be called before fit")
    if (!verbose) {
      LoggerFilter.redirectSparkInfoLogs()
    }
    val optimizer = Optimizer(
      model = model,
      dataset = x,
      criterion = this.compile.criterion)
    if (validationData != null) {
      require(this.compile.vMethods != null, "Validation metrics haven't been set yet")
      optimizer.setValidation(trigger = Trigger.everyEpoch,
        dataset = validationData,
        vMethods = this.compile.vMethods)
    }
    optimizer.setOptimMethod(this.compile.optimMethod)
      .setEndWhen(Trigger.maxEpoch(epochs))
    optimizer.optimize()
  }

}
