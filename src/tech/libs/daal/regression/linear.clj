(ns tech.libs.daal.regression.linear
  (:require [tech.libs.daal.regression.base :as base]
            [tech.ml.model :as model]
            [tech.ml.gridsearch :as gs]
            [tech.datatype :as dtype]
            [tech.compute.tensor :as ct])
  (:import [com.intel.daal.algorithms.linear_regression Model]
           [com.intel.daal.algorithms.linear_regression.training
            TrainingBatch TrainingMethod TrainingInputId TrainingResultId]
           [com.intel.daal.algorithms.linear_regression.prediction
            PredictionBatch PredictionInputId PredictionResult PredictionResultId
            PredictionMethod]
           [com.intel.daal.data_management.data HomogenNumericTable]
           [com.intel.daal.services DaalContext]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn options->training-method
  [options]
  (let [method (or (:training-method options)
                   :default)]
    (case method
      :default TrainingMethod/qrDense
      :qr TrainingMethod/qrDense
      :norm-eq TrainingMethod/normEqDense)))


(defn train
  [^DaalContext daal-context
   options
   ^HomogenNumericTable feature-table
   ^HomogenNumericTable label-table]
  (let [batch-trainer (TrainingBatch. daal-context (.getNumericType feature-table)
                                      (options->training-method options))
        trainer-input (doto (.input batch-trainer)
                        (.set TrainingInputId/data feature-table)
                        (.set TrainingInputId/dependentVariable label-table))]
    (let [train-result (.compute batch-trainer)
          ^Model linear-model (.get train-result TrainingResultId/model)]
      (base/pack-model linear-model (.getBeta linear-model)))))


(defn predict
  [^DaalContext daal-context
   options
   model-data
   ^HomogenNumericTable feature-table]
  (let [model (base/unpack-model daal-context model-data)
        pred-batch (PredictionBatch. daal-context (.getNumericType feature-table)
                                     PredictionMethod/defaultDense)
        pred-input (doto (.input pred-batch)
                     (.set PredictionInputId/data feature-table)
                     (.set PredictionInputId/model model))]
    (-> (.compute pred-batch)
        (.get PredictionResultId/prediction)
        (base/unpack-prediction))))


(base/register-regression-system! :linear
                                  {:training-method (gs/nominative [:qr :norm-eq])}
                                  train
                                  predict)
