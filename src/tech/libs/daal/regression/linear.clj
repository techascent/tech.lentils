(ns tech.libs.daal.regression.linear
  (:require [tech.libs.daal.regression.base :as base]
            [tech.ml.model :as model]
            [tech.ml.gridsearch :as gs]
            [tech.datatype :as dtype]
            [tech.compute.tensor :as ct])
  (:import [com.intel.daal.algorithms.linear_regression Model ModelQR]
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
      ;;bothans died for this one
      (.pack linear-model)

      (model/model->byte-array linear-model))))


(defn predict
  [^DaalContext daal-context
   options
   ^bytes model-data
   ^HomogenNumericTable feature-table]
  (let [^Model model (model/byte-array->model model-data)
        _ (.unpack model daal-context)
        pred-batch (PredictionBatch. daal-context (.getNumericType feature-table)
                                     PredictionMethod/defaultDense)
        pred-input (doto (.input pred-batch)
                     (.set PredictionInputId/data feature-table)
                     (.set PredictionInputId/model model))
        pred-results (.compute pred-batch)
        ^NumericTable result-table (.get pred-results PredictionResultId/prediction)
        [n-rows n-cols] (ct/shape result-table)
        table-data (dtype/->array-copy result-table)]
    (if (= 1 n-cols)
      table-data
      (ct/in-place-reshape table-data (ct/shape result-table)))))


(base/register-regression-system! :linear
                                  {:training-method (gs/nominative [:qr :norm-eq])}
                                  train
                                  predict)
