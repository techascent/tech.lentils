(ns tech.libs.daal.regression.ridge
  (:require [tech.libs.daal.regression.base :as base]
            [tech.ml.model :as model]
            [tech.ml.gridsearch :as gs]
            [tech.datatype :as dtype]
            [tech.compute.tensor :as ct])
  (:import [com.intel.daal.algorithms.ridge_regression Model]
           [com.intel.daal.algorithms.ridge_regression.training
            TrainingBatch TrainingMethod TrainingInputId TrainingResultId]
           [com.intel.daal.algorithms.ridge_regression.prediction
            PredictionBatch PredictionInputId PredictionResult PredictionResultId
            PredictionMethod]
           [com.intel.daal.data_management.data HomogenNumericTable]
           [com.intel.daal.services DaalContext]))



(defn train
  [^DaalContext daal-context
   options
   ^HomogenNumericTable feature-table
   ^HomogenNumericTable label-table]
  (let [batch-trainer (TrainingBatch. daal-context (.getNumericType feature-table)
                                      TrainingMethod/normEqDense)
        trainer-input (doto (.input batch-trainer)
                        (.set TrainingInputId/data feature-table)
                        (.set TrainingInputId/dependentVariable label-table))]
    (let [train-result (.compute batch-trainer)
          ^Model ridge-model (.get train-result TrainingResultId/model)]
      (base/pack-model ridge-model (.getBeta ridge-model)))))


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


(base/register-regression-system! :ridge
                                  {}
                                  train
                                  predict)
