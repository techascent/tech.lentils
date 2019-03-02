(ns tech.libs.daal.regression.gradient-boosted-trees
  (:require [tech.libs.daal.regression.base :as base]
            [tech.ml.model :as model]
            [tech.ml.gridsearch :as gs]
            [tech.datatype :as dtype]
            [tech.compute.tensor :as ct])
  (:import [com.intel.daal.algorithms.gbt.regression Model]
           [com.intel.daal.algorithms.gbt.regression.training InputId
            TrainingBatch TrainingMethod TrainingResultId Parameter]
           [com.intel.daal.algorithms.gbt.training SplitMethod]
           [com.intel.daal.algorithms.gbt.regression.prediction
            PredictionBatch PredictionResult PredictionResultId
            PredictionMethod ModelInputId NumericTableInputId]
           [com.intel.daal.data_management.data HomogenNumericTable]
           [com.intel.daal.services DaalContext]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(def split-method-table
  {:exact SplitMethod/exact
   :inexact SplitMethod/inexact})

(defn keyword->split-method
  ^SplitMethod [kwd]
  (if-let [retval (get split-method-table kwd)]
    retval
    (throw (ex-info (format "Unrecognized split method: %s" kwd) {}))))


(def option-map
  {:max-nodes {:setter #(.setMaxTreeDepth ^Parameter %1 (long %2))
               :gs (gs/linear-long [1e1 1e3])}
   :shrinkage {:setter #(.setShrinkage ^Parameter %1 (double %2))
               :gs (gs/linear [1e-2 0.99])}
   :sampling-fraction {:setter #(.setObservationsPerTreeFraction ^Parameter %1 (double %2))
                       :gs (gs/linear [1e-2 0.99])}
   :max-iterations {:setter #(.setMaxIterations ^Parameter %1 (long %2))
                    :gs (gs/linear-long [1e1 1e3])}
   :split-method {:setter #(.setSplitMethod ^Parameter %1 (keyword->split-method %2))
                  :gs (gs/nominative [:exact :inexact])}})



(defn train
  [^DaalContext daal-context
   options
   ^HomogenNumericTable feature-table
   ^HomogenNumericTable label-table]
  (let [batch-trainer (TrainingBatch. daal-context (.getNumericType feature-table)
                                      TrainingMethod/defaultDense)
        batch-params (.parameter batch-trainer)
        _ (doseq [[item-key item-val] (select-keys options (keys option-map))]
            ((get-in option-map [item-key :setter]) batch-params item-val))]
    ;;Intel takes care of the threading so we don't have to.
    (locking option-map
      (doto (.input batch-trainer)
        (.set InputId/data feature-table)
        (.set InputId/dependentVariable label-table))
      (let [train-result (.compute batch-trainer)
            ^Model gbt-model (.get train-result TrainingResultId/model)]
        (base/pack-model gbt-model nil)))))



(defn predict
  [^DaalContext daal-context
   options
   model-data
   ^HomogenNumericTable feature-table]
  (let [model (base/unpack-model daal-context model-data)
        pred-batch (PredictionBatch. daal-context (.getNumericType feature-table)
                                     PredictionMethod/defaultDense)
        pred-input (doto (.input pred-batch)
                     (.set NumericTableInputId/data feature-table)
                     (.set ModelInputId/model ^Model model))]
    (-> (.compute pred-batch)
        (.get PredictionResultId/prediction)
        (base/unpack-prediction))))


(defn gridsearch-map
  []
  (->> option-map
       (map (fn [[k v-map]]
              [k (:gs v-map)]))
       (into {})))


(base/register-regression-system! :gradient-boosted-trees
                                  (gridsearch-map)
                                  train
                                  predict)
