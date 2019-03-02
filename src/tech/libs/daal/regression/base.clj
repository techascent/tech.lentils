(ns tech.libs.daal.regression.base
  (:require [tech.ml.protocols.system :as ml-system]
            [tech.ml.registry :as registry]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.options :as ds-opts]
            [tech.ml.model :as ml-model]
            [tech.datatype :as dtype]
            ;;Ensure protocols are loaded
            [tech.libs.daal.numeric-table :as numeric-table]
            [clojure.core.matrix :as m]
            [tech.libs.daal.context :as daal-ctx])
  (:import [com.intel.daal.data_management.data HomogenNumericTable]
           [com.intel.daal.services DaalContext]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn create-homogeneous-table
  ^HomogenNumericTable [table-data n-rows n-columns ^DaalContext daal-context]
  (let [n-rows (long n-rows)
        n-cols (long n-columns)]
    (when-not (= (* n-rows n-cols)
                 (dtype/ecount table-data))
      (throw (ex-info "Table data ecount mismatch"
                      {:expected-ecount (* n-rows n-cols)
                       :actual-ecount (dtype/ecount table-data)})))
    (case (dtype/get-datatype table-data)
      :float64 (HomogenNumericTable. daal-context ^doubles table-data n-cols n-rows)
      :float32 (HomogenNumericTable. daal-context ^floats table-data n-cols n-rows))))


(defn create-tables
  [row-major-data n-rows row-key-seq daal-context]
  (let [first-item (first row-major-data)]
    (->> row-key-seq
         (mapv (fn [row-key]
                 (let [first-row-data (get first-item row-key)
                       item-ecount (dtype/ecount first-row-data)
                       datatype (dtype/get-datatype first-row-data)]
                   (println {:n-rows n-rows
                             :item-ecount item-ecount})
                   (-> (dtype/copy-raw->item! (map row-key row-major-data)
                                              (dtype/make-array-of-type datatype (* (long n-rows) item-ecount))
                                              0)
                       first
                       (create-homogeneous-table n-rows item-ecount daal-context))))))))


(defonce sub-systems (atom {}))


(defn register-regression-system!
  [sub-type gs-opts train-fn predict-fn]
  (swap! sub-systems assoc sub-type
         {:sub-type sub-type
          :gs-opts gs-opts
          :train-fn train-fn
          :predict-fn predict-fn})
  (keys @sub-systems))


(defn get-regression-system
  [sub-type]
  (if-let [retval (get @sub-systems sub-type)]
    retval
    (throw (ex-info (format "Failed to find regression subsystem %s" sub-type)
                    {:sub-type sub-type}))))

(defn options->regression-system
  [options]
  (-> (ml-model/options->model-type options)
      get-regression-system))


(defrecord DaalSystem []
  ml-system/PMLSystem
  (system-name [system] :daal.regression)
  (gridsearch-options [system options]
    (merge (-> (options->regression-system options)
               :gs-opts)
           options))
  (train [system options dataset]
    (daal-ctx/with-daal-context
      (let [datatype (or (:datatype options) :float64)
            [n-cols n-rows] (m/shape dataset)
            n-rows (long n-rows)
            [feature-table label-table]
            (-> (ds/->row-major dataset
                                {:features (ds-opts/feature-column-names options)
                                 :labels (ds-opts/label-column-names options)}
                                {:datatype datatype})
                (create-tables n-rows [:features :labels] daal-context))
            train-fn (-> (options->regression-system options)
                         :train-fn)]
        (train-fn daal-context options feature-table label-table))))

  (predict [system options model dataset]
    (daal-ctx/with-daal-context
      (let [datatype (or (:datatype options) :float64)
            [n-cols n-rows] (m/shape dataset)
            n-rows (long n-rows)
            [feature-table]
            (-> (ds/->row-major dataset
                                {:features (ds-opts/feature-column-names options)
                                 :labels (ds-opts/label-column-names options)}
                                {:datatype datatype})
                (create-tables n-rows [:features :labels] daal-context))
            predict-fn (-> (options->regression-system options)
                           :predict-fn)]
        (predict-fn daal-context options model feature-table)))))


(def system (->DaalSystem))

(registry/register-system system)
