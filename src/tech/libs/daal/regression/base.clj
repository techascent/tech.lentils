(ns tech.libs.daal.regression.base
  (:require [tech.ml.protocols.system :as ml-system]
            [tech.ml.registry :as registry]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.column :as ds-col]
            [tech.ml.dataset.options :as ds-opts]
            [tech.ml.model :as ml-model]
            [tech.datatype :as dtype]
            ;;Ensure protocols are loaded
            [tech.libs.daal.numeric-table :as numeric-table]
            [clojure.core.matrix :as m]
            [tech.libs.daal.context :as daal-ctx]
            [tech.compute.tensor :as ct])
  (:import [com.intel.daal.data_management.data
            HomogenNumericTable NumericTable SerializableBase]
           [com.intel.daal.services DaalContext]
           [com.intel.daal.algorithms Model]
           [com.intel.daal.data_management.data DataFeatureUtils$FeatureType]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn create-homogeneous-table
  ^HomogenNumericTable [table-data n-rows n-columns metadata-list ^DaalContext daal-context]
  (let [n-rows (long n-rows)
        n-cols (long n-columns)]
    (when-not (= (* n-rows n-cols)
                 (dtype/ecount table-data))
      (throw (ex-info "Table data ecount mismatch"
                      {:expected-ecount (* n-rows n-cols)
                       :actual-ecount (dtype/ecount table-data)})))
    (let [^HomogenNumericTable retval
          (case (dtype/get-datatype table-data)
            :float64 (HomogenNumericTable. daal-context ^doubles table-data n-cols n-rows)
            :float32 (HomogenNumericTable. daal-context ^floats table-data n-cols n-rows))
          table-dict (.getDictionary retval)]
      ;;It is important for certain algorithms for the input data to be marked as
      ;;categorical
      (doseq [categorical-idx (->> metadata-list
                                   (map-indexed vector)
                                   (filter #(contains? (second %) :categorical?))
                                   (map first))]
        (.setFeature table-dict (.getNumericType retval) (int categorical-idx)
                     DataFeatureUtils$FeatureType/DAAL_CATEGORICAL))
      retval)))


(defn dataset->metadata-map
  [dataset]
  (->> (ds/columns dataset)
       (map (fn [col]
              [(ds-col/column-name col)
               (ds-col/metadata col)]))
       (into {})))


(defn create-tables
  [dataset feature-row-map row-key-seq datatype daal-context]
  (let [metadata-map (dataset->metadata-map dataset)
        row-major-data (ds/->row-major dataset feature-row-map {:datatype datatype})
        first-item (first row-major-data)
        [n-cols n-rows] (m/shape dataset)]
    (->> row-key-seq
         (mapv (fn [row-key]
                 (let [first-row-data (get first-item row-key)
                       item-ecount (dtype/ecount first-row-data)
                       metadata-seq (mapv #(get metadata-map %) (get feature-row-map row-key))]
                   (-> (dtype/copy-raw->item! (map row-key row-major-data)
                                              (dtype/make-array-of-type datatype (* (long n-rows) item-ecount))
                                              0)
                       first
                       (create-homogeneous-table n-rows item-ecount metadata-seq daal-context))))))))


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


(defn pack-model
  [^SerializableBase model ^NumericTable model-beta]
  (.pack model)
  (merge
   {:model-bytes (ml-model/model->byte-array model)}
   (when model-beta
     {:explanatory-variables (dtype/->array-copy model-beta)})))


(defn unpack-model
  [^DaalContext daal-context {:keys [model-bytes]}]
  (let [^SerializableBase retval (ml-model/byte-array->model model-bytes)]
    (.unpack retval daal-context)
    retval))

(defn unpack-prediction
  [^NumericTable pred]
  (let [[n-rows n-cols] (m/shape pred)
        table-data (dtype/->array-copy pred)]
    (if (= 1 n-cols)
      table-data
      (ct/in-place-reshape table-data (ct/shape pred)))))


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
            (create-tables dataset {:features (ds-opts/feature-column-names options)
                                    :labels (ds-opts/label-column-names options)}
                           [:features :labels]
                           datatype
                           daal-context)
            train-fn (-> (options->regression-system options)
                         :train-fn)]
        (train-fn daal-context options feature-table label-table))))

  (predict [system options model dataset]
    (daal-ctx/with-daal-context
      (let [datatype (or (:datatype options) :float64)
            [n-cols n-rows] (m/shape dataset)
            n-rows (long n-rows)
            [feature-table]
            (create-tables dataset {:features (ds-opts/feature-column-names options)}
                           [:features]
                           datatype
                           daal-context)
            predict-fn (-> (options->regression-system options)
                           :predict-fn)]
        (predict-fn daal-context options model feature-table)))))


(def system (->DaalSystem))

(registry/register-system system)
