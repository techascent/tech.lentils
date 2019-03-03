(ns tech.libs.daal.pca
  (:require [tech.ml.protocols.etl :as etl-proto]
            [tech.ml.dataset :as ds]
            [tech.libs.daal.context :as ctx]
            [tech.libs.daal.numeric-table :as numeric-table]
            [tech.ml.dataset.etl.pipeline-operators :as pip-ops]
            [tech.compute.tensor :as ct]
            [tech.ml.dataset.column :as ds-col]
            [tech.ml.dataset.tensor :as ds-tens]
            [tech.datatype :as dtype])
  (:import [com.intel.daal.algorithms.pca Batch InputId Method
            Result ResultId ResultsToComputeId]
           [com.intel.daal.data_management.data NumericTable
            HomogenNumericTable KeyValueDataCollection]
           [com.intel.daal.algorithms.pca.transform
            TransformBatch TransformInput TransformInputId
            TransformMethod TransformComponentId
            TransformDataInputId TransformResultId]))


(def keyword->results-to-compute-map
  {:means ResultsToComputeId/mean
   :variances ResultsToComputeId/variance
   :eigenvalues ResultsToComputeId/eigenvalue})


(defn keyword->results-to-compute
  ^long [kwd]
  (if-let [retval (get keyword->results-to-compute-map kwd)]
    retval
    (throw (ex-info (format "Unrecognized results %s" kwd)
                    {:kwd kwd}))))


(defn key-seq->results-flags
  ^long [key-seq]
  (->> key-seq
       (map keyword->results-to-compute)
       (reduce #(bit-or (long %1) (long %2)))
       long))


(defn pca
  "Perform PCA.  Return map containing the computed results.

  datatype - either :float32 or :float64.  If not specified, defaults to :float64
  key-seq - some or all of [:means :variances :eigenvalues]
  method - either :correlation or :svd

  Returns (depending on keys passed in):
  {:means - vector of means
   :variances - vector of variances
   :eigenvectors - dense matrix of eigenvectors, row major, one per row.
   :eigenvalues - vector of eigenvalues in same order as eigenvector rows."
  [^HomogenNumericTable row-major-data & {:keys [datatype key-seq method]
                                          :or {datatype :float64
                                               key-seq [:means :variances :eigenvalues]
                                               method :svd}}]
  (println "INPUT!!")
  (println (vec (numeric-table/->array-of-type row-major-data :float32)))

  (ctx/with-daal-context
    (let [[n-rows n-cols] (ct/shape row-major-data)
          algo (Batch. daal-context (.getNumericType row-major-data) (case method
                                                                       :svd Method/svdDense
                                                                       :correlation Method/correlationDense))
          _ (doto (.parameter algo)
              (.setResultsToCompute (key-seq->results-flags key-seq)))
          _ (doto (.input algo)
              (.set InputId/data row-major-data))
          ;;Not a cheap call...
          algo-result (.compute algo)
          result->ary #(numeric-table/->array-of-type (.get algo-result %) datatype)]
      (println "OUTPUT!!")
      (println (vec (numeric-table/->array-of-type (.get algo-result ResultId/eigenVectors) :float32)))
      (->> (set key-seq)
           (mapcat (fn [algo-key]
                     (case algo-key
                       :means [[:means (result->ary ResultId/means)]]
                       :variances [[:variances (result->ary ResultId/variances)]]
                       :eigenvalues [[:eigenvalues (result->ary ResultId/eigenValues)]
                                     [:eigenvectors (-> (result->ary ResultId/eigenVectors)
                                                        (ct/in-place-reshape [n-rows n-cols]))]])))
           (into {})))))


(defn pca-dataset
    "Perform PCA.  Return map containing the computed results.  Dataset must not have any
  non-numeric or missing values; non-numeric or missing causes errors.

  datatype - either :float32 or :float64.  If not specified, defaults to :float64
  key-seq - some or all of [:means :variances :eigenvalues]
  method - either :correlation or :svd

  Returns (depending on keys passed in):
  {:means - vector of means
   :variances - vector of variances
   :eigenvectors - dense matrix of eigenvectors, row major, one per row.
   :eigenvalues - vector of eigenvalues in same order as eigenvector rows."
  [dataset & {:keys [datatype key-seq method]
              :or {datatype :float64
                   key-seq [:means :variances :eigenvalues]
                   method :svd}}]
  (ctx/with-daal-context
    (pca (numeric-table/dataset->row-major-homogen-table dataset daal-context datatype)
         :datatype datatype
         :key-seq key-seq
         :method method)))



(defn pca-transform-dataset
  "Transform the dataset producing a new dataset.  n-components must be
  less than or equal to n-cols of the dataset.
  pca-info must have at least eigenvectors.  It may also contain means, variances,
  and eigenvalues."
  [dataset pca-info n-components datatype]
  (ctx/with-daal-context
    (let [[n-cols n-rows] (ct/shape dataset)
          _ (when-not (<= (long n-components) (long n-cols))
              (throw (ex-info "Num components is not less than or equal to n-cols")))
          ds-table (numeric-table/dataset->row-major-homogen-table
                    dataset daal-context datatype)

          algo (TransformBatch. daal-context (.getNumericType ds-table)
                                TransformMethod/defaultDense n-components)]
      (when-not (contains? pca-info :eigenvectors)
        (throw (ex-info "PCA info must contain eigenvectors" {})))
      (let [algo-input
            (doto (.input algo)
              (.set TransformInputId/data ds-table)
              (.set TransformInputId/eigenvectors (numeric-table/tensor->table
                                                   (:eigenvectors pca-info)
                                                   daal-context)))
            data-collection  (KeyValueDataCollection. daal-context)]
        (when (contains? pca-info :means)
          (.set data-collection
                (.getValue TransformComponentId/mean)
                (numeric-table/tensor->table (:means pca-info) daal-context)))
        (when (contains? pca-info :variances)
          (.set data-collection
                (.getValue TransformComponentId/variance)
                (numeric-table/tensor->table (:variances pca-info) daal-context)))
        (when (contains? pca-info :eigenvalues)
          (.set data-collection
                (.getValue TransformComponentId/eigenvalue)
                (numeric-table/tensor->table (:eigenvalues pca-info) daal-context)))
        (.set algo-input TransformDataInputId/dataForTransform data-collection)
        (let [results (.compute algo)
              result-table (-> (.get results TransformResultId/transformedData)
                               (numeric-table/table->tensor datatype))
              [n-result-rows n-result-cols] (ct/shape result-table)
              first-src-col (first (ds/columns dataset))]
          (ds-tens/row-major-tensor->dataset result-table dataset "pca-result"))))))
