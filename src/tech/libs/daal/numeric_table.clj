(ns tech.libs.daal.numeric-table
  (:require [tech.datatype.base :as dtype-base]
            [tech.datatype.java-primitive :as java-primitive]
            [clojure.core.matrix.protocols :as mp]
            [tech.datatype :as dtype]
            [tech.compute.tensor :as ct]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.tensor :as ds-tens]
            [tech.datatype :as dtype])
  (:import [com.intel.daal.data_management.data
            NumericTable HomogenNumericTable]
           [com.intel.daal.services DaalContext]
           [com.intel.daal.data_management.data DataFeatureUtils$FeatureType]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn to-homogen
  ^HomogenNumericTable [m]
  (when-not (instance? HomogenNumericTable m)
    (throw (ex-info "Table is not homogeneous"
                    {:table-type (type m)})))
  m)


(extend-type NumericTable
  mp/PDimensionInfo
  (dimensionality [m] (count (mp/get-shape m)))
  (get-shape [m]
    ;;Normal row-major shape
    [(.getNumberOfRows m)
     (.getNumberOfColumns m)])
  (is-scalar? [m] false)
  (is-vector? [m] true)
  (dimension-count [m dimension-number]
    (let [shape (mp/get-shape m)]
      (if (<= (count shape) (long dimension-number))
        (get shape dimension-number)
        (throw (ex-info "Array does not have specific dimension"
                        {:dimension-number dimension-number
                         :shape shape})))))

  mp/PElementCount
  (element-count [m] (apply * (mp/get-shape m)))

  dtype-base/PDatatype
  (get-datatype [m]
    (let [m (to-homogen m)
          table-type (.getNumericType m)]
      (println table-type (.getNumericType m))
      (println (type m))
      (cond
        (or (= Double table-type)
            (= Double/TYPE table-type))
        :float64
        (or (= Float table-type)
            (= Float/TYPE table-type))
        :float32
        :else
        (throw (ex-info (format "Unrecognized numeric type: %s"
                                (.getNumericType m))
                        {:numeric-type (.getNumericType m)})))))
  java-primitive/PToArray
  (->array [m] nil)
  (->array-copy [m]
    (let [m (to-homogen m)]
      (if (.getNumericType m)
        (case (dtype-base/get-datatype m)
          :float64 (.getDoubleArray m)
          :float32 (.getFloatArray m))
        (.getDoubleArray m))))

  mp/PDoubleArrayOutput
  (to-double-array [item]
    (let [new-ary (java-primitive/->array-copy item)]
      (if (= :float64 (dtype-base/get-datatype item))
        new-ary
        (java-primitive/make-array-of-type :float64 new-ary
                                           {:unchecked? true}))))
  (as-double-array [item] nil))


(defn ->array-of-type
  [num-table dtype]
  (let [num-table (to-homogen num-table)]
    (case dtype
      :float32 (.getFloatArray num-table)
      :float64 (.getFloatArray num-table))))


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


(defn tensor->table
  ^HomogenNumericTable [tens-data ^DaalContext daal-context]
  (let [tens-shape (ct/shape tens-data)
        _ (when (> (count tens-shape) 2)
            (throw (ex-info "Tables cannot have more than 2 dims"
                            {:tensor-shape (ct/shape tens-data)})))
        [n-rows n-cols] (case (count tens-shape)
                          2 tens-shape
                          1 [1 (first tens-shape)])
        n-rows (long n-rows)
        n-cols (long n-cols)
        table-data (or (dtype/->array tens-data)
                       (dtype/->array-copy tens-data))]
    (case (dtype/get-datatype tens-data)
      :float64 (HomogenNumericTable. daal-context ^doubles table-data n-cols n-rows)
      :float32 (HomogenNumericTable. daal-context ^floats table-data n-cols n-rows))))


(defn table->tensor
  [table datatype]
  (-> (->array-of-type table datatype)
      (ct/in-place-reshape (ct/shape table))))


(defn dataset->row-major-homogen-table
  [dataset daal-context datatype]
  (let [[n-cols n-rows] (ct/shape dataset)]
    (-> (ds-tens/dataset->row-major-tensor dataset datatype)
        (tensor->table daal-context))))
