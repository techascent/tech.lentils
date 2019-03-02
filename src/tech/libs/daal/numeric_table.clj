(ns tech.libs.daal.numeric-table
  (:require [tech.datatype.base :as dtype-base]
            [tech.datatype.java-primitive :as java-primitive]
            [clojure.core.matrix.protocols :as mp])
  (:import [com.intel.daal.data_management.data
            NumericTable HomogenNumericTable]))


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
