(ns tech.libs.daal.pca-test
  (:require [clojure.test :refer :all]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.tensor :as ds-tens]
            [tech.compute.tensor :as ct]
            [tech.datatype :as dtype]
            [tech.libs.daal.pca :as daal-pca])
  (:import [smile.projection PCA]))



;;Smile PCA is working fine.  Intel PCA, however...not working fine...
(deftest basic-pca-test
  (let [src-tens (ct/->tensor (->> (repeatedly 25 rand)
                                   shuffle
                                   (partition 5)))
        src-dataset (ds-tens/row-major-tensor->dataset src-tens)
        pca-info (daal-pca/pca-dataset src-dataset)]

    (println "TENS-INPUT" src-tens)

    (is (= #{:means :variances :eigenvectors :eigenvalues} (set (keys pca-info))))
    (println "DAAL Variances:" (dtype/->vector (:variances pca-info)))
    (println "DAAL Eigenvectors:" (:eigenvectors pca-info))

    (println "SMILE!!!")
    (let [array-of-arrays (->> (range 5)
                               (map #(dtype/make-array-of-type
                                      :float64
                                      (ct/select src-tens % :all)))
                               (into-array (Class/forName "[D")))
          smile-pca (PCA. array-of-arrays)
          eigenvectors (-> (.getLoadings smile-pca)
                           (.data)
                           (ct/in-place-reshape [5 5]))]
      (println "Smile Variances" (dtype/->vector (.getVariance smile-pca)))
      (println "Smile Eigenvectors " eigenvectors))))
