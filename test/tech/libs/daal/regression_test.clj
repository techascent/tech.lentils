(ns tech.libs.daal.regression-test
  (:require [tech.libs.daal.regression]
            [tech.verify.ml.regression :as verify-reg]
            [clojure.test :refer :all]))


(deftest linear-basic
  (verify-reg/basic-regression {:model-type :daal.regression/linear}))


(deftest ridge-basic
  (verify-reg/basic-regression {:model-type :daal.regression/ridge}))


(deftest gradient-boosted-trees
  (verify-reg/basic-regression {:model-type :daal.regression/gradient-boosted-trees}))


;; Gridsearching the boosted trees yields crashes for me.
;; (deftest gradient-boosted-trees-gridsearch
;;   (verify-reg/auto-gridsearch-simple {:model-type :daal.regression/gradient-boosted-trees
;;                                       :gridsearch-depth 20
;;                                       :mse-loss 1000}))
