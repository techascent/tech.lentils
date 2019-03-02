(ns tech.libs.daal.regression-test
  (:require [tech.libs.daal.regression]
            [tech.verify.ml.regression :as verify-reg]
            [clojure.test :refer :all]))


(deftest linear-basic
  (verify-reg/basic-regression {:model-type :daal.regression/linear}))
