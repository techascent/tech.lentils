(ns tech.libs.daal.context
  (:require [tech.resource :as resource])
  (:import [com.intel.daal.services DaalContext]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn create-daal-context
  "Create a stack-based daal resource"
  []
  (let [retval (DaalContext.)]
    (resource/track retval #(.dispose retval) #{:stack})))


(defmacro with-daal-context
  [& body]
  `(resource/stack-resource-context
    (let [~'daal-context (create-daal-context)]
      ~@body)))
