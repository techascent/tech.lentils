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


(defonce ^:dynamic *bound-context* nil)

(defn context
  []
  (if-let [retval *bound-context*]
    retval
    (throw (ex-info "There is no daal context currently bound" {}))))


(defmacro with-daal-context
  [& body]
  `(resource/stack-resource-context
    (let [~'daal-context (or *bound-context* (create-daal-context))]
      (with-bindings {#'*bound-context* ~'daal-context}
        ~@body))))
