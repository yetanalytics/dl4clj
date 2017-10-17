(ns ^{:doc "A provider for an DataSet rdd.

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/data/DataSetProvider.html"}
    dl4clj.spark.data.dataset-provider
  (:import [org.deeplearning4j.spark.data DataSetProvider])
  (:require [dl4clj.utils :refer [obj-or-code?]]
            [clojure.core.match :refer [match]]))

;; WIP

(defn spark-context-to-dataset-rdd
  ;; revisit, is this a DataSetProvider or an actual data-set?
  "Return an rdd of type dataset"
  ;; not sure what this is here
  ;; docs dont specify any implementing classes
  ;; seems like another case where gen-class would need to be used
  [& {:keys [dataset spark-context as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:dataset (_ :guard seq?)
           :spark-context (_ :guard seq?)}]
         (obj-or-code? as-code? `(.data ~dataset ~spark-context))
         [{:dataset _
           :spark-context _}]
         `(.data ~dataset ~spark-context)))
