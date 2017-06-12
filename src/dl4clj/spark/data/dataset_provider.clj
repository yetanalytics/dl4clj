(ns ^{:doc "A provider for an DataSet rdd.

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/data/DataSetProvider.html"}
    dl4clj.spark.data.dataset-provider
  (:import [org.deeplearning4j.spark.data DataSetProvider]))

(defn spark-context-to-dataset-rdd
  "Return an rdd of type dataset"
  ;; not sure what this is here
  ;; docs dont specify any implementing classes
  [& {:keys [this spark-context]}]
  (.data this spark-context))
