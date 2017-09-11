(ns ^{:doc "A provider for an DataSet rdd.

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/data/DataSetProvider.html"}
    dl4clj.spark.data.dataset-provider
  (:import [org.deeplearning4j.spark.data DataSetProvider]))

(defn spark-context-to-dataset-rdd
  ;; revisit, is this a DataSetProvider or an actual data-set?
  "Return an rdd of type dataset"
  ;; not sure what this is here
  ;; docs dont specify any implementing classes
  ;; seems like another case where gen-class would need to be used
  [& {:keys [dataset spark-context]}]
  (.data dataset spark-context))
