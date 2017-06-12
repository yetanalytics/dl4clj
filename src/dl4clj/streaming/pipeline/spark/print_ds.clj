(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/streaming/pipeline/spark/PrintDataSet.html"}
    dl4clj.streaming.pipline.spark.print-ds
  (:import [org.deeplearning4j.streaming.pipeline.spark PrintDataSet]))

(defn new-print-ds-constructor
  []
  (PrintDataSet.))

(defn spark-call!
  "will need to test to determine this doc string"
  [& {:keys [java-rdd-ds print-ds-constructor]}]
  (doto print-ds-constructor (.call java-rdd-ds)))
