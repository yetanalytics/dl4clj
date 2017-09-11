(ns ^{:doc "name space for creating layers trainable via spark"}
    dl4clj.spark.dl4j-layer
  (:import [org.deeplearning4j.spark.impl.layer SparkDl4jLayer]))

;; update to deaulting to code

(defn new-spark-dl4j-layer
  "creates a layer to be trained by spark given a nn-conf.

  analogous to dl4clj.nn.layers.layer-creation but in the context of spark

  :spark-context (sc), the spark context.
   org.apache.spark.api.java.JavaSparkContext or
   org.apache.spark.SparkContext

  :nn-conf (nn), the configuration for a single layer neural network
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/impl/layer/SparkDl4jLayer.html"
  [& {:keys [spark-context nn-conf]}]
  (SparkDl4jLayer. spark-context nn-conf))

(defn train-spark-layer!
  "creates a new instance of SparkDl4jLayer and fits it.
   - instance created from the spark context contained within the JavaRDD and the supplied nn-conf
   - fits the new instance with the data found within the passed JavaRDD

  :rdd (JavaRDD<org.apache.spark.mllib.regression.LabeledPoint>)

  :nn-conf (nn), the configuration for a single layer nn
   - see: dl4clj.nn.conf.builders.nn-conf-builder"
  [& {:keys [nn-conf rdd]}]
  (.train rdd nn-conf))
