(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/impl/layer/SparkDl4jLayer.html"}
    dl4clj.spark.dl4j-layer
  (:import [org.deeplearning4j.spark.impl.layer SparkDl4jLayer])
  (:require [datavec.api.records.interface :refer [reset-rr!]]))

(defn new-spark-dl4j-layer
  "creates a layer to be trained by spark given a nn-conf.

  analogous to dl4clj.nn.layers.layer-creation but in the context of spark

  :spark-context (sc), the spark context.
   org.apache.spark.api.java.JavaSparkContext or
   org.apache.spark.SparkContext

  :nn-conf (nn), the configuration for a single layer neural network
   - see: dl4clj.nn.conf.builders.nn-conf-builder"
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

(defn fit-spark-layer!
  "fit the spark-layer

  2 options for fitting the data
   1) supply :spark-context and :rdd
   2) supply :path :label-dix :record-reader

  :spark-context (org.apache.spark.api.java.JavaSparkContext) the spark context

  :rdd (JavaRDD<org.apache.spark.mllib.regression.LabeledPoint>) the data

  :path (str), the path to a org.deeplearning4j.spark context text file

  :label-idx (int), the index of the label

  :record-reader (record-reader), a record reader containing the data you want to fit

  this fn checks to see if the spark-context is supplied (option 1) and if not, tries to
   to load the spark context text file from path (option 2)
    - either way, the fit layer is returned"
  [& {:keys [spark-layer spark-context rdd path label-idx record-reader]
      :as opts}]
  (if (contains? opts :spark-context)
    (.fit spark-layer spark-context rdd)
    (.fit spark-layer path label-idx (reset-rr! record-reader))))

(defn fit-spark-layer-with-ds!
  "fit the spark layer with a rdd which contains a DataSet

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>), a java RDD which contains a data-set

  returns the fit layer"
  [& {:keys [spark-layer rdd]}]
  (.fitDataSet spark-layer rdd))

(defn predict
  "predict the label for a given feature matrix or vector

  :spark-data (matrix or vector). the data to be fed into the layer
   - org.apache.spark.mllib.linalg.Matrix or org.apache.spark.mllib.linalg.Vector

  the matrix or vector is fed through the layer and activations collected and reported"
  [& {:keys [spark-layer spark-data]}]
  (.predict spark-layer spark-data))
