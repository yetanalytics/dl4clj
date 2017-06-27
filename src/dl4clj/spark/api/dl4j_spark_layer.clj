(ns dl4clj.spark.api.dl4j-spark-layer
  (:import [org.deeplearning4j.spark.impl.layer SparkDl4jLayer])
  (:require [dl4clj.datasets.api.record-readers :refer [reset-rr!]]))

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
