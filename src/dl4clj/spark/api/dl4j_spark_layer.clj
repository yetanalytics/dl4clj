(ns dl4clj.spark.api.dl4j-spark-layer
  (:import [org.deeplearning4j.spark.impl.layer SparkDl4jLayer])
  (:require [dl4clj.datasets.api.record-readers :refer [reset-rr!]]
            [clojure.core.match :refer [match]]))

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
  (match [opts]
         [{:spark-layer (_ :guard seq?)
           :spark-context (_ :guard seq?)
           :rdd (_ :guard seq?)}]
         `(.fit ~spark-layer ~spark-context ~rdd)
         [{:spark-layer _
           :spark-context _
           :rdd _}]
         (.fit spark-layer spark-context rdd)
         [{:spark-layer (_ :guard seq?)
           :path (:or (_ :guard string?)
                      (_ :guard seq?))
           :label-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           record-reader (_ :guard seq?)}]
         `(.fit ~spark-layer ~path (int ~label-idx) ~record-reader)
         [{:spark-layer _
           :path _
           :label-idx _
           record-reader _}]
         (.fit spark-layer path label-idx (reset-rr! record-reader))))

(defn fit-spark-layer-with-ds!
  "fit the spark layer with a rdd which contains a DataSet

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>), a java RDD which contains a data-set

  returns the fit layer"
  [& {:keys [spark-layer rdd]
      :as opts}]
  (match [opts]
         [{:spark-layer (_ :guard seq?)
           :rdd (_ :guard seq?)}]
         `(.fitDataSet ~spark-layer ~rdd)
         :else
         (.fitDataSet spark-layer rdd)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; remember to look into creation of the spark matrices/vectors
;; should have a conversion fn for clj-vector/matrix -> spark-vector/matrix
(defn predict
  "predict the label for a given feature matrix or vector

  :spark-data (matrix or vector). the data to be fed into the layer
   - org.apache.spark.mllib.linalg.Matrix or org.apache.spark.mllib.linalg.Vector

  the matrix or vector is fed through the layer and activations collected and reported"
  [& {:keys [spark-layer spark-data]
      :as opts}]
  (match [opts]
         [{:spark-layer (_ :guard seq?)
           :spark-data (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         `(.predict ~spark-layer ~spark-data)
         :else
         (.predict spark-layer spark-data)))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
