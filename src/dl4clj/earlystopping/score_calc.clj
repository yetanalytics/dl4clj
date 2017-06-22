(ns ^{:doc "Given a DataSetIterator: calculate the total loss for the model on that data set. Typically used to calculate the loss on a test set.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculator.html"}
    dl4clj.earlystopping.score-calc
  (:import [org.deeplearning4j.earlystopping.scorecalc
            DataSetLossCalculator ScoreCalculator]
           [org.deeplearning4j.spark.earlystopping SparkDataSetLossCalculator])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti score-calc generic-dispatching-fn)

(defmethod score-calc :dataset-loss [opts]
  (let [conf (:dataset-loss opts)
        {iter :ds-iter
         avg? :average?} conf]
    (DataSetLossCalculator. iter avg?)))

(defmethod score-calc :spark-ds-loss [opts]
  (let [conf (:spark-ds-loss opts)
        {rdd :rdd
         average? :average?
         sc :spark-context} conf]
    (SparkDataSetLossCalculator. rdd average? sc)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-data-set-loss-calculator
  "used to calc the total loss for the model on the supplied dataset

  :ds-iter (dataset-iterator), supplies the data to calc the loss on.
   - see: datavec.api.records.readers

  :average? (boolean), Whether to return the average (sum of loss / N) or just (sum of loss)"
  [& {:keys [ds-iter average?]
      :as opts}]
  (score-calc {:dataset-loss opts}))

(defn new-spark-ds-loss-calculator
  "Score calculator to calculate the total loss for the MLN
  on the provided JavaRDD data set (test set)

  :rdd (JavaRDD), dataset to calc the score for

  :average? (boolean), Whether to return the average (sum of loss / N),
                       or just the sum of the loss

  :spark-context (org.apache.spark.SparkContext), the spark context"
  [& {:keys [test-rdd average? spark-context]
      :as opts}]
  (score-calc {:spark-ds-loss opts}))

(defn calculate-score
  "used to calculate a score for a neural network.
  For example, the loss function, test set accuracy, F1"
  [& {:keys [score-calculator mln]}]
  (.calculateScore score-calculator mln))





;; the sister fn for computational graphs is not implemented as cgs are not implemented...yet

;; comp-graph versions will be implemented later
;; https://deeplearning4j.org/doc/org/deeplearning4j/spark/earlystopping/SparkLossCalculatorComputationGraph.html
