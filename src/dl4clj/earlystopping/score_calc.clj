(ns ^{:doc "Given a DataSetIterator: calculate the total loss for the model on that data set. Typically used to calculate the loss on a test set.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculator.html"}
    dl4clj.earlystopping.score-calc
  (:import [org.deeplearning4j.earlystopping.scorecalc
            DataSetLossCalculator])
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

;; the sister fn for computational graphs is not implemented as cgs are not implemented...yet

;; spark versions will be implemented later
;; https://deeplearning4j.org/doc/org/deeplearning4j/spark/earlystopping/SparkDataSetLossCalculator.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/spark/earlystopping/SparkLossCalculatorComputationGraph.html
