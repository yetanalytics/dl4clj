(ns ^{:doc "Utility fns for performing evaluation.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/EvaluationUtils.html"}
    dl4clj.eval.eval-utils
  (:import [org.deeplearning4j.eval EvaluationUtils]))

(defn new-evaluation-utils
  "creates an evaluation utils class constructor"
  []
  (EvaluationUtils.))

(defn extract-non-masked-time-steps
  "returns the original time series given the labels, the output of a model
  and the mask applied to that output"
  [& {:keys [labels predicted output-mask]}]
  (.extractNonMaskedTimeSteps labels predicted output-mask))

(defn reshape-time-series-to-2d
  "reshapes a time series to be two dimensional"
  [labels]
  (.reshapeTimeSeriesTo2d labels))
