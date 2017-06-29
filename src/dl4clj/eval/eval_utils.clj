(ns ^{:doc "Utility fns for performing evaluation.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/EvaluationUtils.html"}
    dl4clj.eval.eval-utils
  (:import [org.deeplearning4j.eval EvaluationUtils])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn extract-non-masked-time-steps
  "returns the original time series given the labels, the output of a model
  and the mask applied to that output"
  [& {:keys [labels predicted output-mask]}]
  (EvaluationUtils/extractNonMaskedTimeSteps
   (vec-or-matrix->indarray labels)
   (vec-or-matrix->indarray predicted)
   (vec-or-matrix->indarray output-mask)))

(defn reshape-time-series-to-2d
  "reshapes a time series to be two dimensional"
  [labels]
  (EvaluationUtils/reshapeTimeSeriesTo2d (vec-or-matrix->indarray labels)))
