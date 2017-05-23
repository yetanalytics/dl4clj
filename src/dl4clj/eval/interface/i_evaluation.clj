(ns ^{:doc "A general purpose interface for evaluating neural networks

see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/IEvaluation.html"}
    dl4clj.eval.interface.i-evaluation
  (:import [org.deeplearning4j.eval IEvaluation])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn merge!
  "merges objects that implemented the IEvaluation interface

  evaler and other-evaler can be evaluations, ROCs or MultiClassRocs"
  [& {:keys [evaler other-evaler]}]
  (doto evaler (.merge other-evaler)))

(defn eval-time-series!
  "evalatues a time series given labels and predictions.

  labels-mask is optional and only applies when there is a mask"
  [& {:keys [labels predicted labels-mask evaler]
      :as opts}]
  (cond (contains? opts :labels-mask)
        (doto evaler (.evalTimeSeries labels predicted labels-mask))
        (false? (contains? opts :labels-mask))
        (doto evaler (.evalTimeSeries labels predicted))
        :else
        (assert false "you must supply labels-mask and/or labels and predicted values")))

(defn eval!
  "evaluate the output of a network.

  :labels (INDArray), the actual labels of the data (target labels)

  :network-predictions (INDArray), the output of the network

  :mask-array (INDArray), the mask array for the data if there is one

  :record-meta-data (coll) meta data that extends java.io.Serializable

  NOTE: for evaluating classification problems, use eval-classification! in
   dl4clj.eval.evaluation, (when the evaler is created by new-classification-evaler)"
  [& {:keys [labels network-predictions mask-array record-meta-data evaler]
      :as opts}]
  (assert (contains? opts :evaler) "you must provide an evaler to evaluate a classification task")
  (cond (contains-many? opts :labels :network-predictions :record-meta-data)
        (doto evaler (.eval labels network-predictions (into '() record-meta-data)))
        (contains-many? opts :labels :network-predictions :mask-array)
        (doto evaler (.eval labels network-predictions mask-array))
        (contains-many? opts :labels :network-predictions)
        (doto evaler (.eval labels network-predictions))
        :else
        (assert false "you must supply an evaler, the correct labels and the network predicted labels")))
