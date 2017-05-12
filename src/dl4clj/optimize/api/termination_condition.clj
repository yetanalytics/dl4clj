(ns ^{:doc "interface for determing if termination is needed
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/TerminationCondition.html"}
    dl4clj.optimize.api.termination-condition
  (:import [org.deeplearning4j.optimize.api TerminationCondition]))

(defn terminate?
  "Whether to terminate based on the given metadata

  :cost (double), the current cost of the optimizer

  :old-cost (double), the old cost of the optimizer

  :other-params (object), some other meta data in the form of a java object"
  [& {:keys [optim cost old-cost other-params]}]
  (.terminate optim cost old-cost other-params))
