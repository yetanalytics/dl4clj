(ns dl4clj.clustering.condition.conditions
  (:import [org.deeplearning4j.clustering.algorithm.condition
            ConvergenceCondition FixedIterationCountCondition VarianceVariationCondition])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti termination-conditions generic-dispatching-fn)

(defmethod termination-conditions :convergence [opts]
  (let [conf (:convergence opts)
        rate (:distribution-change-rate conf)]
    (ConvergenceCondition/distributionVariationRateLessThan rate)))

(defmethod termination-conditions :fixed-iteration [opts]
  (let [conf (:fixed-iteration opts)
        n (:max-iterations conf)]
    (FixedIterationCountCondition/iterationCountGreaterThan n)))

(defmethod termination-conditions :variance-variation [opts]
  (let [conf (:variance-variation opts)
        {variation :variation
         period :period} conf]
    (VarianceVariationCondition/varianceVariationLessThan variation period)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-convergence-termination-condition
  "Creates a new point convergence termiantion condition"
  [& {:keys [^Double distribution-change-rate]
      :as opts}]
  (termination-conditions {:convergence opts}))

(defn new-iteration-count-termination-condition
  "creates a new max iterations termination condition"
  [& {:keys [^Integer max-iterations]
      :as opts}]
  (termination-conditions {:fixed-iteration opts}))

(defn new-variance-variation-termination-condition
  "creates a new distribution variance variation termination condition"
  [& {:keys [^Double variation ^Integer period]
      :as opts}]
  (termination-conditions {:variance-variation opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn is-satisfied?
  "checks to see if the termination condition is satisifed given the current iteration history"
  [& {:keys [termination-condition iteration-history]}]
  (.isSatisfied termination-condition iteration-history))
