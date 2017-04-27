(ns dl4clj.clustering.algorithm.convergence-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition ConvergenceCondition])
  (:require [dl4clj.clustering.algorithm.clustering-algorithm-condition :refer [is-satisfied?]]))

(defn distribution-variation-rate-less-than
  "comparison method"
  [point-dist-change-rate]
  (.distributionVariationRateLessThan point-dist-change-rate))
