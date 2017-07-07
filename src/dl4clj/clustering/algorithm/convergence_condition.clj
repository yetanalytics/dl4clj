(ns dl4clj.clustering.algorithm.convergence-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition ConvergenceCondition]))

(defn distribution-variation-rate-less-than
  "creates a convergence condition obj"
  [rate]
  (ConvergenceCondition/distributionVariationRateLessThan rate))

;; add in is-convergence-satisfied?

;; move this to a different dir
