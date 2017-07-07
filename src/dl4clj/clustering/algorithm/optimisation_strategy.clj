(ns dl4clj.clustering.algorithm.optimisation-strategy
  (:import [org.deeplearning4j.clustering.algorithm.strategy OptimisationStrategy]))

(defn new-optimisation-strat
  [& {:keys [initial-cluster-count distance-fn]}]
  (.setup initial-cluster-count distance-fn))

(defn get-clustering-optimization-val
  [clustering-optim]
  (.getClusteringOptimizationValue clustering-optim))

(defn clustering-optimization-type?
  [& {:keys [clustering-optim optim-type]}]
   (.isClusteringOptimizationType clustering-optim optim-type))

(defn is-optimization-applicable-now?
  [& {:keys [clustering-optim iteration-history]}]
  (.isOptimizationApplicableNow clustering-optim iteration-history))

(defn is-optimization-defined?
  [clustering-optim]
  (.isOptimizationDefined clustering-optim))

(defn optimize
  [& {:keys [clustering-optim optim-type value]}]
  (.optimize clustering-optim optim-type value))

(defn optimize-when-iteration-count-multiple-of
  [& {:keys [clustering-optim multiple]}]
  (.optimizeWhenIterationCountMultipleOf clustering-optim multiple))

(defn optimize-when-point-dist-variation-rate-less-than
  [& {:keys [clustering-optim rate]}]
  (.optimizeWhenPointDistributionVariationRateLessThan clustering-optim rate))
