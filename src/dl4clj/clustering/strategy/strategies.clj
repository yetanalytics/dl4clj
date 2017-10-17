(ns dl4clj.clustering.strategy.strategies
  (:import [org.deeplearning4j.clustering.algorithm.strategy FixedClusterCountStrategy
            OptimisationStrategy])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method for obj creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti clustering-strategy generic-dispatching-fn)

(defmethod clustering-strategy :fixed-cluster-count [opts]
  (let [conf (:fixed-cluster-count opts)
        {cluster-count :cluster-count
         distance-fn :distance-fn} conf]
    (FixedClusterCountStrategy/setup cluster-count distance-fn)))

(defmethod clustering-strategy :optimization [opts]
  (let [conf (:optimization opts)
        {cluster-count :cluster-count
         distance-fn :distance-fn} conf]
    (OptimisationStrategy/setup cluster-count distance-fn)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns, still need doc strings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-fixed-cluster-count-strategy
  "Creates a clustering strategy with the strategy type set to fixed-cluster-count

  :cluster-count (int), the number of clusters (in the cluster-set)
   this strategy will be applied to

  :distance-fn (str), the fn used to calculate the distance between points/clusters
   - one of: 'cosinesimilarity', 'euclidean', 'manhattan'"
  [& {:keys [cluster-count distance-fn]
      :as opts}]
  (clustering-strategy {:fixed-cluster-count opts}))

(defn new-optimization-strategy
  "Creates a clustering strategy with the strategy type set to optimization

  :cluster-count (int), the number of clusters (in the cluster-set)
   this strategy will be applied to

  :distance-fn (str), the fn used to calculate the distance between points/clusters
   - one of: 'cosinesimilarity', 'euclidean', 'manhattan'"
  [& {:keys [cluster-count distance-fn]
      :as opts}]
  (clustering-strategy {:optimization opts}))
