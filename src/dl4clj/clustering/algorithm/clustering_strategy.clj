(ns dl4clj.clustering.algorithm.clustering-strategy
  (:import [org.deeplearning4j.clustering.algorithm.strategy ClusteringStrategy]))

(defn end-when-dist-variation-rate-less-than
  [clustering-strategy rate]
  (.endWhenDistributionVariationRateLessThan clustering-strategy rate))

(defn end-when-iteration-count-equals
  [clustering-strategy max-iterations]
  (.endWhenIterationCountEquals clustering-strategy max-iterations))

(defn get-distance-fn
  [clustering-strategy]
  (.getDistanceFunction clustering-strategy))

(defn get-initial-cluster-count
  [clustering-strategy]
  (.getInitialClusterCount clustering-strategy))

(defn get-terminal-condition
  [clustering-strategy]
  (.getTerminationCondition clustering-strategy))

(defn get-clustering-strategy-type
  [clustering-strategy]
  (.getType clustering-strategy))

(defn )
