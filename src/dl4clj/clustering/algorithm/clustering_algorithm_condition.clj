(ns dl4clj.clustering.algorithm.clustering-algorithm-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition ClusteringAlgorithmCondition]))

(defn is-satisfied?
  [& {:keys [termination-condition iteration-history]}]
  (.isSatisfied termination-condition iteration-history))

;; checks to see if the termination condition is satisifed given the current iteration history
;; used in the private fn iterations within baseClusteringAlgorithm
