(ns dl4clj.clustering.algorithm.clustering-algorithm-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition ClusteringAlgorithmCondition]))

(defn is-satisfied?
  [this iteration-history]
  (.isSatisfied this iteration-history))
