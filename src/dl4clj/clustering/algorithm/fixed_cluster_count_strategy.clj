(ns dl4clj.clustering.algorithm.fixed-cluster-count-strategy
  (:import [org.deeplearning4j.clustering.algorithm.strategy FixedClusterCountStrategy]))

(defn set-up
  [& {:keys [cluster-count distance-fn]}]
  (.setup cluster-count distance-fn))
