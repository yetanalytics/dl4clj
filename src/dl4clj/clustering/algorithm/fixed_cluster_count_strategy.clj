(ns dl4clj.clustering.algorithm.fixed-cluster-count-strategy
  (:import [org.deeplearning4j.clustering.algorithm.strategy FixedClusterCountStrategy])
  (:require [dl4clj.clustering.algorithm.base-clustering-strategy :refer :all]
            [dl4clj.clustering.algorithm.clustering-strategy :refer :all]))

(defn set-up
  [cluster-count distance-fn]
  (.setup cluster-count distance-fn))
