(ns dl4clj.clustering.algorithm.base-clustering-algorithm
  (:import [org.deeplearning4j.clustering.algorithm BaseClusteringAlgorithm])
  (:require [dl4clj.utils :refer [type-checking]]))
;; implement clustering specific types in util
;; apply type-checking to the fns here
(defn apply-to
  "applies a clustering algorithm to a collection of points"
  [& {:keys [cluster-algo points]}]
  (.applyTo cluster-algo points))

(defn set-up
  "sets up a clustering strategy for a clustering algorithm"
  [clustering-strat]
  (.setup clustering-strat))
