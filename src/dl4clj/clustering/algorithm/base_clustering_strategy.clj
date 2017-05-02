(ns dl4clj.clustering.algorithm.base-clustering-strategy
  (:import [org.deeplearning4j.clustering.algorithm.strategy BaseClusteringStrategy])
  (:require [dl4clj.clustering.algorithm.clustering-strategy :refer :all]))

(defn set-initial-cluster-count!
  [& {:keys [cluster-strat cluster-count]}]
  (doto cluster-strat (.setInitialClusterCount cluster-count)))

(defn set-distance-fn!
  [& {:keys [clustering-strat distance-fn]}]
  (doto clustering-strat (.setDistanceFunction distance-fn)))

(defn set-allow-empty-clusters!
  [& {:keys [clustering-strat allow?]}]
  (doto clustering-strat (.setAllowEmptyClusters allow?)))

(defn get-optimization-phase-condition
  [clustering-strat]
  (.getOptimizationPhaseCondition clustering-strat))
