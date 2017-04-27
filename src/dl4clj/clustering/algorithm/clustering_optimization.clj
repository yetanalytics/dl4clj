(ns dl4clj.clustering.algorithm.clustering-optimization
  (:import [org.deeplearning4j.clustering.algorithm.optimisation
            ClusteringOptimization
            ClusteringOptimizationType])
  (:require [dl4clj.utils :refer [type-checking]]))

(defn value-of
  [optimization-type]
  (cond
    (= optimization-type :minimize-avg-point-to-center)
    (ClusteringOptimizationType/MINIMIZE_AVERAGE_POINT_TO_CENTER_DISTANCE)
    (= optimization-type :minimize-avg-point-to-point)
    (ClusteringOptimizationType/MINIMIZE_AVERAGE_POINT_TO_POINT_DISTANCE)
    (= optimization-type :minimize-max-point-to-center)
    (ClusteringOptimizationType/MINIMIZE_MAXIMUM_POINT_TO_CENTER_DISTANCE)
    (= optimization-type :minimize-max-point-to-point)
    (ClusteringOptimizationType/MINIMIZE_MAXIMUM_POINT_TO_POINT_DISTANCE)
    (= optimization-type :minimize-per-cluster-point-count)
    (ClusteringOptimizationType/MINIMIZE_PER_CLUSTER_POINT_COUNT)
    :else
    (assert false "you must supply a valid optimization-type")))

(defn new-clustering-optimization
  [optimization-type value]
  (ClusteringOptimization. (value-of optimization-type)
                           value))

(defn get-type
  [clustering-optimization]
  (.getType clustering-optimization))

(defn get-value
  [clustering-optimization]
  (.getValue clustering-optimization))

(defn set-type!
  [clustering-optimization optimization-type]
  (doto clustering-optimization (.setType (value-of optimization-type))))

(defn set-value!
  [clustering-optimization value]
  (doto clustering-optimization (.setValue value)))
