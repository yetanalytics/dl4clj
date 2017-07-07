(ns dl4clj.clustering.algorithm.clustering-optimization
  (:import [org.deeplearning4j.clustering.algorithm.optimisation
            ClusteringOptimization
            ClusteringOptimizationType])
  (:require [dl4clj.constants :refer [value-of]]))

(defn new-clustering-optimization
  [& {:keys [optimization-type value]}]
  (ClusteringOptimization. (value-of {:clustering-optimization optimization-type})
                           value))
;; used by apply optimization in clustering utils

(defn get-type
  [clustering-optimization]
  (.getType clustering-optimization))

(defn get-value
  [clustering-optimization]
  (.getValue clustering-optimization))

(defn set-type!
  [& {:keys [clustering-optimization optimization-type]}]
  (doto clustering-optimization (.setType (value-of optimization-type))))

(defn set-value!
  [& {:keys [clustering-optimization value]}]
  (doto clustering-optimization (.setValue value)))
