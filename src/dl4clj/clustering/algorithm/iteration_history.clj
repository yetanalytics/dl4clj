(ns dl4clj.clustering.algorithm.iteration-history
  (:import [org.deeplearning4j.clustering.algorithm.iteration IterationHistory]))

(defn new-iteration-history
  "invokes the IterationHistory constructor"
  []
  (IterationHistory.))

(defn get-iteration-count
  [iteration-history]
  (.getIterationCount iteration-history))

(defn get-iteration-info
  [iteration-history iteration-idx]
  (.getIterationInfo iteration-history iteration-idx))

(defn get-iteration-infos
  ;; look into if this is still a thing
  [iteration-history]
  (.getIterationInfos iteration-history))

(defn get-most-recent-cluster-set-info
  [cluster-set]
  (.getMostRecentClusterSetInfo cluster-set))

(defn get-most-recent-iteration-info
  [iteration-history]
  (.getMostRecentIterationInfo iteration-history))

(defn set-iteration-infos!
  [iteration-history iterations-infos]
  (doto iteration-history (.setIterationsInfos iterations-infos)))
