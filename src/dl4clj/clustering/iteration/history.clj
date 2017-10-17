(ns dl4clj.clustering.iteration.history
  (:import [org.deeplearning4j.clustering.algorithm.iteration IterationHistory]))

(defn new-iteration-history
  "invokes the IterationHistory constructor"
  []
  (IterationHistory.))

(defn get-iteration-count
  [iteration-history]
  (.getIterationCount iteration-history))

(defn get-iteration-info
  [& {:keys [iteration-history iteration-idx]}]
  (.getIterationInfo iteration-history iteration-idx))

(defn get-most-recent-cluster-set-info
  [iteration-history]
  (.getMostRecentClusterSetInfo iteration-history))

(defn get-most-recent-iteration-info
  [iteration-history]
  (.getMostRecentIterationInfo iteration-history))

(defn set-iteration-infos!
  [& {:keys [iteration-history iterations-infos]}]
  (doto iteration-history (.setIterationsInfos iterations-infos)))
