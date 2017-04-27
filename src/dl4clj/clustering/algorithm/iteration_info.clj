(ns dl4clj.clustering.algorithm.iteration-info
  (:import [org.deeplearning4j.clustering.algorithm.iteration IterationInfo]))

(defn new-iteration-info
  [& {:keys [idx cluster-set-info]
      :as opts}]
  (assert (contains? opts :idx) "you must provide an index for the info to be about")
  (if (contains? opts :cluster-set-info)
    (IterationInfo. idx cluster-set-info)
    (IterationInfo. idx)))

(defn get-cluster-set-info
  [cluster-set]
  (.getClusterSetInfo cluster-set))

(defn get-index
  [cluster-set]
  (.getIndex cluster-set))

(defn is-strategy-applied?
  [cluster-set]
  (.isStrategyApplied cluster-set))

(defn set-cluster-set-info!
  [cluster-set cluster-set-info]
  (doto cluster-set (.setClusterSetInfo cluster-set-info)))

(defn set-index!
  [cluster-set idx]
  (doto cluster-set (.setIndex idx)))

(defn set-strategy-applied!
  [cluster-set optimization-applied?]
  (doto cluster-set (.setStrategyApplied optimization-applied?)))
