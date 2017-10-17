(ns dl4clj.clustering.iteration.info
  (:import [org.deeplearning4j.clustering.algorithm.iteration IterationInfo]))

(defn new-iteration-info
  [& {:keys [idx cluster-set-info]
      :as opts}]
  (assert (contains? opts :idx) "you must provide an index for the info to be about")
  (if (contains? opts :cluster-set-info)
    (IterationInfo. idx cluster-set-info)
    (IterationInfo. idx)))

(defn get-cluster-set-info
  [iteration-info]
  (.getClusterSetInfo iteration-info))

(defn get-index
  [iteration-info]
  (.getIndex iteration-info))

(defn is-strategy-applied?
  [iteration-info]
  (.isStrategyApplied iteration-info))

(defn set-cluster-set-info!
  [& {:keys [iteration-info cluster-set-info]}]
  (doto iteration-info (.setClusterSetInfo cluster-set-info)))

(defn set-index!
  [& {:keys [iteration-info idx]}]
  (doto iteration-info (.setIndex idx)))

(defn set-strategy-applied!
  [& {:keys [iteration-info optimization-applied?]}]
  (doto iteration-info (.setStrategyApplied optimization-applied?)))
