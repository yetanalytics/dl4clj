(ns dl4clj.clustering.cluster.cluster
  (:import [org.deeplearning4j.clustering.cluster Cluster]))

(defn new-cluster
  [& {:keys [center-point distance-fn]
      :as opts}]
  (if (empty? opts)
    (Cluster.)
    (Cluster. center-point distance-fn)))

(defn add-point!
  [& {:keys [cluster point move-center?]
      :as opts}]
  (if (contains? opts :move-center?)
    (doto cluster (.addPoint point move-center?))
    (doto cluster (.addPoint point))))

(defn get-center
  [cluster]
  (.getCenter cluster))

(defn get-distance-to-center
  [& {:keys [cluster point]}]
  (.getDistanceToCenter cluster point))

(defn get-id
  [cluster]
  (.getId cluster))

(defn get-label
  [cluster]
  (.getLabel cluster))

(defn get-point
  [& {:keys [cluster point-id]}]
  (.getPoint cluster point-id))

(defn get-points
  [cluster]
  (.getPoints cluster))

(defn is-empty?
  [cluster]
  (.isEmpty cluster))

(defn remove-point!
  [& {:keys [cluster point-id]}]
  (doto cluster (.removePoint point-id)))

(defn remove-points!
  [cluster]
  (doto cluster (.removePoints)))

(defn set-center!
  [& {:keys [cluster point]}]
  (doto cluster (.setCenter point)))

(defn set-id!
  [& {:keys [cluster id]}]
  (doto cluster (.setId id)))

(defn set-label!
  [& {:keys [cluster label]}]
  (doto cluster (.setLabel label)))

(defn set-points!
  [& {:keys [cluster points]}]
  (doto cluster (.setPoints points)))
