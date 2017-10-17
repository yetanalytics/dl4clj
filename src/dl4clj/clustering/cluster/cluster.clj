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

(defn get-cluster-id
  [cluster]
  (.getId cluster))

(defn get-cluster-label
  [cluster]
  (.getLabel cluster))

(defn get-point
  [& {:keys [cluster point-id]}]
  (.getPoint cluster point-id))

(defn get-points
  [cluster]
  (.getPoints cluster))

(defn empty-cluster?
  [cluster]
  (.isEmpty cluster))

(defn remove-point!
  ;; doesnt work if you set a clusters label and id
  ;; need to look into this even tho should
  ;; promote imutability whenever possible
  [& {:keys [cluster point-id]}]
  (doto cluster (.removePoint point-id)))

(defn remove-points!
  ;; doesnt work if you set a clusters label and id
  ;; need to look into this even tho should
  ;; promote imutability whenever possible
  [cluster]
  (doto cluster (.removePoints)))

(defn set-center!
  [& {:keys [cluster point]}]
  (doto cluster (.setCenter point)))

(defn set-cluster-id!
  [& {:keys [cluster id]}]
  (doto cluster (.setId id)))

(defn set-cluster-label!
  [& {:keys [cluster label]}]
  (doto cluster (.setLabel label)))

#_(defn set-points!
  ;; causes a down stream error
  [& {:keys [cluster points]}]
  (doto cluster (.setPoints points)))

(defn set-points!
  [& {:keys [cluster points move-center?]
      :or {move-center? true}}]
  (loop [c cluster
         ps! points]
    (if (empty? ps!)
      c
      (recur (add-point! :cluster c :point (first ps!) :move-center? move-center?)
             (rest ps!)))))
