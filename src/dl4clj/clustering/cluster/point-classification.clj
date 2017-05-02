(ns dl4clj.clustering.cluster.point-classification
  (:import [org.deeplearning4j.clustering.cluster PointClassification]))

(defn new-point-classification
  [& {:keys [cluster distance-from-center new-location?]}]
  (PointClassification. cluster distance-from-center new-location?))

(defn get-cluster
  [point-classification]
  (.getCluster point-classification))

(defn get-distance-from-center
  [point-classification]
  (.getDistanceFromCenter point-classification))

(defn new-location?
  [point-classification]
  (.isNewLocation point-classification))

(defn set-cluster!
  [& {:keys [point-classification cluster]}]
  (doto point-classification (.setCluster cluster)))

(defn set-distance-from-center!
  [& {:keys [point-classification distance-from-center]}]
  (doto point-classification (.setDistanceFromCenter distance-from-center)))

(defn set-new-location!
  [& {:keys [point-classification new-location?]}]
  (doto point-classification (.setNewLocation new-location?)))
