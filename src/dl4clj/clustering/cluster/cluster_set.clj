(ns dl4clj.clustering.cluster.cluster-set
  (:import [org.deeplearning4j.clustering.cluster ClusterSet])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-cluster-set
  [& {:keys [distance-fn]
      :as opts}]
  (if (contains? opts :distance-fn)
    (ClusterSet. distance-fn)
    (ClusterSet.)))

(defn add-new-cluster-with-center
  [cluster-set center-point]
  (.addNewClusterWithCenter cluster-set center-point))

(defn classify-point
  [& {:keys [cluster-set point move-cluster-center?]
      :as opts}]
  (assert (contains-many? opts :cluster-set :point)
          "you must provide a cluster and a point to classify")
  (if (contains? opts :move-cluster-center?)
    (.classifyPoint cluster-set point move-cluster-center?)
    (.classifyPoint cluster-set point)))

(defn classify-points!
  [& {:keys [cluster-set points move-cluster-center?]
   :as opts}]
  (assert (contains-many? opts :cluster-set :points)
          "you must provide a cluster and a list of points to classify")
  (if (contains? opts :move-cluster-center?)
    (doto cluster-set (.classifyPoints points move-cluster-center?))
    (doto cluster-set (.classifyPoints points))))

(defn get-accumulation
  [cluster-set]
  (.getAccumulation cluster-set))

(defn get-cluster-center
  [cluster-set cluster-id]
  (.getClusterCenter cluster-set cluster-id))

(defn get-cluster
  [cluster-set cluster-id]
  (.getCluster cluster-set cluster-id))

(defn get-cluster-center-id
  [cluster-set cluster-id]
  (.getClusterCenterId cluster-set cluster-id))

(defn get-cluster-count
  [cluster-set]
  (.getClusterCount cluster-set))

(defn get-clusters
  [cluster-set]
  (.getClusters cluster-set))

(defn get-distance
  [cluster-set point-1 point-2]
  (.getDistance cluster-set point-1 point-2))

(defn get-distance-from-nearest-cluster
  [cluster-set point]
  (.getDistanceFromNearestCluster cluster-set point))

(defn get-most-populated-clusters
  [cluster-set n]
  (.getMostPopulatedClusters cluster-set n))

(defn get-point-distribution
  [cluster-set]
  (.getPointDistribution cluster-set))

(defn nerest-cluster-to-point
  [cluster-set point]
  (.nearestCluster cluster-set point))

(defn remove-empty-clusters!
  [cluster-set]
  (.removeEmptyClusters cluster-set))

(defn remove-points!
  [cluster-set]
  (doto cluster-set (.removePoints)))

(defn set-accumulation!
  [cluster-set distance-fn]
  (doto cluster-set (.setAccumulation distance-fn)))

(defn set-clusters!
  [cluster-set clusters]
  (doto cluster-set (.setClusters clusters)))

(defn set-point-distribution!
  [cluster-set point-dists]
  (doto cluster-set (.setPointDistribution point-dists)))
