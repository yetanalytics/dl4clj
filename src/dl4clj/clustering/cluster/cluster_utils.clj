(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/clustering/cluster/ClusterUtils.html
fns which require an executor service have not been implemented.  Algorithm fns use those behind the scenes...I think

also utils which just point to fns in other classes or are helper fns are not implemented.

see: https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nearestneighbors-parent/nearestneighbor-core/src/main/java/org/deeplearning4j/clustering/cluster/ClusterUtils.java"}
    dl4clj.clustering.cluster.cluster-utils
  (:import [org.deeplearning4j.clustering.cluster ClusterUtils])
  (:require [dl4clj.clustering.cluster.cluster-set-info :refer [get-clusters-info
                                                                get-cluster-info
                                                                set-clusters-info!
                                                                new-cluster-set-info-obj]]
            [dl4clj.clustering.cluster.cluster-set :refer [get-clusters]]
            [dl4clj.clustering.cluster.cluster :refer [get-cluster-id]]))

(defn derive-cluster-info-distance-stats
  "helper fn built into compute-cluster-(set)-info"
  [cluster-info]
  (doto cluster-info ClusterUtils/deriveClusterInfoDistanceStatistics))

(defn compute-cluster-infos!
  "creates a cluster-info object from the supplied cluster and distance fn.

  that cluster-info object is then passed to derive-cluster-info-distance-stats

  for actual calculation of the infos

  :cluster (cluster), a non empty cluster
   - see: dl4clj.clustering.cluster.cluster

  :distance-fn (str), a distance fn for calculating distances
   - one of: 'manhattan', 'euclidean' or 'cosinesimilarity'"
  [& {:keys [cluster distance-fn]}]
  (doto (ClusterUtils/computeClusterInfos cluster distance-fn)
    derive-cluster-info-distance-stats))

(defn compute-cluster-set-info!
  "given a cluster-set, creates a cluster set info object from it with all

  distance stats calculated for each cluster within the set.

  for creation of cluster sets, see: dl4clj.clustering.cluster.cluster-set

  returns the cluster-set-info object"
  [cluster-set]
  (let [cs-info (ClusterUtils/computeClusterSetInfo cluster-set)
        _ (dorun (for [each (get-clusters-info cs-info)
                       :let [[c-id c-info] each]]
                   (derive-cluster-info-distance-stats c-info)))]
    cs-info))

(defn refresh-cluster-center!
  "refreshes the center point of the supplied cluster  and returns that cluster

  :cluster (cluster), a non-empty cluster
   - see: dl4clj.clustering.cluster.cluster

  :cluster-info (cluster-info), the cluster info object for the supplied cluster
   - typicaly extracted from a cluster-set-info object or created using compute-cluster-infos!

   - see: dl4clj.clustering.cluster.cluster-set-info and dl4clj.clustering.cluster.cluster-info"
  [& {:keys [cluster cluster-info]}]
  (doto cluster (ClusterUtils/refreshClusterCenter cluster-info)))

(defn refresh-cluster-set-centers!
  "refreshes the center points of the clusters within the supplied cluster-set and

  calculates their distanct stats.

  :cluster-set (cluster-set), a set of clusters
   - see: dl4clj.clustering.cluster.cluster-set

  :cluster-set-info (cluster-set-info), an info object for the cluster set
   - see: compute-cluster-set-info!

  returns a map containing the cluster-set and cluster-set-info"
  [& {:keys [cluster-set cluster-set-info]}]
  (let [_ (dorun (for [each (get-clusters cluster-set)
                       ;; get-cluster-info is used over get-clusters-info to ensure
                       ;; we get the cluster-info for the current cluster
                       :let [info (get-cluster-info :cluster-set-info cluster-set-info
                                                    :cluster-id (get-cluster-id each))]]
                   (do
                     (refresh-cluster-center! :cluster each
                                              :cluster-info info)
                     (derive-cluster-info-distance-stats info))))]
    {:cluster-set cluster-set
     :cluster-set-info cluster-set-info}))

(defn get-most-spread-out-clusters
  "returns a vector of n clusters (from the cluster set) containing the most spread out points

  :cluster-set (cluster-set), a set of clusters
   - see: dl4clj.clustering.cluster.cluster-set

  :cluster-set-info (cluster-set-info), an info object for the cluster set
   - see: compute-cluster-set-info!

  :n (int), the number of clusters to return
   - must be equal to or less than the number of clusters in the cluster-set"
  [& {:keys [cluster-set cluster-set-info n]}]
  (ClusterUtils/getMostSpreadOutClusters cluster-set cluster-set-info n))

(defn get-clusters-where-max-distance-from-center-greater-than
  "returns a vector of clusters which contain a point that is further than max-distance away from the

  center point of the respective cluster

  :cluster-set (cluster-set), a set of clusters
   - see: dl4clj.clustering.cluster.cluster-set

  :cluster-set-info (cluster-set-info), an info object for the cluster set
   - see: compute-cluster-set-info!

  :max-distance (double), the max distance from the center point a point is allowed to be"
  [& {:keys [cluster-set cluster-set-info max-distance]}]
  (ClusterUtils/getClustersWhereMaximumDistanceFromCenterGreaterThan
   cluster-set cluster-set-info max-distance))

(defn get-clusters-where-avg-distance-from-center-greater-than
  "returns a vector of clusters which contain a point that is further than max-avg-distance away from the

  center point of the respective cluster

  :cluster-set (cluster-set), a set of clusters
   - see: dl4clj.clustering.cluster.cluster-set

  :cluster-set-info (cluster-set-info), an info object for the cluster set
   - see: compute-cluster-set-info!

  :max-avg-distance (double), the max average distance from the center point points are allowed to be"
  [& {:keys [cluster-set cluster-set-info max-avg-distance]}]
  (ClusterUtils/getClustersWhereAverageDistanceFromCenterGreaterThan
    cluster-set cluster-set-info max-avg-distance))
