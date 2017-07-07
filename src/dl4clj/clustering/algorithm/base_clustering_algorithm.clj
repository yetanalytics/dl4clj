(ns dl4clj.clustering.algorithm.base-clustering-algorithm
  (:import [org.deeplearning4j.clustering.algorithm BaseClusteringAlgorithm]))

(defn new-base-clustering-algorithm
  "creates a new instance of BaseClusteringAlgoirthm using the clusteringStrategy"
  [clustering-strat]
  (.setup clustering-strat))

(defn apply-to-points
  "applies a clustering algorithm to a collection of points"
  [& {:keys [cluster-algo points]}]
  (.applyTo cluster-algo points))

;; under the hood ^ does

;;1) reset-state

;; creates a new iteration history
;; sets the current iteration to 0
;; sets the inital cluster-set to nil
;; sets initial points to the ones supplied

;;2) init clusters

;; creates a new cluster set using
;; 1. the distance-fn (from the clustering strat)
;; 2. the result of inverseDistanceCalculation (a boolean)

;; sets a single center for the cluster (picks a random point from the list as the center)
;; uses add-new-cluster-with-center
;; defines the cluster-count from the clustering-strat

;; creates an array of distances between each point and the nearest cluster to those points
;; which is used to generate the initial cluster centers, by randomly selecting a point between 0 and max distance
;; call to compute-square-distances-from-nearest-cluster

;; adds a new cluster to the cluster set for each point
;; the points are randomly chosen based on their distance from the initial center

;; cluster-set-info is then created from the resulting cluster set with number of clusters = number of points
;; creates an iteration info to be stored as iterationHistory

;;3) iterations

;; get the termination condition from the clustering strat
;; while these things are true
;; checks to see if the term-cond is satisifed using the iteration history
;; and checks to make sure that the strategy is applied via the most recent iteration info
;; incs current iteration, calls remove points, classify points, apply clustering strat
;; - remove points removes the points from the cluster set, they were just used to set up the centers
;; - classifyPoints calls classify pionts using the cluster set and the initial points (used to create the clusters with centers)
;;   - it then refreshes the cluster centers, and creates a new iteration info obj and adds it to the iteration history
;; - apply clustering strat checks to see if there can be empty clusters, if not, removes all empty clusters
;;   - then checks the clustering strategy type.
;;     - if fixed-cluster-count, and clusters were removed, calls split-most-spread-out-clusters
;;     - if optimization, calls optimize

;; optimize calls applyOptimization using the optimizationStrategy and clustering strategy (from clustering utils)
;; - syntax for passing the optimization is odd: optimization = (OptimisationStrategy) clusteringStrategy
;; -- will need to look into OptimizationStrategy
;; the clustering-util fn checks to see the type of clustering  optimization
;; - either minimize-average-point-to-center-distance or minimize-maximum-point-to-center-distance
;; calls the appropriate split fn based on clustering optimization type
;; returns true if either of the split fns were called and the return value was greater than 0
;; otherwise returns false
