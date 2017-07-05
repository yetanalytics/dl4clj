(ns dl4clj.clustering.kmeans-clustering
  (:import [org.deeplearning4j.clustering.kmeans KMeansClustering]
           [org.nd4j.linalg.api.ops.impl.accum.distances CosineSimilarity EuclideanDistance
            ManhattanDistance])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn set-up-kmeans
  "distance-fn is one of: 'cosinesimilarity', 'euclidean', 'manhattan'"
  [& {:keys [n-clusters distance-fn min-distribution-variation-rate
             allow-empty-clusters? max-iterations]
      :as opts}]
  (assert (contains-many? opts :n-clusters :distance-fn) "you must supply the number of clusters and a distance function")
  (if (contains? opts :max-iterations)
    (KMeansClustering/setup (int n-clusters) (int max-iterations) distance-fn)
    (KMeansClustering/setup (int n-clusters) (double min-distribution-variation-rate)
                            distance-fn allow-empty-clusters?)))
