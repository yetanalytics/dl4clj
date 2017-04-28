(ns dl4clj.clustering.kmeans-clustering
  (:import [org.deeplearning4j.clustering.kmeans KMeansClustering])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.clustering.algorithm.base-clustering-algorithm :refer :all]))

(defn set-up
  [& {:keys [n-clusters distance-fn min-distribution-variation-rate
             allow-empty-clusters? max-iterations]
      :as opts}]
  (assert (contains-many? opts :n-clusters :distance-fn))
  (if (contains? opts :max-iteration-count)
    (.setup n-clusters max-iterations distance-fn)
    (.setup n-clusters min-distribution-variation-rate
            distance-fn allow-empty-clusters?)))
