(ns dl4clj.clustering.kdtree.hyper-rectangle
  (:import [org.deeplearning4j.clustering.kdtree HyperRect HyperRect$Interval])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn new-hyper-rect-interval
  [& {:keys [^Double lower ^Double higher]}]
  (HyperRect$Interval. lower higher))

(defn new-hyper-rect
  "creates a new hyper rectangle given a list of intervals"
  [& {:keys [intervals]}]
  (HyperRect. intervals))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; interval api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn does-interval-contain?
  [& {:keys [interval ^Double point]}]
  (.contains interval point))

(defn enlarge-interval!
  "if the supplied value is less than the existing lower bound, the lower bound will be updated.

  if the supplied value is greater thant he existing upper bound, the upper bound will be updated."
  [& {:keys [interval ^Double higher-or-lower-val]}]
  (doto interval (.enlarge higher-or-lower-val)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; hyper-rect api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn does-rect-contain?
  "checks to see if the elements of a supplied vector are contained within
   the hyper rectangle

  points can be an existing INDarray or a vector of data"
  [& {:keys [hyper-rect points]}]
  (.contains hyper-rect (vec-or-matrix->indarray points)))

;; finish this
#_https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nearestneighbors-parent/nearestneighbor-core/src/main/java/org/deeplearning4j/clustering/kdtree/HyperRect.java
#_https://deeplearning4j.org/doc/org/deeplearning4j/clustering/kdtree/HyperRect.html
