(ns dl4clj.clustering.quadtree.quad-tree
  (:import [org.deeplearning4j.clustering.quadtree QuadTree])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-quad-tree
  [& {:keys [boundary data parent-tree]
   :as opts}]
  (assert (or (contains? opts :data)
              (contains? opts :boundary))
          "you must supply data or a cell to create a new tree")
  (cond (contains-many? opts :boundary :data :parent-tree)
        (QuadTree. parent-tree data boundary)
        (contains? opts :data)
        (QuadTree. data)
        :else
        (QuadTree. boundary)))

(defn compute-edge-forces!
  [& {:keys [tree row-p col-p val-p n pos-f]}]
  (doto tree (.computeEdgeForces row-p col-p val-p n pos-f)))

(defn compute-non-edge-forces!
  "Compute non edge forces using barnes hut"
  [& {:keys [tree point-idx theta negative-force sum-q]}]
  (doto tree (.computeNonEdgeForces point-idx theta negative-force sum-q)))

(defn get-depth
  [tree]
  (.depth tree))

(defn get-boundary
  [tree]
  (.getBoundary tree))

(defn get-center-of-mass
  [tree]
  (.getCenterOfMass tree))

(defn get-cumulative-size
  [tree]
  (.getCumSize tree))

(defn get-north-east
  [tree]
  (.getNorthEast tree))

(defn get-north-west
  [tree]
  (.getNorthWest tree))

(defn get-parent
  [tree]
  (.getParent tree))

(defn get-size
  [tree]
  (.getSize tree))

(defn get-south-east
  [tree]
  (.getSouthEast tree))

(defn set-south-west
  [tree]
  (.getSouthWest tree))

(defn insert-at-idx
  ;; this returns a boolean???
  [& {:keys [tree idx]}]
  (.insert tree idx))

(defn is-correct?
  "Returns whether the tree is consistent or not"
  [tree]
  (.isCorrect tree))

(defn is-leaf?
  [tree]
  (.isLeaf tree))

(defn set-boundary!
  [& {:keys [tree boundary]}]
  (doto tree (.setBoundary boundary)))

(defn set-center-of-mass!
  [& {:keys [tree center-of-mass]}]
  (doto tree (.setCenterOfMass center-of-mass)))

(defn set-cumulative-size!
  [& {:keys [tree size]}]
  (doto tree (.setCumSize size)))

(defn set-leaf!
  [& {:keys [tree leaf?]}]
  (doto tree (.setLeaf leaf?)))

(defn set-north-east!
  [& {:keys [tree other-tree]}]
  (doto tree (.setNorthEast other-tree)))

(defn set-north-west!
  [& {:keys [tree other-tree]}]
  (doto tree (.setNorthWest other-tree)))

(defn set-parent!
  [& {:keys [tree parent-tree]}]
  (doto tree (.setParent parent-tree)))

(defn set-size!
  [& {:keys [tree size]}]
  (doto tree (.setSize size)))

(defn set-south-east!
  [& {:keys [tree other-tree]}]
  (doto tree (.setSouthEast other-tree)))

(defn set-south-west!
  [& {:keys [tree other-tree]}]
  (doto tree (.setSouthWest other-tree)))

(defn sub-divide!
  [tree]
  (doto tree (.subDivide tree)))
